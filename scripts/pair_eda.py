import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SAMPLE_COLUMNS = [
    "first_name",
    "last_name",
    "house_num",
    "street_name",
    "zip_code",
    "age",
    "sex",
    "race_desc",
    "ethnic_desc",
]


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def load_prepared_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "id" not in df.columns:
        raise ValueError(f"Prepared data is missing required column 'id': {path}")
    df["id"] = df["id"].astype(str)
    return df


def load_pairs(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"id1", "id2"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Pair file {path} is missing required columns: {sorted(missing)}")
    pairs = pairs[["id1", "id2"]].copy()
    pairs["id1"] = pairs["id1"].astype(str)
    pairs["id2"] = pairs["id2"].astype(str)
    return pairs


def merge_pairs_with_records(pairs: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    left = data_df.add_suffix("_1").rename(columns={"id_1": "id1"})
    right = data_df.add_suffix("_2").rename(columns={"id_2": "id2"})
    merged = pairs.merge(left, on="id1", how="inner")
    merged = merged.merge(right, on="id2", how="inner")
    return merged


def print_samples(
    merged: pd.DataFrame,
    sample_size: int,
    sample_columns: Iterable[str],
    seed: int,
) -> None:
    if merged.empty:
        print("No sample rows to print.")
        return

    sample = merged.sample(n=min(sample_size, len(merged)), random_state=seed)
    print("\nSample pairs:")
    for _, row in sample.iterrows():
        print(f"Pair: {row['id1']} - {row['id2']}")
        for column in sample_columns:
            c1 = f"{column}_1"
            c2 = f"{column}_2"
            if c1 in merged.columns and c2 in merged.columns:
                print(f"  {column}: {row[c1]}  |  {row[c2]}")
        print("-" * 40)


def analyze_pairs(
    label: str,
    pairs: pd.DataFrame,
    data_df: pd.DataFrame,
    sample_size: int,
    sample_columns: list[str],
    seed: int,
) -> set[str]:
    print(f"\n--- {label} ---")
    print(f"Total pairs in file: {len(pairs)}")

    ids_in_pairs = set(pairs["id1"]).union(set(pairs["id2"]))
    print(f"Unique IDs in pairs: {len(ids_in_pairs)}")

    known_ids = set(data_df["id"])
    missing_id1 = (~pairs["id1"].isin(known_ids)).sum()
    missing_id2 = (~pairs["id2"].isin(known_ids)).sum()
    print(f"Missing IDs in prepared data: id1={int(missing_id1)}, id2={int(missing_id2)}")

    valid_pairs = pairs[pairs["id1"].isin(known_ids) & pairs["id2"].isin(known_ids)].copy()
    print(f"Pairs with both IDs present: {len(valid_pairs)}")

    merged = merge_pairs_with_records(valid_pairs, data_df)
    print(f"Merged pairs: {len(merged)}")

    if "first_name_1" in merged.columns and "first_name_2" in merged.columns:
        first_dist = [
            levenshtein(a, b)
            for a, b in zip(merged["first_name_1"].astype(str), merged["first_name_2"].astype(str))
        ]
        print(f"Mean first_name Levenshtein distance: {sum(first_dist) / len(first_dist):.2f}")

    if "last_name_1" in merged.columns and "last_name_2" in merged.columns:
        last_dist = [
            levenshtein(a, b)
            for a, b in zip(merged["last_name_1"].astype(str), merged["last_name_2"].astype(str))
        ]
        print(f"Mean last_name Levenshtein distance: {sum(last_dist) / len(last_dist):.2f}")

    print_samples(
        merged=merged,
        sample_size=sample_size,
        sample_columns=sample_columns,
        seed=seed,
    )
    return ids_in_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair-level EDA for DPL and NDPL files.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-columns",
        default=",".join(DEFAULT_SAMPLE_COLUMNS),
        help="Comma-separated column names to display for sampled pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_columns = [c.strip() for c in args.sample_columns.split(",") if c.strip()]

    data_df = load_prepared_data(Path(args.data_path))
    dpl_pairs = load_pairs(Path(args.dpl_path))
    ndpl_pairs = load_pairs(Path(args.ndpl_path))

    dpl_ids = analyze_pairs(
        label="Duplicate pairs (DPL)",
        pairs=dpl_pairs,
        data_df=data_df,
        sample_size=args.sample_size,
        sample_columns=sample_columns,
        seed=args.seed,
    )
    ndpl_ids = analyze_pairs(
        label="Non-duplicate pairs (NDPL)",
        pairs=ndpl_pairs,
        data_df=data_df,
        sample_size=args.sample_size,
        sample_columns=sample_columns,
        seed=args.seed,
    )

    overlap = dpl_ids.intersection(ndpl_ids)
    print(f"\nIDs appearing in both DPL and NDPL: {len(overlap)}")


if __name__ == "__main__":
    main()
