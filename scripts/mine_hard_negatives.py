import argparse
from pathlib import Path

import pandas as pd

try:
    from src.data_utils import load_prepared_data
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_utils import load_prepared_data


DEFAULT_SAMPLE_COLUMNS = [
    "first_name",
    "last_name",
    "age",
    "sex",
    "house_num",
    "street_name",
    "zip_code",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard-negative buckets from NCVoters NDPL pairs.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--output-dir", default="data/processed/hard_negatives")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-columns",
        default=",".join(DEFAULT_SAMPLE_COLUMNS),
        help="Comma-separated columns to include in sampled examples.",
    )
    return parser.parse_args()


def load_ndpl_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"id1", "id2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"NDPL file missing required columns: {sorted(missing)}")
    df = df[["id1", "id2"]].copy()
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df.insert(0, "pair_id", range(len(df)))
    return df


def merge_pairs_with_records(pairs_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    left = data_df.add_suffix("_1").rename(columns={"id_1": "id1"})
    right = data_df.add_suffix("_2").rename(columns={"id_2": "id2"})
    merged = pairs_df.merge(left, on="id1", how="inner")
    merged = merged.merge(right, on="id2", how="inner")
    return merged


def exact_match(df: pd.DataFrame, column: str) -> pd.Series:
    c1 = f"{column}_1"
    c2 = f"{column}_2"
    return (df[c1] == df[c2]) & (df[c1] != "") & (df[c2] != "")


def exact_diff(df: pd.DataFrame, column: str) -> pd.Series:
    c1 = f"{column}_1"
    c2 = f"{column}_2"
    return (df[c1] != df[c2]) & (df[c1] != "") & (df[c2] != "")


def build_hard_negative_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    first_eq = exact_match(df, "first_name")
    last_eq = exact_match(df, "last_name")
    age_eq = exact_match(df, "age")
    age_diff = exact_diff(df, "age")
    sex_diff = exact_diff(df, "sex")
    house_eq = exact_match(df, "house_num")
    street_eq = exact_match(df, "street_name")
    zip_eq = exact_match(df, "zip_code")
    address_eq = house_eq & street_eq & zip_eq

    first_dist = pd.Series(
        [
            levenshtein(a, b)
            for a, b in zip(df["first_name_1"].astype(str), df["first_name_2"].astype(str))
        ],
        index=df.index,
    )
    last_dist = pd.Series(
        [
            levenshtein(a, b)
            for a, b in zip(df["last_name_1"].astype(str), df["last_name_2"].astype(str))
        ],
        index=df.index,
    )

    first_close = (first_dist <= 2) & (df["first_name_1"] != "") & (df["first_name_2"] != "")
    last_close = (last_dist <= 1) & (df["last_name_1"] != "") & (df["last_name_2"] != "")

    return {
        "same_first_last_diff_age": first_eq & last_eq & age_diff,
        "same_last_and_address": last_eq & address_eq,
        "same_first_and_age_diff_last": first_eq & age_eq & exact_diff(df, "last_name"),
        "same_address_diff_age_or_sex": address_eq & (age_diff | sex_diff),
        "near_first_same_last_address": first_close & last_eq & address_eq & ~first_eq,
        "same_full_name_address": first_eq & last_eq & address_eq,
        "same_full_name_zip_diff_address": first_eq & last_eq & zip_eq & ~address_eq,
        "same_first_last_zip_diff_age": first_eq & last_eq & zip_eq & age_diff,
        "same_first_age_zip_diff_last": first_eq & age_eq & zip_eq & exact_diff(df, "last_name"),
        "near_name_same_zip": first_close & last_close & zip_eq & ~(first_eq & last_eq),
    }


def summarize_masks(merged: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    total = len(merged)
    for bucket, mask in masks.items():
        count = int(mask.sum())
        rows.append(
            {
                "bucket": bucket,
                "pair_count": count,
                "pair_rate": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_count", "bucket"], ascending=[False, True])


def build_example_rows(
    merged: pd.DataFrame,
    masks: dict[str, pd.Series],
    sample_columns: list[str],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for bucket, mask in masks.items():
        subset = merged.loc[mask].copy()
        if subset.empty:
            continue
        sample = subset.sample(n=min(sample_size, len(subset)), random_state=seed)
        for _, row in sample.iterrows():
            example = {"bucket": bucket, "pair_id": row["pair_id"], "id1": row["id1"], "id2": row["id2"]}
            for col in sample_columns:
                example[f"{col}_1"] = row.get(f"{col}_1", "")
                example[f"{col}_2"] = row.get(f"{col}_2", "")
            rows.append(example)
    if not rows:
        return pd.DataFrame(columns=["bucket", "pair_id", "id1", "id2"])
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    sample_columns = [c.strip() for c in args.sample_columns.split(",") if c.strip()]

    data_df = load_prepared_data(args.data_path)
    ndpl_pairs = load_ndpl_pairs(Path(args.ndpl_path))
    merged = merge_pairs_with_records(ndpl_pairs, data_df)
    masks = build_hard_negative_masks(merged)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summarize_masks(merged, masks)
    summary_path = output_dir / "hard_negative_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    examples_df = build_example_rows(
        merged=merged,
        masks=masks,
        sample_columns=sample_columns,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    examples_path = output_dir / "hard_negative_examples.tsv"
    examples_df.to_csv(examples_path, sep="\t", index=False)

    print(f"Merged NDPL pairs: {len(merged)}")
    print("\nHard-negative bucket counts:")
    for _, row in summary_df.iterrows():
        print(f"- {row['bucket']}: {int(row['pair_count'])} ({row['pair_rate']:.2%})")

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved examples: {examples_path}")


if __name__ == "__main__":
    main()
