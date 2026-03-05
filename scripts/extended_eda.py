import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BLOCKING_KEYS = ["zip_code", "last_name", "first_name", "street_name", "age"]


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
    known_ids = set(data_df["id"])
    pairs = pairs[pairs["id1"].isin(known_ids) & pairs["id2"].isin(known_ids)].copy()
    left = data_df.add_suffix("_1").rename(columns={"id_1": "id1"})
    right = data_df.add_suffix("_2").rename(columns={"id_2": "id2"})
    merged = pairs.merge(left, on="id1", how="inner")
    merged = merged.merge(right, on="id2", how="inner")
    return merged


def key_match_mask(merged: pd.DataFrame, key: str) -> pd.Series:
    k1 = f"{key}_1"
    k2 = f"{key}_2"
    if k1 not in merged.columns or k2 not in merged.columns:
        return pd.Series([False] * len(merged), index=merged.index)
    # Treat blank values as non-matches so recall is not inflated by empty fields.
    return (merged[k1] == merged[k2]) & (merged[k1] != "") & (merged[k2] != "")


def analyze_blocking_recall(
    merged: pd.DataFrame,
    label: str,
    blocking_keys: list[str],
) -> None:
    print(f"\n--- Blocking Analysis ({label}) ---")
    total_pairs = len(merged)
    if total_pairs == 0:
        print("No merged pairs available.")
        return

    print(f"Merged pairs: {total_pairs}")
    masks: dict[str, pd.Series] = {}
    for key in blocking_keys:
        mask = key_match_mask(merged, key)
        masks[key] = mask
        matches = int(mask.sum())
        print(f"Key '{key}': {matches}/{total_pairs} ({matches / total_pairs:.2%})")

    if "first_name" in masks and "age" in masks:
        any_mask = masks["first_name"] | masks["age"]
        all_mask = masks["first_name"] & masks["age"]
        any_matches = int(any_mask.sum())
        all_matches = int(all_mask.sum())
        print(f"'first_name OR age': {any_matches}/{total_pairs} ({any_matches / total_pairs:.2%})")
        print(f"'first_name AND age': {all_matches}/{total_pairs} ({all_matches / total_pairs:.2%})")


def analyze_attribute_changes(merged: pd.DataFrame, label: str) -> None:
    print(f"\n--- Attribute Change Analysis ({label}) ---")
    if merged.empty:
        print("No merged pairs available.")
        return

    attributes = []
    for col in merged.columns:
        if col.endswith("_1"):
            attr = col[:-2]
            if attr != "id" and f"{attr}_2" in merged.columns:
                attributes.append(attr)

    rows: list[dict[str, float | str]] = []
    total = float(len(merged))
    for attr in sorted(set(attributes)):
        c1 = f"{attr}_1"
        c2 = f"{attr}_2"
        both_filled = (merged[c1] != "") & (merged[c2] != "")
        rows.append(
            {
                "attribute": attr,
                "match_pct": (((merged[c1] == merged[c2]) & both_filled).sum()) / total,
                "diff_pct": (((merged[c1] != merged[c2]) & both_filled).sum()) / total,
                "one_empty_pct": (((merged[c1] == "") ^ (merged[c2] == "")).sum()) / total,
                "both_empty_pct": (((merged[c1] == "") & (merged[c2] == "")).sum()) / total,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("match_pct", ascending=False)
    print(result_df.to_string(index=False))


def analyze_string_lengths(data_df: pd.DataFrame, columns: list[str]) -> None:
    print("\n--- Serialized String Length Analysis ---")
    valid_cols = [c for c in columns if c in data_df.columns]
    if not valid_cols:
        print("No valid columns selected for serialization length analysis.")
        return

    def serialize(row: pd.Series) -> str:
        parts = [f"{col}:{row[col]}" for col in valid_cols if row[col] != ""]
        return " ".join(parts)

    lengths = data_df.apply(lambda row: len(serialize(row)), axis=1)
    print(f"Columns used: {','.join(valid_cols)}")
    print(f"Mean length: {lengths.mean():.2f}")
    print(f"Median length: {lengths.median():.2f}")
    print(f"95th percentile: {lengths.quantile(0.95):.2f}")
    print(f"Max length: {int(lengths.max())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended EDA for NCVoters duplicate detection.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument(
        "--blocking-keys",
        default=",".join(DEFAULT_BLOCKING_KEYS),
        help="Comma-separated blocking keys for match-rate analysis.",
    )
    parser.add_argument(
        "--serialization-columns",
        default="first_name,last_name,house_num,street_name,zip_code,age,sex,race_desc,ethnic_desc",
        help="Comma-separated columns for serialized length analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocking_keys = [k.strip() for k in args.blocking_keys.split(",") if k.strip()]
    serialization_columns = [c.strip() for c in args.serialization_columns.split(",") if c.strip()]

    data_df = load_prepared_data(Path(args.data_path))
    dpl_pairs = load_pairs(Path(args.dpl_path))
    ndpl_pairs = load_pairs(Path(args.ndpl_path))

    dpl_merged = merge_pairs_with_records(dpl_pairs, data_df)
    ndpl_merged = merge_pairs_with_records(ndpl_pairs, data_df)

    analyze_blocking_recall(dpl_merged, "DPL", blocking_keys)
    analyze_blocking_recall(ndpl_merged, "NDPL", blocking_keys)
    analyze_attribute_changes(dpl_merged, "DPL")
    analyze_attribute_changes(ndpl_merged, "NDPL")
    analyze_string_lengths(data_df, serialization_columns)


if __name__ == "__main__":
    main()
