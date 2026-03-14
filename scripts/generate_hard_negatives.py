import argparse
import itertools
from pathlib import Path

import pandas as pd

try:
    from src.data_utils import load_prepared_data
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_utils import load_prepared_data


DEFAULT_EXPORT_COLUMNS = [
    "first_name",
    "last_name",
    "age",
    "sex",
    "house_num",
    "street_name",
    "zip_code",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate heuristic hard-negative candidate pairs from NCVoters."
    )
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--output-dir", default="data/processed/generated_hard_negatives")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-group-size", type=int, default=50)
    parser.add_argument("--status-filter", choices=["all", "unlabeled", "known_ndpl"], default="all")
    parser.add_argument(
        "--export-columns",
        default=",".join(DEFAULT_EXPORT_COLUMNS),
        help="Comma-separated columns to include in pair exports.",
    )
    parser.add_argument("--min-age-gap-generational", type=int, default=8)
    parser.add_argument("--max-age-gap-same-person-lookalike", type=int, default=1)
    return parser.parse_args()


def load_pair_set(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"id1", "id2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pair file {path} missing required columns: {sorted(missing)}")
    return {
        canonical_pair(str(row.id1), str(row.id2))
        for row in df[["id1", "id2"]].itertuples(index=False)
    }


def canonical_pair(id1: str, id2: str) -> str:
    return "||".join(sorted((str(id1), str(id2))))


def non_empty(row: dict, column: str) -> bool:
    return row.get(column, "") != ""


def exact_match(row1: dict, row2: dict, column: str) -> bool:
    return non_empty(row1, column) and non_empty(row2, column) and row1[column] == row2[column]


def exact_diff(row1: dict, row2: dict, column: str) -> bool:
    return non_empty(row1, column) and non_empty(row2, column) and row1[column] != row2[column]


def same_address(row1: dict, row2: dict) -> bool:
    return (
        exact_match(row1, row2, "house_num")
        and exact_match(row1, row2, "street_name")
        and exact_match(row1, row2, "zip_code")
    )


def different_address(row1: dict, row2: dict) -> bool:
    address_cols = ["house_num", "street_name", "zip_code"]
    return any(exact_diff(row1, row2, col) for col in address_cols)


def age_gap(row1: dict, row2: dict):
    if not non_empty(row1, "age") or not non_empty(row2, "age"):
        return None
    try:
        return abs(int(row1["age"]) - int(row2["age"]))
    except ValueError:
        return None


def name_distances(row1: dict, row2: dict) -> tuple[int | None, int | None]:
    first_dist = None
    last_dist = None
    if non_empty(row1, "first_name") and non_empty(row2, "first_name"):
        first_dist = levenshtein(row1["first_name"], row2["first_name"])
    if non_empty(row1, "last_name") and non_empty(row2, "last_name"):
        last_dist = levenshtein(row1["last_name"], row2["last_name"])
    return first_dist, last_dist


def build_bucket_specs(args):
    return [
        {
            "bucket": "same_full_name_address_diff_age",
            "group_keys": ["first_name", "last_name", "house_num", "street_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "first_name")
                and exact_match(r1, r2, "last_name")
                and same_address(r1, r2)
                and (age_gap(r1, r2) is not None and age_gap(r1, r2) >= args.min_age_gap_generational)
            ),
        },
        {
            "bucket": "same_full_name_address_diff_sex",
            "group_keys": ["first_name", "last_name", "house_num", "street_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "first_name")
                and exact_match(r1, r2, "last_name")
                and same_address(r1, r2)
                and exact_diff(r1, r2, "sex")
            ),
        },
        {
            "bucket": "same_last_address_same_age_diff_first",
            "group_keys": ["last_name", "house_num", "street_name", "zip_code", "age"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "last_name")
                and same_address(r1, r2)
                and exact_match(r1, r2, "age")
                and exact_diff(r1, r2, "first_name")
            ),
        },
        {
            "bucket": "same_last_address_close_first",
            "group_keys": ["last_name", "house_num", "street_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "last_name")
                and same_address(r1, r2)
                and non_empty(r1, "first_name")
                and non_empty(r2, "first_name")
                and r1["first_name"] != r2["first_name"]
                and levenshtein(r1["first_name"], r2["first_name"]) <= 2
            ),
        },
        {
            "bucket": "same_first_age_zip_diff_last",
            "group_keys": ["first_name", "age", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "first_name")
                and exact_match(r1, r2, "age")
                and exact_match(r1, r2, "zip_code")
                and exact_diff(r1, r2, "last_name")
            ),
        },
        {
            "bucket": "same_first_last_zip_same_age_diff_address",
            "group_keys": ["first_name", "last_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "first_name")
                and exact_match(r1, r2, "last_name")
                and exact_match(r1, r2, "zip_code")
                and (age_gap(r1, r2) is not None and age_gap(r1, r2) <= args.max_age_gap_same_person_lookalike)
                and different_address(r1, r2)
            ),
        },
        {
            "bucket": "same_address_diff_age_or_sex_close_name",
            "group_keys": ["house_num", "street_name", "zip_code"],
            "predicate": lambda r1, r2: (
                same_address(r1, r2)
                and (
                    (name_distances(r1, r2)[0] is not None and name_distances(r1, r2)[0] <= 2)
                    or exact_match(r1, r2, "last_name")
                )
                and (
                    (age_gap(r1, r2) is not None and age_gap(r1, r2) >= 5)
                    or exact_diff(r1, r2, "sex")
                )
            ),
        },
        {
            "bucket": "same_full_name_zip_diff_address_large_age_gap",
            "group_keys": ["first_name", "last_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "first_name")
                and exact_match(r1, r2, "last_name")
                and exact_match(r1, r2, "zip_code")
                and different_address(r1, r2)
                and (age_gap(r1, r2) is not None and age_gap(r1, r2) >= args.min_age_gap_generational)
            ),
        },
        {
            "bucket": "same_last_address_small_age_gap_diff_first",
            "group_keys": ["last_name", "house_num", "street_name", "zip_code"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "last_name")
                and same_address(r1, r2)
                and exact_diff(r1, r2, "first_name")
                and (age_gap(r1, r2) is not None and age_gap(r1, r2) <= 5)
            ),
        },
        {
            "bucket": "near_full_name_same_zip_same_age",
            "group_keys": ["zip_code", "age"],
            "predicate": lambda r1, r2: (
                exact_match(r1, r2, "zip_code")
                and exact_match(r1, r2, "age")
                and non_empty(r1, "first_name")
                and non_empty(r2, "first_name")
                and non_empty(r1, "last_name")
                and non_empty(r2, "last_name")
                and levenshtein(r1["first_name"], r2["first_name"]) <= 1
                and levenshtein(r1["last_name"], r2["last_name"]) <= 1
                and not (exact_match(r1, r2, "first_name") and exact_match(r1, r2, "last_name"))
            ),
        },
    ]


def build_candidate_row(pair_key, row1, row2, export_columns):
    first_dist, last_dist = name_distances(row1, row2)
    gap = age_gap(row1, row2)
    id1, id2 = pair_key.split("||")
    export = {
        "id1": id1,
        "id2": id2,
        "age_gap": gap,
        "first_name_distance": first_dist,
        "last_name_distance": last_dist,
        "same_address": same_address(row1, row2),
        "same_first_name": exact_match(row1, row2, "first_name"),
        "same_last_name": exact_match(row1, row2, "last_name"),
        "same_age": exact_match(row1, row2, "age"),
        "same_sex": exact_match(row1, row2, "sex"),
    }
    for col in export_columns:
        export[f"{col}_1"] = row1.get(col, "")
        export[f"{col}_2"] = row2.get(col, "")
    return export


def collect_bucket_candidates(
    data_df,
    bucket_spec,
    existing_rows,
    max_group_size,
    export_columns,
):
    group_keys = bucket_spec["group_keys"]
    selected_columns = ["id"]
    for col in group_keys + export_columns:
        if col not in selected_columns:
            selected_columns.append(col)
    working = data_df[selected_columns].copy()
    for key in set(group_keys):
        working = working[working[key] != ""]

    skipped_groups = 0
    hit_count = 0
    for _, group in working.groupby(group_keys, sort=False):
        if len(group) < 2:
            continue
        if len(group) > max_group_size:
            skipped_groups += 1
            continue

        records = group.to_dict(orient="records")
        for row1, row2 in itertools.combinations(records, 2):
            if not bucket_spec["predicate"](row1, row2):
                continue
            pair_key = canonical_pair(row1["id"], row2["id"])
            if pair_key not in existing_rows:
                existing_rows[pair_key] = build_candidate_row(pair_key, row1, row2, export_columns)
                existing_rows[pair_key]["buckets"] = set()
            existing_rows[pair_key]["buckets"].add(bucket_spec["bucket"])
            hit_count += 1
    return hit_count, skipped_groups


def assign_pair_status(pair_key, dpl_pairs, ndpl_pairs):
    if pair_key in dpl_pairs:
        return "known_dpl"
    if pair_key in ndpl_pairs:
        return "known_ndpl"
    return "unlabeled"


def build_output_frames(pair_rows, dpl_pairs, ndpl_pairs):
    rows = []
    summary_rows = []
    for pair_key, payload in pair_rows.items():
        status = assign_pair_status(pair_key, dpl_pairs, ndpl_pairs)
        row = dict(payload)
        row["pair_key"] = pair_key
        row["pair_status"] = status
        row["bucket_count"] = len(payload["buckets"])
        row["buckets"] = ",".join(sorted(payload["buckets"]))
        rows.append(row)
        for bucket in payload["buckets"]:
            summary_rows.append({"bucket": bucket, "pair_status": status, "pair_key": pair_key})

    pair_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = (
            summary_df.groupby(["bucket", "pair_status"], as_index=False)["pair_key"]
            .nunique()
            .rename(columns={"pair_key": "pair_count"})
            .sort_values(["bucket", "pair_status"])
        )
    return pair_df, summary_df


def sample_examples(pair_df, sample_size, seed):
    if pair_df.empty:
        return pair_df
    exploded = pair_df.assign(bucket=pair_df["buckets"].str.split(",")).explode("bucket")
    parts = []
    for bucket, subset in exploded.groupby("bucket", sort=False):
        parts.append(subset.sample(n=min(sample_size, len(subset)), random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=pair_df.columns)


def main():
    args = parse_args()
    export_columns = [c.strip() for c in args.export_columns.split(",") if c.strip()]

    data_df = load_prepared_data(args.data_path)
    dpl_pairs = load_pair_set(Path(args.dpl_path))
    ndpl_pairs = load_pair_set(Path(args.ndpl_path))

    pair_rows = {}
    generation_rows = []
    for bucket_spec in build_bucket_specs(args):
        hit_count, skipped_groups = collect_bucket_candidates(
            data_df=data_df,
            bucket_spec=bucket_spec,
            existing_rows=pair_rows,
            max_group_size=args.max_group_size,
            export_columns=export_columns,
        )
        generation_rows.append(
            {
                "bucket": bucket_spec["bucket"],
                "raw_bucket_hits": hit_count,
                "skipped_large_groups": skipped_groups,
            }
        )

    pair_df, summary_df = build_output_frames(pair_rows, dpl_pairs, ndpl_pairs)
    pair_df = pair_df[pair_df["pair_status"] != "known_dpl"].copy()
    if args.status_filter != "all":
        pair_df = pair_df[pair_df["pair_status"] == args.status_filter].copy()

    if not pair_df.empty:
        allowed_keys = set(pair_df["pair_key"])
        summary_df = summary_df[summary_df["pair_key"].isin(allowed_keys)] if "pair_key" in summary_df.columns else summary_df

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_df = pd.DataFrame(generation_rows).sort_values("raw_bucket_hits", ascending=False)
    generation_df.to_csv(output_dir / "generation_rule_summary.tsv", sep="\t", index=False)

    pair_df = pair_df.sort_values(["bucket_count", "pair_status", "id1", "id2"], ascending=[False, True, True, True])
    pair_df.to_csv(output_dir / "generated_hard_negative_candidates.tsv", sep="\t", index=False)

    if not summary_df.empty:
        summary_df = (
            pair_df.assign(bucket=pair_df["buckets"].str.split(","))
            .explode("bucket")
            .groupby(["bucket", "pair_status"], as_index=False)["pair_key"]
            .nunique()
            .rename(columns={"pair_key": "pair_count"})
            .sort_values(["bucket", "pair_status"])
        )
    summary_df.to_csv(output_dir / "generated_hard_negative_summary.tsv", sep="\t", index=False)

    example_df = sample_examples(pair_df, sample_size=args.sample_size, seed=args.seed)
    example_df.to_csv(output_dir / "generated_hard_negative_examples.tsv", sep="\t", index=False)

    print(f"Generated candidate pairs (after filtering): {len(pair_df)}")
    print("Rule hit summary:")
    for _, row in generation_df.iterrows():
        print(
            f"- {row['bucket']}: raw_hits={int(row['raw_bucket_hits'])}, "
            f"skipped_large_groups={int(row['skipped_large_groups'])}"
        )
    if not pair_df.empty:
        status_counts = pair_df["pair_status"].value_counts().to_dict()
        print(f"Status counts: {status_counts}")

    print(f"Saved rule summary: {output_dir / 'generation_rule_summary.tsv'}")
    print(f"Saved candidates: {output_dir / 'generated_hard_negative_candidates.tsv'}")
    print(f"Saved summary: {output_dir / 'generated_hard_negative_summary.tsv'}")
    print(f"Saved examples: {output_dir / 'generated_hard_negative_examples.tsv'}")


if __name__ == "__main__":
    main()
