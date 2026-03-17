#!/usr/bin/env python3
"""
Train and evaluate the classical supervised baseline on labeled pair data.
"""

import argparse

import pandas as pd

from src.baseline import evaluate_pair_classifier, save_pair_classifier, train_pair_classifier
from src.label_construction import build_silver_groups, load_existing_pairs, validate_gold_groups
from src.pair_generation import (
    combine_pairs,
    entity_level_split,
    generate_block_negatives,
    generate_demographic_negatives,
    generate_gold_positive_pairs,
    generate_hard_negative_pairs,
    generate_name_collision_negatives,
    generate_near_miss_negatives,
    pairs_from_dpl,
    pairs_from_ndpl,
)
from src.preprocessing import normalize_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="ncvoters.tsv", help="Path to raw voter records TSV/CSV")
    parser.add_argument("--dpl", default="ncvoters_DPL.tsv", help="Path to duplicate pair list TSV")
    parser.add_argument("--ndpl", default="ncvoters_NDPL.tsv", help="Path to non-duplicate pair list TSV")
    parser.add_argument(
        "--model-type",
        choices=["logreg", "hgbt"],
        default="logreg",
        help="Classical baseline type",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out entity split fraction")
    parser.add_argument("--ndpl-max-pairs", type=int, default=10_000, help="Max NDPL negatives to use")
    parser.add_argument("--output", default="baseline_model.joblib", help="Path to save trained model bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sep = "\t" if args.records.endswith((".tsv", ".txt")) else ","
    raw_df = pd.read_csv(args.records, sep=sep, dtype=str, na_filter=False, encoding="utf-8")
    raw_df = raw_df.loc[:, ~raw_df.columns.str.contains(r"^Unnamed")]

    records = build_silver_groups(normalize_dataframe(raw_df))
    gold_df, conflict_df, stats = validate_gold_groups(records)
    dpl, ndpl = load_existing_pairs(args.dpl, args.ndpl)

    all_pairs = combine_pairs(
        generate_gold_positive_pairs(gold_df),
        generate_hard_negative_pairs(conflict_df),
        generate_block_negatives(records, n_negatives=5000),
        generate_name_collision_negatives(records, n_negatives=3000),
        generate_near_miss_negatives(records, n_negatives=2000, min_shared_fields=4),
        generate_demographic_negatives(records, n_negatives=5000),
        pairs_from_dpl(dpl),
        pairs_from_ndpl(ndpl, max_pairs=args.ndpl_max_pairs),
    )
    train_pairs, test_pairs = entity_level_split(all_pairs, records, test_size=args.test_size)

    print(f"Records: {len(records):,}")
    print(f"Gold groups: {stats['n_gold_groups']:,} | Conflict groups: {stats['n_conflict_groups']:,}")
    print(
        f"Train pairs: {len(train_pairs):,} "
        f"(pos={(train_pairs['label'] == 1).sum():,}, neg={(train_pairs['label'] == 0).sum():,})"
    )
    print(
        f"Test pairs:  {len(test_pairs):,} "
        f"(pos={(test_pairs['label'] == 1).sum():,}, neg={(test_pairs['label'] == 0).sum():,})"
    )

    bundle = train_pair_classifier(train_pairs, records, model_type=args.model_type)
    metrics = evaluate_pair_classifier(bundle, test_pairs, records)
    save_pair_classifier(bundle, args.output)

    print(f"\nModel type: {args.model_type}")
    print(f"Saved model: {args.output}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:10s}: {value:.4f}")


if __name__ == "__main__":
    main()
