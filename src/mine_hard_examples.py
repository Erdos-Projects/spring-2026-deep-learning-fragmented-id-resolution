import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .hard_examples import mine_hard_examples, summarize_hard_examples_for_report
except ImportError:
    from hard_examples import mine_hard_examples, summarize_hard_examples_for_report


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard positives and hard negatives from real labeled pairs.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--hard-positive-threshold", type=float, default=None)
    parser.add_argument("--hard-negative-threshold", type=float, default=None)
    parser.add_argument("--hard-positive-quantile", type=float, default=0.85)
    parser.add_argument("--hard-negative-quantile", type=float, default=0.99)
    parser.add_argument("--output-dir", default="data/processed/hard_examples")
    parser.add_argument("--output-prefix", default="labeled_hard_examples")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_df = mine_hard_examples(
        data_path=args.data_path,
        dpl_path=args.dpl_path,
        ndpl_path=args.ndpl_path,
        hard_positive_threshold=args.hard_positive_threshold,
        hard_negative_threshold=args.hard_negative_threshold,
        hard_positive_quantile=args.hard_positive_quantile,
        hard_negative_quantile=args.hard_negative_quantile,
    )

    main_path = output_dir / f"{args.output_prefix}.tsv"
    hard_df.to_csv(main_path, sep="\t", index=False)

    summary_payload = summarize_hard_examples_for_report(hard_df)
    summary_payload.update(
        {
            "hard_positive_threshold": float(hard_df["hard_positive_threshold"].iloc[0]),
            "hard_negative_threshold": float(hard_df["hard_negative_threshold"].iloc[0]),
            "hard_positive_quantile": args.hard_positive_quantile,
            "hard_negative_quantile": args.hard_negative_quantile,
            "paths": {"artifact": str(main_path)},
        }
    )
    summary_json = output_dir / f"{args.output_prefix}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    reason_rows = summary_payload["top_reasons"]
    pd.DataFrame(reason_rows).to_csv(
        output_dir / f"{args.output_prefix}_top_reasons.tsv", sep="\t", index=False
    )
    hard_df.loc[hard_df["hard_example"].astype(bool)].head(200).to_csv(
        output_dir / f"{args.output_prefix}_examples.tsv", sep="\t", index=False
    )

    print(f"Saved hard-example artifact: {main_path}")
    print(
        f"Hard examples: {summary_payload['hard_example_count']} "
        f"(hard positives={summary_payload['hard_positive_count']}, "
        f"hard negatives={summary_payload['hard_negative_count']})"
    )


if __name__ == "__main__":
    main()
