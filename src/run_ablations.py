import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from .data_utils import ATTRIBUTE_SET_NAMES
except ImportError:
    from data_utils import ATTRIBUTE_SET_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline vs extended attribute ablation.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--split-strategy", choices=["pair_random", "id_disjoint"], default="id_disjoint")
    parser.add_argument("--split-path", default=None, help="Use existing split file for both runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder", choices=["bilstm", "charcnn"], default="bilstm")
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--cnn-kernel-sizes", default="3,4,5")
    parser.add_argument("--classifier-hidden-dim", type=int, default=64)
    parser.add_argument("--pair-feature-set", choices=["none", "sex_aware_name"], default="none")
    parser.add_argument(
        "--monitor-metric",
        choices=[
            "f1",
            "pr_auc",
            "recall",
            "precision",
            "accuracy",
            "hard_subset_f1",
            "hard_positive_recall",
            "hard_negative_rejection_rate",
            "blended_score",
        ],
        default="pr_auc",
    )
    parser.add_argument(
        "--attribute-sets",
        default="baseline,extended",
        help=f"Comma-separated attribute presets to compare. Available: {', '.join(ATTRIBUTE_SET_NAMES)}",
    )
    parser.add_argument("--blocking-keys", default="none")
    parser.add_argument("--blocking-mode", choices=["all", "any"], default="any")
    parser.add_argument("--hard-example-path", default=None)
    parser.add_argument("--hard-weighting", choices=["none", "loss", "sampler", "both"], default="none")
    parser.add_argument("--hard-weight-scale", type=float, default=1.0)
    parser.add_argument("--hard-positive-weight-scale", type=float, default=None)
    parser.add_argument("--hard-negative-weight-scale", type=float, default=None)
    parser.add_argument("--blended-overall-weight", type=float, default=0.5)
    parser.add_argument("--blended-hard-positive-weight", type=float, default=0.25)
    parser.add_argument("--blended-hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--run-root", default="models/runs/ablations")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def run_single(attribute_set, args, run_dir):
    cmd = [
        sys.executable,
        "src/train.py",
        "--data-path",
        args.data_path,
        "--dpl-path",
        args.dpl_path,
        "--ndpl-path",
        args.ndpl_path,
        "--split-strategy",
        args.split_strategy,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--max-len",
        str(args.max_len),
        "--embedding-dim",
        str(args.embedding_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--encoder",
        args.encoder,
        "--cnn-channels",
        str(args.cnn_channels),
        "--cnn-kernel-sizes",
        args.cnn_kernel_sizes,
        "--classifier-hidden-dim",
        str(args.classifier_hidden_dim),
        "--pair-feature-set",
        args.pair_feature_set,
        "--attribute-set",
        attribute_set,
        "--monitor-metric",
        args.monitor_metric,
        "--blocking-keys",
        args.blocking_keys,
        "--blocking-mode",
        args.blocking_mode,
        "--run-dir",
        str(run_dir),
    ]
    if args.split_path:
        cmd.extend(["--split-path", args.split_path])
    if args.hard_example_path:
        cmd.extend(["--hard-example-path", args.hard_example_path])
    if args.hard_weighting != "none":
        cmd.extend(["--hard-weighting", args.hard_weighting])
        cmd.extend(["--hard-weight-scale", str(args.hard_weight_scale)])
    if args.hard_positive_weight_scale is not None:
        cmd.extend(["--hard-positive-weight-scale", str(args.hard_positive_weight_scale)])
    if args.hard_negative_weight_scale is not None:
        cmd.extend(["--hard-negative-weight-scale", str(args.hard_negative_weight_scale)])
    cmd.extend(["--blended-overall-weight", str(args.blended_overall_weight)])
    cmd.extend(["--blended-hard-positive-weight", str(args.blended_hard_positive_weight)])
    cmd.extend(["--blended-hard-negative-weight", str(args.blended_hard_negative_weight)])
    if args.disable_progress:
        cmd.append("--disable-progress")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    attribute_sets = [item.strip() for item in str(args.attribute_sets).split(",") if item.strip()]
    invalid_sets = [item for item in attribute_sets if item not in ATTRIBUTE_SET_NAMES]
    if invalid_sets:
        raise ValueError(
            f"Unknown attribute set(s): {', '.join(invalid_sets)}. Available: {', '.join(ATTRIBUTE_SET_NAMES)}"
        )

    rows = []
    for attribute_set in attribute_sets:
        run_dir = run_root / attribute_set
        run_single(attribute_set, args, run_dir)

        with open(run_dir / "best_metrics.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "attribute_set": attribute_set,
                "best_epoch": payload["best_epoch"],
                "best_monitor_value": payload["best_monitor_value"],
                "best_val_f1": payload["best_val_metrics"]["f1"],
                "best_val_pr_auc": payload["best_val_metrics"]["pr_auc"],
                "test_f1": payload["test_metrics"]["f1"],
                "test_pr_auc": payload["test_metrics"]["pr_auc"],
                "test_roc_auc": payload["test_metrics"]["roc_auc"],
                "checkpoint": payload["paths"]["checkpoint"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("best_monitor_value", ascending=False)
    summary_tsv = run_root / "ablation_summary.tsv"
    summary_json = run_root / "ablation_summary.json"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    winner = summary_df.iloc[0]
    print("Ablation complete.")
    print(f"Winner by {args.monitor_metric}: {winner['attribute_set']}")
    print(f"Summary TSV: {summary_tsv}")


if __name__ == "__main__":
    main()
