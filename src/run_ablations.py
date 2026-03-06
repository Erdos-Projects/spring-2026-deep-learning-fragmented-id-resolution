import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


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
    parser.add_argument("--monitor-metric", choices=["f1", "pr_auc", "recall", "precision", "accuracy"], default="pr_auc")
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
        "--attribute-set",
        attribute_set,
        "--monitor-metric",
        args.monitor_metric,
        "--run-dir",
        str(run_dir),
    ]
    if args.split_path:
        cmd.extend(["--split-path", args.split_path])
    if args.disable_progress:
        cmd.append("--disable-progress")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for attribute_set in ["baseline", "extended"]:
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
