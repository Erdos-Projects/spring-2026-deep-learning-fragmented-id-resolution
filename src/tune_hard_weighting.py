import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd

try:
    from .data_utils import ATTRIBUTE_SET_NAMES
    from .dataset import PAIR_FEATURE_SET_NAMES
except ImportError:
    from data_utils import ATTRIBUTE_SET_NAMES
    from dataset import PAIR_FEATURE_SET_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep hard-example weighting settings for Siamese training.")
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--split-path", default="data/processed/splits/id_disjoint_seed42.tsv")
    parser.add_argument("--hard-example-path", required=True)
    parser.add_argument("--attribute-set", choices=ATTRIBUTE_SET_NAMES, default="extended")
    parser.add_argument("--pair-feature-set", choices=PAIR_FEATURE_SET_NAMES, default="none")
    parser.add_argument("--blocking-keys", default="first_name,age")
    parser.add_argument("--blocking-mode", choices=["all", "any"], default="any")
    parser.add_argument("--encoder", choices=["bilstm", "charcnn"], default="bilstm")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-weighting-modes", default="both")
    parser.add_argument("--weight-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--hard-positive-scales", default=None)
    parser.add_argument("--hard-negative-scales", default=None)
    parser.add_argument(
        "--monitor-metric",
        choices=[
            "pr_auc",
            "f1",
            "hard_subset_f1",
            "hard_positive_recall",
            "hard_negative_rejection_rate",
            "blended_score",
        ],
        default="blended_score",
    )
    parser.add_argument("--blended-overall-weight", type=float, default=0.5)
    parser.add_argument("--blended-hard-positive-weight", type=float, default=0.25)
    parser.add_argument("--blended-hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--run-root", default="models/experiments/hard_weight_tuning")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _parse_float_csv(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _build_command(args, run_dir, mode, pos_scale, neg_scale):
    command = [
        sys.executable,
        "src/train.py",
        "--data-path",
        args.data_path,
        "--dpl-path",
        args.dpl_path,
        "--ndpl-path",
        args.ndpl_path,
        "--split-path",
        args.split_path,
        "--attribute-set",
        args.attribute_set,
        "--pair-feature-set",
        args.pair_feature_set,
        "--blocking-keys",
        args.blocking_keys,
        "--blocking-mode",
        args.blocking_mode,
        "--encoder",
        args.encoder,
        "--max-len",
        str(args.max_len),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--hard-example-path",
        args.hard_example_path,
        "--hard-weighting",
        mode,
        "--hard-weight-scale",
        "1.0",
        "--hard-positive-weight-scale",
        str(pos_scale),
        "--hard-negative-weight-scale",
        str(neg_scale),
        "--monitor-metric",
        args.monitor_metric,
        "--blended-overall-weight",
        str(args.blended_overall_weight),
        "--blended-hard-positive-weight",
        str(args.blended_hard_positive_weight),
        "--blended-hard-negative-weight",
        str(args.blended_hard_negative_weight),
        "--run-dir",
        str(run_dir),
    ]
    if args.disable_progress:
        command.append("--disable-progress")
    return command


def _run_one(args, mode, pos_scale, neg_scale):
    run_dir = Path(args.run_root) / f"{mode}_pos{pos_scale:.2f}_neg{neg_scale:.2f}_{args.monitor_metric}"
    metrics_path = run_dir / "best_metrics.json"
    if args.reuse_existing and metrics_path.exists():
        payload = json.load(open(metrics_path, "r", encoding="utf-8"))
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(_build_command(args, run_dir, mode, pos_scale, neg_scale), check=True)
        payload = json.load(open(metrics_path, "r", encoding="utf-8"))

    test_metrics = payload["test_metrics"]
    hard_metrics = test_metrics["hard_example_metrics"]
    return {
        "mode": mode,
        "hard_positive_weight_scale": pos_scale,
        "hard_negative_weight_scale": neg_scale,
        "monitor_metric": args.monitor_metric,
        "best_epoch": payload["best_epoch"],
        "best_monitor_value": payload["best_monitor_value"],
        "test_f1": test_metrics["f1"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "hard_subset_f1": hard_metrics["hard_subset_metrics"].get("f1", float("nan")),
        "easy_subset_f1": hard_metrics["easy_subset_metrics"].get("f1", float("nan")),
        "hard_positive_recall": hard_metrics["hard_positive_recall"],
        "hard_negative_rejection_rate": hard_metrics["hard_negative_rejection_rate"],
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()
    modes = _parse_str_csv(args.hard_weighting_modes)
    default_scales = _parse_float_csv(args.weight_scales)
    positive_scales = _parse_float_csv(args.hard_positive_scales) if args.hard_positive_scales else default_scales
    negative_scales = _parse_float_csv(args.hard_negative_scales) if args.hard_negative_scales else default_scales

    rows = []
    for mode, pos_scale, neg_scale in product(modes, positive_scales, negative_scales):
        print(
            f"Running mode={mode} hard_positive_weight_scale={pos_scale:.2f} "
            f"hard_negative_weight_scale={neg_scale:.2f}"
        )
        rows.append(_run_one(args, mode, pos_scale, neg_scale))

    summary_df = pd.DataFrame(rows).sort_values(
        ["hard_subset_f1", "test_f1", "test_pr_auc"], ascending=[False, False, False]
    )
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_tsv = run_root / "tuning_summary.tsv"
    summary_json = run_root / "tuning_summary.json"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)

    print(f"Saved: {summary_tsv}")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
