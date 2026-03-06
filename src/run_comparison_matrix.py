import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run apples-to-apples Siamese vs TF-IDF comparisons across attribute sets and blocking scenarios."
    )
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--split-path", default=None, help="Use an existing split file for all runs.")
    parser.add_argument("--split-strategy", choices=["pair_random", "id_disjoint"], default="id_disjoint")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--siamese-encoders",
        default="bilstm",
        help="Comma-separated list from {bilstm,charcnn}. Example: bilstm,charcnn",
    )
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--cnn-kernel-sizes", default="3,4,5")
    parser.add_argument("--classifier-hidden-dim", type=int, default=64)
    parser.add_argument("--baseline-max-len", type=int, default=80)
    parser.add_argument("--extended-max-len", type=int, default=128)

    parser.add_argument("--blocking-keys", default="first_name,age")
    parser.add_argument("--blocking-mode", choices=["all", "any"], default="any")

    parser.add_argument("--baseline-analyzer", choices=["char", "char_wb", "word"], default="char")
    parser.add_argument("--baseline-ngram-min", type=int, default=2)
    parser.add_argument("--baseline-ngram-max", type=int, default=4)

    parser.add_argument("--monitor-metric", choices=["f1", "pr_auc", "recall", "precision", "accuracy"], default="pr_auc")
    parser.add_argument("--scenario-set", choices=["all", "unblocked", "blocked"], default="all")
    parser.add_argument("--run-root", default="models/comparisons/apples_to_apples")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def parse_encoder_list(value):
    encoders = [v.strip() for v in str(value).split(",") if v.strip()]
    valid = {"bilstm", "charcnn"}
    invalid = [e for e in encoders if e not in valid]
    if invalid:
        raise ValueError(f"Invalid encoder(s): {invalid}. Valid choices: {sorted(valid)}")
    if not encoders:
        raise ValueError("No encoders provided.")
    return encoders


def scenario_specs(args):
    if args.scenario_set == "unblocked":
        return [{"name": "pair_scoring", "blocking_keys": "none"}]
    if args.scenario_set == "blocked":
        return [{"name": "blocked_pipeline", "blocking_keys": args.blocking_keys}]
    return [
        {"name": "pair_scoring", "blocking_keys": "none"},
        {"name": "blocked_pipeline", "blocking_keys": args.blocking_keys},
    ]


def train_siamese(args, attribute_set, scenario, run_dir, encoder):
    max_len = args.baseline_max_len if attribute_set == "baseline" else args.extended_max_len
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
        "--embedding-dim",
        str(args.embedding_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--encoder",
        encoder,
        "--cnn-channels",
        str(args.cnn_channels),
        "--cnn-kernel-sizes",
        args.cnn_kernel_sizes,
        "--classifier-hidden-dim",
        str(args.classifier_hidden_dim),
        "--max-len",
        str(max_len),
        "--attribute-set",
        attribute_set,
        "--monitor-metric",
        args.monitor_metric,
        "--use-weighted-loss",
        "--blocking-keys",
        scenario["blocking_keys"],
        "--blocking-mode",
        args.blocking_mode,
        "--run-dir",
        str(run_dir),
    ]
    if args.split_path:
        split_path = Path(args.split_path)
        if split_path.exists():
            cmd.extend(["--split-path", str(split_path)])
        else:
            cmd.extend(["--split-output", str(split_path)])
    if args.disable_progress:
        cmd.append("--disable-progress")
    subprocess.run(cmd, check=True)


def run_tfidf(args, attribute_set, scenario, run_dir):
    cmd = [
        sys.executable,
        "src/baseline_tfidf.py",
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
        "--attribute-set",
        attribute_set,
        "--analyzer",
        args.baseline_analyzer,
        "--ngram-min",
        str(args.baseline_ngram_min),
        "--ngram-max",
        str(args.baseline_ngram_max),
        "--blocking-keys",
        scenario["blocking_keys"],
        "--blocking-mode",
        args.blocking_mode,
        "--run-dir",
        str(run_dir),
    ]
    if args.split_path:
        split_path = Path(args.split_path)
        if split_path.exists():
            cmd.extend(["--split-path", str(split_path)])
        else:
            cmd.extend(["--split-output", str(split_path)])
    subprocess.run(cmd, check=True)


def collect_siamese_metrics(run_dir):
    with open(run_dir / "best_metrics.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "val_f1": payload["best_val_metrics"]["f1"],
        "val_pr_auc": payload["best_val_metrics"]["pr_auc"],
        "test_f1": payload["test_metrics"]["f1"],
        "test_pr_auc": payload["test_metrics"]["pr_auc"],
        "test_roc_auc": payload["test_metrics"]["roc_auc"],
        "best_threshold": payload["best_threshold"],
        "blocking_summary_test": payload.get("blocking_summary", {}).get("test", {}),
    }


def collect_tfidf_metrics(run_dir):
    with open(run_dir / "baseline_metrics.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "val_f1": payload["metrics"]["val"]["f1"],
        "val_pr_auc": payload["metrics"]["val"]["pr_auc"],
        "test_f1": payload["metrics"]["test"]["f1"],
        "test_pr_auc": payload["metrics"]["test"]["pr_auc"],
        "test_roc_auc": payload["metrics"]["test"]["roc_auc"],
        "best_threshold": payload["selected_distance_threshold"],
        "blocking_summary_test": payload["metrics"]["test"]["blocking"],
    }


def main():
    args = parse_args()
    siamese_encoders = parse_encoder_list(args.siamese_encoders)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in scenario_specs(args):
        for attribute_set in ["baseline", "extended"]:
            tfidf_dir = run_root / "tfidf" / scenario["name"] / attribute_set
            tfidf_dir.mkdir(parents=True, exist_ok=True)

            run_tfidf(args, attribute_set, scenario, tfidf_dir)
            tfidf_metrics = collect_tfidf_metrics(tfidf_dir)

            for encoder in siamese_encoders:
                siamese_dir = run_root / f"siamese_{encoder}" / scenario["name"] / attribute_set
                siamese_dir.mkdir(parents=True, exist_ok=True)
                train_siamese(args, attribute_set, scenario, siamese_dir, encoder)
                siamese_metrics = collect_siamese_metrics(siamese_dir)
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "method": f"siamese_{encoder}",
                        "attribute_set": attribute_set,
                        "blocking_keys": scenario["blocking_keys"],
                        "blocking_mode": args.blocking_mode,
                        "val_f1": siamese_metrics["val_f1"],
                        "val_pr_auc": siamese_metrics["val_pr_auc"],
                        "test_f1": siamese_metrics["test_f1"],
                        "test_pr_auc": siamese_metrics["test_pr_auc"],
                        "test_roc_auc": siamese_metrics["test_roc_auc"],
                        "best_threshold": siamese_metrics["best_threshold"],
                        "test_candidate_count": siamese_metrics["blocking_summary_test"].get("candidate_pair_count"),
                        "test_positive_pass_rate": siamese_metrics["blocking_summary_test"].get("positive_pass_rate"),
                        "test_negative_pass_rate": siamese_metrics["blocking_summary_test"].get("negative_pass_rate"),
                        "run_dir": str(siamese_dir),
                    }
                )
            rows.append(
                {
                    "scenario": scenario["name"],
                    "method": "tfidf",
                    "attribute_set": attribute_set,
                    "blocking_keys": scenario["blocking_keys"],
                    "blocking_mode": args.blocking_mode,
                    "val_f1": tfidf_metrics["val_f1"],
                    "val_pr_auc": tfidf_metrics["val_pr_auc"],
                    "test_f1": tfidf_metrics["test_f1"],
                    "test_pr_auc": tfidf_metrics["test_pr_auc"],
                    "test_roc_auc": tfidf_metrics["test_roc_auc"],
                    "best_threshold": tfidf_metrics["best_threshold"],
                    "test_candidate_count": tfidf_metrics["blocking_summary_test"].get("candidate_pair_count"),
                    "test_positive_pass_rate": tfidf_metrics["blocking_summary_test"].get("positive_pass_rate"),
                    "test_negative_pass_rate": tfidf_metrics["blocking_summary_test"].get("negative_pass_rate"),
                    "run_dir": str(tfidf_dir),
                }
            )

    summary_df = pd.DataFrame(rows).sort_values(
        ["scenario", "attribute_set", "test_pr_auc", "test_f1"],
        ascending=[True, True, False, False],
    )
    summary_tsv = run_root / "comparison_summary.tsv"
    summary_json = run_root / "comparison_summary.json"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print("Comparison matrix complete.")
    print(f"Summary TSV: {summary_tsv}")


if __name__ == "__main__":
    main()
