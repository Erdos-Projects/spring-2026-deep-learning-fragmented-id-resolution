import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from .data_utils import (
        count_missing_pair_ids,
        load_labeled_pairs,
        load_prepared_data,
        parse_attributes,
        set_seed,
    )
    from .metrics import compute_binary_metrics
    from .splits import build_splits, check_split_integrity, load_split_artifact, save_split_artifact
except ImportError:
    from blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from data_utils import (
        count_missing_pair_ids,
        load_labeled_pairs,
        load_prepared_data,
        parse_attributes,
        set_seed,
    )
    from metrics import compute_binary_metrics
    from splits import build_splits, check_split_integrity, load_split_artifact, save_split_artifact


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-IDF embedding-distance baseline for NCVoters duplicate detection."
    )
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument("--dpl-path", default="data/raw/ncvoters_DPL.tsv")
    parser.add_argument("--ndpl-path", default="data/raw/ncvoters_NDPL.tsv")
    parser.add_argument("--split-path", default=None, help="Use an existing split TSV if provided.")
    parser.add_argument(
        "--split-output",
        default=None,
        help="Where to save generated split TSV. Default: data/processed/splits/{strategy}_seed{seed}.tsv",
    )
    parser.add_argument("--split-strategy", choices=["pair_random", "id_disjoint"], default="id_disjoint")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--attribute-set", choices=["baseline", "extended"], default="baseline")
    parser.add_argument("--attributes-csv", default=None, help="Comma-separated list. Overrides --attribute-set.")
    parser.add_argument(
        "--tagged-serialization",
        action="store_true",
        help="Prefix each attribute with its field name when serializing records.",
    )

    parser.add_argument("--analyzer", choices=["char", "char_wb", "word"], default="char")
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)

    parser.add_argument(
        "--blocking-keys",
        default="first_name,age",
        help="Comma-separated blocking keys. Use 'none' to disable blocking.",
    )
    parser.add_argument(
        "--blocking-mode",
        choices=["all", "any"],
        default="any",
        help="Whether all blocking keys must match or at least one must match.",
    )

    parser.add_argument("--threshold-selection", choices=["f1", "precision_target", "fixed"], default="f1")
    parser.add_argument("--precision-target", type=float, default=None)
    parser.add_argument("--fixed-distance-threshold", type=float, default=0.30)
    parser.add_argument("--distance-grid-size", type=int, default=201)

    parser.add_argument("--run-dir", default="models/baselines/tfidf_distance")
    return parser.parse_args()
def serialize_record(record, attributes, tagged=False):
    tokens = []
    for attr in attributes:
        value = str(record.get(attr, "")).strip()
        if tagged:
            tokens.append(f"{attr}:{value}")
        else:
            tokens.append(value)
    return " ".join(tokens).strip()


def fit_vectorizer(data_df, attributes, args):
    texts = data_df.apply(lambda row: serialize_record(row, attributes, tagged=args.tagged_serialization), axis=1)
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
    )
    matrix = vectorizer.fit_transform(texts)
    record_to_row = {record_id: idx for idx, record_id in enumerate(data_df["id"].astype(str))}
    return vectorizer, matrix, record_to_row
def compute_pair_distances(pairs_df, matrix, record_to_row):
    distances = np.ones(len(pairs_df), dtype=float)
    missing_mask = np.zeros(len(pairs_df), dtype=bool)

    row_indices_1 = []
    row_indices_2 = []
    valid_positions = []

    for idx, row in enumerate(pairs_df.itertuples(index=False)):
        row_idx_1 = record_to_row.get(str(row.id1))
        row_idx_2 = record_to_row.get(str(row.id2))
        if row_idx_1 is None or row_idx_2 is None:
            missing_mask[idx] = True
            continue
        row_indices_1.append(row_idx_1)
        row_indices_2.append(row_idx_2)
        valid_positions.append(idx)

    if valid_positions:
        similarities = np.asarray(
            matrix[row_indices_1].multiply(matrix[row_indices_2]).sum(axis=1)
        ).ravel()
        distances[np.asarray(valid_positions, dtype=int)] = 1.0 - similarities

    return distances, missing_mask


def select_distance_threshold(distances, labels, selection_mode="f1", precision_target=None, fixed_threshold=0.30, grid_size=201):
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if selection_mode == "fixed":
        metrics = compute_binary_metrics(labels, 1.0 - distances, threshold=1.0 - fixed_threshold)
        sweep_df = pd.DataFrame([_metrics_row_from_distance(fixed_threshold, metrics)])
        return fixed_threshold, metrics, sweep_df

    max_threshold = min(1.0, float(distances.max()) + 1e-9)
    thresholds = np.linspace(0.0, max_threshold, max(grid_size, 2))
    rows = []
    for distance_threshold in thresholds:
        similarity_threshold = 1.0 - float(distance_threshold)
        metrics = compute_binary_metrics(labels, 1.0 - distances, threshold=similarity_threshold)
        rows.append(_metrics_row_from_distance(distance_threshold, metrics))

    sweep_df = pd.DataFrame(rows)
    if selection_mode == "precision_target":
        if precision_target is None:
            raise ValueError("precision_target threshold selection requires --precision-target.")
        candidates = sweep_df[sweep_df["precision"] >= precision_target]
        if candidates.empty:
            best_row = sweep_df.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]
        else:
            best_row = candidates.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]
    elif selection_mode == "f1":
        best_row = sweep_df.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).iloc[0]
    else:
        raise ValueError(f"Unknown threshold selection mode '{selection_mode}'.")

    best_distance = float(best_row["distance_threshold"])
    best_metrics = compute_binary_metrics(labels, 1.0 - distances, threshold=1.0 - best_distance)
    return best_distance, best_metrics, sweep_df


def summarize_split(pairs_df, distances, block_mask, missing_mask, distance_threshold):
    labels = pairs_df["label"].astype(int).to_numpy()
    metrics = compute_binary_metrics(labels, 1.0 - distances, threshold=1.0 - distance_threshold)
    metrics["distance_threshold"] = float(distance_threshold)
    metrics["missing_pair_count"] = int(missing_mask.sum())
    metrics["mean_distance"] = float(np.mean(distances)) if len(distances) else float("nan")
    metrics["median_distance"] = float(np.median(distances)) if len(distances) else float("nan")
    metrics["blocking"] = summarize_blocking(labels, block_mask)
    return metrics


def save_confusion_matrix(path, metrics_payload):
    cm = metrics_payload["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(path, sep="\t")


def _metrics_row_from_distance(distance_threshold, metrics):
    return {
        "distance_threshold": float(distance_threshold),
        "similarity_threshold": float(1.0 - distance_threshold),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


def _resolve_split_path(args):
    if args.split_path:
        return Path(args.split_path)
    if args.split_output:
        return Path(args.split_output)
    return Path(f"data/processed/splits/{args.split_strategy}_seed{args.seed}.tsv")


def _load_or_build_split(args, pairs_df):
    split_path = _resolve_split_path(args)
    if args.split_path:
        split_df = load_split_artifact(split_path)
    else:
        split_df = build_splits(
            pairs_df,
            strategy=args.split_strategy,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        save_split_artifact(split_df, split_path)
    return split_df, split_path


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    attributes = parse_attributes(args.attribute_set, args.attributes_csv)
    blocking_keys = parse_blocking_keys(args.blocking_keys)

    data_df = load_prepared_data(args.data_path)
    pairs_df = load_labeled_pairs(args.dpl_path, args.ndpl_path, fail_on_duplicate_pairs=True)
    missing_rows, missing_ids = count_missing_pair_ids(pairs_df, data_df["id"].astype(str))
    if missing_rows > 0:
        print(
            f"Warning: {missing_rows} pair rows reference IDs not present in prepared data. "
            f"Unique missing IDs: {len(missing_ids)}"
        )

    split_df, split_path = _load_or_build_split(args, pairs_df)
    split_summary = check_split_integrity(split_df)

    vectorizer, matrix, record_to_row = fit_vectorizer(data_df, attributes, args)

    results = {}
    threshold_sweep_df = None
    selected_distance_threshold = None

    for split_name in ["train", "val", "test"]:
        split_pairs = split_df[split_df["split"] == split_name][["pair_id", "id1", "id2", "label"]].copy()
        split_pairs["id1"] = split_pairs["id1"].astype(str)
        split_pairs["id2"] = split_pairs["id2"].astype(str)

        block_mask = compute_block_mask(split_pairs, data_df, blocking_keys, args.blocking_mode)
        distances, missing_mask = compute_pair_distances(split_pairs, matrix, record_to_row)
        distances = np.where(block_mask, distances, 1.0)
        distances = np.where(missing_mask, 1.0, distances)

        if split_name == "val":
            selected_distance_threshold, _, threshold_sweep_df = select_distance_threshold(
                distances=distances,
                labels=split_pairs["label"].astype(int).to_numpy(),
                selection_mode=args.threshold_selection,
                precision_target=args.precision_target,
                fixed_threshold=args.fixed_distance_threshold,
                grid_size=args.distance_grid_size,
            )

        if split_name == "test" and selected_distance_threshold is None:
            selected_distance_threshold = float(args.fixed_distance_threshold)

        current_threshold = (
            selected_distance_threshold if selected_distance_threshold is not None else float(args.fixed_distance_threshold)
        )
        results[split_name] = summarize_split(
            split_pairs,
            distances=distances,
            block_mask=block_mask,
            missing_mask=missing_mask,
            distance_threshold=current_threshold,
        )

    if threshold_sweep_df is not None:
        threshold_sweep_df.to_csv(run_dir / "val_distance_threshold_sweep.tsv", sep="\t", index=False)

    summary_payload = {
        "model_type": "tfidf_cosine_distance_baseline",
        "attributes": attributes,
        "blocking_keys": blocking_keys,
        "blocking_mode": args.blocking_mode,
        "tagged_serialization": bool(args.tagged_serialization),
        "vectorizer": {
            "analyzer": args.analyzer,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "vocab_size": int(len(vectorizer.vocabulary_)),
        },
        "threshold_selection": args.threshold_selection,
        "selected_distance_threshold": float(selected_distance_threshold),
        "selected_similarity_threshold": float(1.0 - selected_distance_threshold),
        "split_summary": split_summary,
        "metrics": results,
        "paths": {
            "split": str(split_path),
            "run_dir": str(run_dir),
        },
    }

    json_path = run_dir / "baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    flat_metrics = []
    for split_name, payload in results.items():
        row = {"split": split_name}
        row.update({k: v for k, v in payload.items() if k not in {"confusion_matrix", "blocking"}})
        row.update(payload["blocking"])
        row.update(
            {
                "tn": payload["confusion_matrix"]["tn"],
                "fp": payload["confusion_matrix"]["fp"],
                "fn": payload["confusion_matrix"]["fn"],
                "tp": payload["confusion_matrix"]["tp"],
            }
        )
        flat_metrics.append(row)
    pd.DataFrame(flat_metrics).to_csv(run_dir / "baseline_metrics.tsv", sep="\t", index=False)

    save_confusion_matrix(run_dir / "test_confusion_matrix.tsv", results["test"])

    print(f"Baseline complete. Distance threshold: {selected_distance_threshold:.4f}")
    print(
        f"Validation F1={results['val']['f1']:.4f} "
        f"Test F1={results['test']['f1']:.4f} "
        f"Test PR-AUC={results['test']['pr_auc']:.4f}"
    )
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
