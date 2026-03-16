import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .baseline_tfidf import compute_pair_distances, fit_vectorizer
    from .blocking import compute_block_mask
    from .dataset import NCVotersDataset, Vocabulary
    from .model import SiameseNetwork
except ImportError:
    from baseline_tfidf import compute_pair_distances, fit_vectorizer
    from blocking import compute_block_mask
    from dataset import NCVotersDataset, Vocabulary
    from model import SiameseNetwork


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Siamese and TF-IDF predictions on arbitrary candidate pairs."
    )
    parser.add_argument(
        "--candidate-path",
        default="data/processed/generated_hard_negatives/generated_hard_negative_candidates.tsv",
        help="TSV with at least id1 and id2 columns.",
    )
    parser.add_argument("--data-path", default="data/processed/ncvoters_prepared.tsv")
    parser.add_argument(
        "--siamese-model-path",
        default="models/comparisons/apples_to_apples_with_charcnn/siamese_bilstm/pair_scoring/extended/best_model.pth",
    )
    parser.add_argument(
        "--tfidf-run-dir",
        default="models/comparisons/apples_to_apples_with_charcnn/tfidf/pair_scoring/extended",
        help="Directory containing baseline_metrics.json for the TF-IDF run to reuse.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="models/hard_negative_comparison")
    parser.add_argument("--output-prefix", default="generated_hard_negatives")
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_candidate_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"id1", "id2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Candidate file missing required columns: {sorted(missing)}")
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    if "pair_key" not in df.columns:
        df["pair_key"] = df.apply(lambda row: "||".join(sorted((row["id1"], row["id2"]))), axis=1)
    if "label" not in df.columns:
        df["label"] = 0.0
    else:
        df["label"] = df["label"].astype(float)
    return df


def run_siamese_predictions(args, candidate_df):
    device = pick_device(args.device)
    checkpoint = torch.load(args.siamese_model_path, map_location=device)
    config = checkpoint["config"]
    vocab = Vocabulary.from_dict(checkpoint["vocab"])

    dataset = NCVotersDataset(
        data_path=args.data_path,
        pairs_df=candidate_df[["id1", "id2", "label"]].assign(pair_id=np.arange(len(candidate_df))),
        vocab=vocab,
        max_len=int(config["max_len"]),
        attributes=list(config["attributes"]),
        strict_missing_ids=False,
        drop_missing_ids=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiameseNetwork(
        vocab_size=vocab.vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        encoder_type=config.get("encoder_type", "bilstm"),
        cnn_channels=int(config.get("cnn_channels", config.get("hidden_dim", 64))),
        cnn_kernel_sizes=config.get("cnn_kernel_sizes", [3, 4, 5]),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = []
    with torch.no_grad():
        for x1, x2, _ in loader:
            logits = model(x1.to(device), x2.to(device)).squeeze(-1)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    threshold = float(checkpoint["best_threshold"])
    return {
        "encoder_type": config.get("encoder_type", "bilstm"),
        "attributes": list(config["attributes"]),
        "threshold": threshold,
        "probabilities": np.asarray(probs, dtype=float),
        "predictions": (np.asarray(probs, dtype=float) >= threshold).astype(int),
    }


def build_tfidf_args(payload):
    vectorizer_cfg = payload["vectorizer"]
    return _Namespace(
        analyzer=vectorizer_cfg["analyzer"],
        ngram_min=int(vectorizer_cfg["ngram_range"][0]),
        ngram_max=int(vectorizer_cfg["ngram_range"][1]),
        min_df=int(vectorizer_cfg["min_df"]),
        max_features=vectorizer_cfg["max_features"],
        tagged_serialization=bool(payload.get("tagged_serialization", False)),
    )


def run_tfidf_predictions(args, candidate_df):
    run_dir = Path(args.tfidf_run_dir)
    metrics_path = run_dir / "baseline_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data_df = pd.read_csv(args.data_path, sep="\t", dtype=str).fillna("")
    data_df["id"] = data_df["id"].astype(str)

    tfidf_args = build_tfidf_args(payload)
    attributes = list(payload["attributes"])
    vectorizer, matrix, record_to_row = fit_vectorizer(data_df, attributes, tfidf_args)

    pair_df = candidate_df[["id1", "id2", "label"]].copy()
    block_mask = compute_block_mask(
        pair_df,
        data_df,
        payload.get("blocking_keys", []),
        payload.get("blocking_mode", "any"),
    )
    distances, missing_mask = compute_pair_distances(pair_df, matrix, record_to_row)
    distances = np.where(block_mask, distances, 1.0)
    distances = np.where(missing_mask, 1.0, distances)

    distance_threshold = float(payload["selected_distance_threshold"])
    similarity = 1.0 - distances
    predictions = (distances <= distance_threshold).astype(int)
    return {
        "attributes": attributes,
        "threshold": distance_threshold,
        "similarity": similarity,
        "distance": distances,
        "predictions": predictions,
        "vectorizer_vocab_size": int(len(vectorizer.vocabulary_)),
    }


def build_pair_output(candidate_df, siamese_payload, tfidf_payload):
    output_df = candidate_df.copy()
    output_df["siamese_encoder"] = siamese_payload["encoder_type"]
    output_df["siamese_prob"] = siamese_payload["probabilities"]
    output_df["siamese_pred_duplicate"] = siamese_payload["predictions"]
    output_df["tfidf_similarity"] = tfidf_payload["similarity"]
    output_df["tfidf_distance"] = tfidf_payload["distance"]
    output_df["tfidf_pred_duplicate"] = tfidf_payload["predictions"]

    output_df["agreement"] = np.where(
        output_df["siamese_pred_duplicate"] == output_df["tfidf_pred_duplicate"],
        "agree",
        "disagree",
    )
    output_df["baseline_yes_siamese_no"] = (
        (output_df["tfidf_pred_duplicate"] == 1) & (output_df["siamese_pred_duplicate"] == 0)
    ).astype(int)
    output_df["siamese_yes_baseline_no"] = (
        (output_df["siamese_pred_duplicate"] == 1) & (output_df["tfidf_pred_duplicate"] == 0)
    ).astype(int)
    return output_df


def summarize_by_bucket(pair_df):
    if "bucket" not in pair_df.columns and "buckets" not in pair_df.columns:
        return pd.DataFrame()

    working = pair_df.copy()
    if "bucket" not in working.columns:
        working["bucket"] = working["buckets"].str.split(",")
        working = working.explode("bucket")

    summary = (
        working.groupby("bucket", as_index=False)
        .agg(
            pair_count=("pair_key", "nunique"),
            siamese_duplicate_count=("siamese_pred_duplicate", "sum"),
            tfidf_duplicate_count=("tfidf_pred_duplicate", "sum"),
            disagreement_count=("agreement", lambda s: int((s == "disagree").sum())),
            baseline_yes_siamese_no=("baseline_yes_siamese_no", "sum"),
            siamese_yes_baseline_no=("siamese_yes_baseline_no", "sum"),
            mean_siamese_prob=("siamese_prob", "mean"),
            mean_tfidf_similarity=("tfidf_similarity", "mean"),
        )
        .sort_values(["baseline_yes_siamese_no", "disagreement_count", "pair_count"], ascending=[False, False, False])
    )
    summary["siamese_duplicate_rate"] = summary["siamese_duplicate_count"] / summary["pair_count"]
    summary["tfidf_duplicate_rate"] = summary["tfidf_duplicate_count"] / summary["pair_count"]
    return summary


def build_text_report(pair_df, bucket_summary, siamese_payload, tfidf_payload):
    total = len(pair_df)
    tfidf_dup = int(pair_df["tfidf_pred_duplicate"].sum())
    siamese_dup = int(pair_df["siamese_pred_duplicate"].sum())
    desired = int(pair_df["baseline_yes_siamese_no"].sum())
    reverse = int(pair_df["siamese_yes_baseline_no"].sum())
    disagree = int((pair_df["agreement"] == "disagree").sum())

    lines = []
    lines.append("Hard-Negative Candidate Comparison Report")
    lines.append("")
    lines.append(f"- Candidate pairs scored: {total}")
    lines.append(f"- Siamese encoder: {siamese_payload['encoder_type']}")
    lines.append(f"- Siamese threshold: {siamese_payload['threshold']:.4f}")
    lines.append(f"- TF-IDF distance threshold: {tfidf_payload['threshold']:.4f}")
    lines.append("")
    lines.append("Overall prediction counts:")
    lines.append(f"- TF-IDF predicts duplicate: {tfidf_dup}")
    lines.append(f"- Siamese predicts duplicate: {siamese_dup}")
    lines.append(f"- Disagreements: {disagree}")
    lines.append(f"- TF-IDF duplicate, Siamese non-duplicate: {desired}")
    lines.append(f"- Siamese duplicate, TF-IDF non-duplicate: {reverse}")

    if not bucket_summary.empty:
        lines.append("")
        lines.append("Bucket summary:")
        for _, row in bucket_summary.iterrows():
            lines.append(
                f"- {row['bucket']}: pairs={int(row['pair_count'])}, "
                f"tfidf_dup={int(row['tfidf_duplicate_count'])}, "
                f"siamese_dup={int(row['siamese_duplicate_count'])}, "
                f"baseline_yes_siamese_no={int(row['baseline_yes_siamese_no'])}"
            )

    focus = pair_df[pair_df["baseline_yes_siamese_no"] == 1].copy()
    if not focus.empty:
        lines.append("")
        lines.append("Most promising cases (TF-IDF says duplicate, Siamese says different):")
        focus = focus.sort_values(["tfidf_similarity", "siamese_prob"], ascending=[False, True]).head(10)
        for _, row in focus.iterrows():
            bucket = row["bucket"] if "bucket" in row else row.get("buckets", "")
            lines.append(
                f"- {row['pair_key']} | bucket={bucket} | tfidf_similarity={float(row['tfidf_similarity']):.4f} | "
                f"siamese_prob={float(row['siamese_prob']):.4f}"
            )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    candidate_df = load_candidate_pairs(Path(args.candidate_path))

    siamese_payload = run_siamese_predictions(args, candidate_df)
    tfidf_payload = run_tfidf_predictions(args, candidate_df)

    output_df = build_pair_output(candidate_df, siamese_payload, tfidf_payload)
    bucket_summary = summarize_by_bucket(output_df)
    report = build_text_report(output_df, bucket_summary, siamese_payload, tfidf_payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_path = output_dir / f"{args.output_prefix}_pair_predictions.tsv"
    bucket_path = output_dir / f"{args.output_prefix}_bucket_summary.tsv"
    report_path = output_dir / f"{args.output_prefix}_report.txt"

    output_df.to_csv(pair_path, sep="\t", index=False)
    bucket_summary.to_csv(bucket_path, sep="\t", index=False)
    report_path.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Saved pair predictions: {pair_path}")
    print(f"Saved bucket summary: {bucket_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
