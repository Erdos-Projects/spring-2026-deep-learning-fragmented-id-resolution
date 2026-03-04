import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from .dataset import NCVotersDataset, Vocabulary
    from .metrics import compute_binary_metrics, select_threshold
    from .model import SiameseNetwork
    from .splits import check_split_integrity, load_split_artifact
except ImportError:
    from blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from dataset import NCVotersDataset, Vocabulary
    from metrics import compute_binary_metrics, select_threshold
    from model import SiameseNetwork
    from splits import check_split_integrity, load_split_artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Siamese LSTM checkpoint on a split file.")
    parser.add_argument("--model-path", required=True, help="Path to saved checkpoint from src/train.py")
    parser.add_argument("--split-path", required=True, help="Path to split TSV generated during training")
    parser.add_argument("--split-name", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-path", default=None, help="Override prepared data path in checkpoint config")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold-mode", choices=["checkpoint", "f1", "precision_target", "fixed"], default="checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Used only when --threshold-mode=fixed")
    parser.add_argument("--precision-target", type=float, default=None)
    parser.add_argument(
        "--blocking-keys",
        default=None,
        help="Override checkpoint blocking keys. Use 'none' to disable. Default: use checkpoint setting.",
    )
    parser.add_argument("--blocking-mode", choices=["all", "any"], default=None)
    parser.add_argument("--output-dir", default="models/eval")
    parser.add_argument("--output-prefix", default="evaluation")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, loader, device, disable_progress=False):
    model.eval()
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for x1, x2, labels in tqdm(loader, desc="Inference", disable=disable_progress):
            x1 = x1.to(device)
            x2 = x2.to(device)
            logits = model(x1, x2).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels.numpy())

    if not probs_all:
        raise ValueError("No batches were produced for inference.")

    y_prob = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all).astype(int)
    return y_true, y_prob


def restore_full_probabilities(full_pairs_df, candidate_mask, candidate_prob):
    full_prob = np.zeros(len(full_pairs_df), dtype=float)
    full_prob[np.flatnonzero(candidate_mask)] = candidate_prob
    return full_prob


def save_confusion_matrix(path, metrics_payload):
    cm = metrics_payload["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(path, sep="\t")


def main():
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint["config"]
    vocab = Vocabulary.from_dict(checkpoint["vocab"])

    data_path = args.data_path if args.data_path else config["data_path"]
    split_df = load_split_artifact(args.split_path)
    split_summary = check_split_integrity(split_df)

    split_pairs = split_df[split_df["split"] == args.split_name][["pair_id", "id1", "id2", "label"]].copy()
    if split_pairs.empty:
        raise ValueError(f"Split '{args.split_name}' is empty in {args.split_path}")

    blocking_keys = (
        parse_blocking_keys(args.blocking_keys, default_keys=[])
        if args.blocking_keys is not None
        else list(config.get("blocking_keys", []))
    )
    blocking_mode = args.blocking_mode if args.blocking_mode is not None else config.get("blocking_mode", "any")

    data_df = pd.read_csv(data_path, sep="\t", dtype=str).fillna("")
    data_df["id"] = data_df["id"].astype(str)
    candidate_mask = compute_block_mask(split_pairs, data_df, blocking_keys, blocking_mode)
    candidate_pairs = split_pairs.loc[candidate_mask].copy().reset_index(drop=True)

    dataset = NCVotersDataset(
        data_path=data_path,
        pairs_df=candidate_pairs,
        vocab=vocab,
        max_len=int(config["max_len"]),
        attributes=list(config["attributes"]),
        strict_missing_ids=False,
        drop_missing_ids=True,
    )
    loader = None
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiameseNetwork(
        vocab_size=vocab.vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true = split_pairs["label"].astype(int).to_numpy()
    if loader is None:
        y_prob = np.zeros(len(split_pairs), dtype=float)
    else:
        _, candidate_prob = run_inference(model, loader, device, disable_progress=args.disable_progress)
        y_prob = restore_full_probabilities(split_pairs, candidate_mask, candidate_prob)

    if args.threshold_mode == "checkpoint":
        chosen_threshold = float(checkpoint.get("best_threshold", 0.5))
        threshold_source = "checkpoint"
    elif args.threshold_mode == "fixed":
        chosen_threshold = float(args.threshold)
        threshold_source = "fixed"
    elif args.threshold_mode in {"f1", "precision_target"}:
        chosen_threshold, _, sweep_df = select_threshold(
            y_true,
            y_prob,
            mode=args.threshold_mode,
            precision_target=args.precision_target,
        )
        threshold_source = args.threshold_mode
    else:
        raise ValueError(f"Unknown threshold mode: {args.threshold_mode}")

    metrics = compute_binary_metrics(y_true, y_prob, threshold=chosen_threshold)
    metrics["blocking"] = summarize_blocking(y_true, candidate_mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.output_prefix}_{args.split_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        payload = {
            "split_name": args.split_name,
            "threshold_source": threshold_source,
            "chosen_threshold": chosen_threshold,
            "metrics": metrics,
            "split_summary": split_summary,
            "blocking_keys": blocking_keys,
            "blocking_mode": blocking_mode,
            "model_path": args.model_path,
            "split_path": args.split_path,
            "data_path": data_path,
        }
        json.dump(payload, f, indent=2)

    tsv_path = output_dir / f"{args.output_prefix}_{args.split_name}_metrics.tsv"
    pd.DataFrame([metrics]).to_csv(tsv_path, sep="\t", index=False)

    cm_path = output_dir / f"{args.output_prefix}_{args.split_name}_confusion_matrix.tsv"
    save_confusion_matrix(cm_path, metrics)

    if args.threshold_mode in {"f1", "precision_target"}:
        sweep_path = output_dir / f"{args.output_prefix}_{args.split_name}_threshold_sweep.tsv"
        sweep_df.to_csv(sweep_path, sep="\t", index=False)

    print(f"Evaluation complete on split '{args.split_name}'.")
    print(f"Threshold: {chosen_threshold:.2f} ({threshold_source})")
    print(f"F1={metrics['f1']:.4f} PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
