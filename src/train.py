import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from .data_utils import (
        count_missing_pair_ids,
        load_labeled_pairs,
        load_prepared_data,
        parse_attributes,
        set_seed,
    )
    from .dataset import NCVotersDataset
    from .metrics import compute_binary_metrics, select_threshold
    from .model import SiameseNetwork
    from .splits import build_splits, check_split_integrity, load_split_artifact, save_split_artifact
except ImportError:
    from blocking import compute_block_mask, parse_blocking_keys, summarize_blocking
    from data_utils import count_missing_pair_ids, load_labeled_pairs, load_prepared_data, parse_attributes, set_seed
    from dataset import NCVotersDataset
    from metrics import compute_binary_metrics, select_threshold
    from model import SiameseNetwork
    from splits import build_splits, check_split_integrity, load_split_artifact, save_split_artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Train Siamese LSTM for NCVoters duplicate detection.")
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

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--attribute-set", choices=["baseline", "extended"], default="baseline")
    parser.add_argument("--attributes-csv", default=None, help="Comma-separated list. Overrides --attribute-set.")
    parser.add_argument(
        "--blocking-keys",
        default="none",
        help="Comma-separated blocking keys for shared candidate generation. Use 'none' to disable blocking.",
    )
    parser.add_argument(
        "--blocking-mode",
        choices=["all", "any"],
        default="any",
        help="Whether all blocking keys must match or at least one must match.",
    )

    parser.add_argument("--use-weighted-loss", action="store_true")
    parser.add_argument("--class-weight-pos", type=float, default=None)

    parser.add_argument("--monitor-metric", choices=["f1", "pr_auc", "recall", "precision", "accuracy"], default="pr_auc")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--threshold-mode", choices=["f1", "precision_target"], default="f1")
    parser.add_argument("--precision-target", type=float, default=None)

    parser.add_argument("--run-dir", default="models/runs/default")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True, disable_progress=False):
    if train_mode:
        model.train()
        desc = "Train"
    else:
        model.eval()
        desc = "Eval"

    running_loss = 0.0
    probs_all = []
    labels_all = []

    iterator = tqdm(loader, desc=desc, disable=disable_progress)
    for x1, x2, labels in iterator:
        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device).float()

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(x1, x2).squeeze(-1)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        labels_all.append(labels.detach().cpu().numpy())

    if len(loader) == 0:
        raise ValueError("Loader is empty; check split generation and filters.")

    y_prob = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all).astype(int)
    return running_loss / len(loader), y_true, y_prob


def compute_pos_weight(train_pairs, explicit_pos_weight=None, use_weighted_loss=False):
    if explicit_pos_weight is not None:
        return float(explicit_pos_weight)
    if not use_weighted_loss:
        return None

    train_labels = train_pairs["label"].astype(float)
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    if n_pos == 0:
        raise ValueError("Cannot compute positive class weight because train split has zero positives.")
    if n_neg == 0:
        return None
    return n_neg / n_pos


def _resolve_split_path(args):
    if args.split_path:
        return Path(args.split_path)
    if args.split_output:
        return Path(args.split_output)
    return Path(f"data/processed/splits/{args.split_strategy}_seed{args.seed}.tsv")


def _build_or_load_splits(args, pairs_df):
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


def _save_confusion_matrix(path, metrics_payload):
    cm = metrics_payload["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(path, sep="\t")


def build_dataset_and_loader(args, pairs_df, vocab, attributes, shuffle=False):
    dataset = NCVotersDataset(
        data_path=args.data_path,
        pairs_df=pairs_df,
        vocab=vocab,
        max_len=args.max_len,
        attributes=attributes,
        strict_missing_ids=False,
        drop_missing_ids=True,
    )
    if len(dataset) == 0:
        return dataset, None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers
    )
    return dataset, loader


def restore_full_probabilities(full_pairs_df, candidate_mask, candidate_prob):
    full_prob = np.zeros(len(full_pairs_df), dtype=float)
    candidate_positions = np.flatnonzero(candidate_mask)
    full_prob[candidate_positions] = candidate_prob
    return full_prob


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    attributes = parse_attributes(args.attribute_set, args.attributes_csv)
    blocking_keys = parse_blocking_keys(args.blocking_keys, default_keys=[])

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Attributes: {attributes}")
    print(f"Blocking keys: {blocking_keys if blocking_keys else 'none'}")

    data_df = load_prepared_data(args.data_path)
    pairs_df = load_labeled_pairs(args.dpl_path, args.ndpl_path, fail_on_duplicate_pairs=True)

    missing_rows, missing_ids = count_missing_pair_ids(pairs_df, data_df["id"].astype(str))
    if missing_rows > 0:
        print(
            f"Warning: {missing_rows} pair rows reference IDs not present in prepared data. "
            f"Unique missing IDs: {len(missing_ids)}"
        )

    split_df, split_path = _build_or_load_splits(args, pairs_df)
    split_summary = check_split_integrity(split_df)

    print(f"Split artifact: {split_path}")
    print(f"Split summary: {split_summary}")
    if args.split_strategy == "id_disjoint":
        overlap_fields = ["train_val_id_overlap", "train_test_id_overlap", "val_test_id_overlap"]
        for field in overlap_fields:
            if split_summary[field] != 0:
                raise ValueError(f"ID leakage detected for id_disjoint split ({field}={split_summary[field]}).")

    train_pairs = split_df[split_df["split"] == "train"][["pair_id", "id1", "id2", "label"]].copy()
    val_pairs = split_df[split_df["split"] == "val"][["pair_id", "id1", "id2", "label"]].copy()
    test_pairs = split_df[split_df["split"] == "test"][["pair_id", "id1", "id2", "label"]].copy()

    train_block_mask = compute_block_mask(train_pairs, data_df, blocking_keys, args.blocking_mode)
    val_block_mask = compute_block_mask(val_pairs, data_df, blocking_keys, args.blocking_mode)
    test_block_mask = compute_block_mask(test_pairs, data_df, blocking_keys, args.blocking_mode)
    blocking_summary = {
        "train": summarize_blocking(train_pairs["label"].astype(int).to_numpy(), train_block_mask),
        "val": summarize_blocking(val_pairs["label"].astype(int).to_numpy(), val_block_mask),
        "test": summarize_blocking(test_pairs["label"].astype(int).to_numpy(), test_block_mask),
    }
    print(f"Blocking summary: {blocking_summary}")

    train_candidate_pairs = train_pairs.loc[train_block_mask].copy().reset_index(drop=True)
    val_candidate_pairs = val_pairs.loc[val_block_mask].copy().reset_index(drop=True)
    test_candidate_pairs = test_pairs.loc[test_block_mask].copy().reset_index(drop=True)

    train_dataset = NCVotersDataset(
        data_path=args.data_path,
        pairs_df=train_candidate_pairs,
        max_len=args.max_len,
        attributes=attributes,
        strict_missing_ids=False,
        drop_missing_ids=True,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training candidate set is empty after blocking.")
    vocab = train_dataset.vocab

    val_dataset, val_loader = build_dataset_and_loader(
        args=args,
        pairs_df=val_candidate_pairs,
        vocab=vocab,
        attributes=attributes,
        shuffle=False,
    )
    test_dataset, test_loader = build_dataset_and_loader(
        args=args,
        pairs_df=test_candidate_pairs,
        vocab=vocab,
        attributes=attributes,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = SiameseNetwork(
        vocab_size=vocab.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    pos_weight = compute_pos_weight(
        train_pairs=train_candidate_pairs,
        explicit_pos_weight=args.class_weight_pos,
        use_weighted_loss=args.use_weighted_loss,
    )
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        print(f"Using weighted BCEWithLogitsLoss with pos_weight={pos_weight:.6f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted BCEWithLogitsLoss.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_metric_value = float("-inf")
    best_epoch = -1
    best_threshold = 0.5
    best_state_dict = None
    best_val_metrics = None
    best_threshold_sweep = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_candidate_y_true, train_candidate_y_prob = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            disable_progress=args.disable_progress,
        )
        train_y_true = train_pairs["label"].astype(int).to_numpy()
        train_y_prob = restore_full_probabilities(train_pairs, train_block_mask, train_candidate_y_prob)

        if val_loader is None:
            val_loss = float("nan")
            val_y_true = val_pairs["label"].astype(int).to_numpy()
            val_y_prob = np.zeros(len(val_pairs), dtype=float)
        else:
            val_loss, _, val_candidate_y_prob = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train_mode=False,
                disable_progress=args.disable_progress,
            )
            val_y_true = val_pairs["label"].astype(int).to_numpy()
            val_y_prob = restore_full_probabilities(val_pairs, val_block_mask, val_candidate_y_prob)

        epoch_threshold, _, threshold_sweep = select_threshold(
            val_y_true,
            val_y_prob,
            mode=args.threshold_mode,
            precision_target=args.precision_target,
        )
        train_metrics = compute_binary_metrics(train_y_true, train_y_prob, threshold=epoch_threshold)
        val_metrics = compute_binary_metrics(val_y_true, val_y_prob, threshold=epoch_threshold)

        monitor_value = val_metrics[args.monitor_metric]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "threshold": epoch_threshold,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "monitor_metric": args.monitor_metric,
                "monitor_value": monitor_value,
                "train_candidate_count": int(train_block_mask.sum()),
                "val_candidate_count": int(val_block_mask.sum()),
            }
        )

        print(
            f"Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} "
            f"Val F1={val_metrics['f1']:.4f} Val PR-AUC={val_metrics['pr_auc']:.4f} "
            f"Val ROC-AUC={val_metrics['roc_auc']:.4f} Threshold={epoch_threshold:.2f}"
        )

        if monitor_value > best_metric_value:
            best_metric_value = monitor_value
            best_epoch = epoch
            best_threshold = epoch_threshold
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            best_threshold_sweep = threshold_sweep.copy()
            epochs_without_improvement = 0
            print(f"Improved {args.monitor_metric} to {monitor_value:.6f}.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered (patience={args.early_stopping_patience}).")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state_dict)
    if test_loader is None:
        test_loss = float("nan")
        test_y_true = test_pairs["label"].astype(int).to_numpy()
        test_y_prob = np.zeros(len(test_pairs), dtype=float)
    else:
        test_loss, _, test_candidate_y_prob = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            disable_progress=args.disable_progress,
        )
        test_y_true = test_pairs["label"].astype(int).to_numpy()
        test_y_prob = restore_full_probabilities(test_pairs, test_block_mask, test_candidate_y_prob)
    test_metrics = compute_binary_metrics(test_y_true, test_y_prob, threshold=best_threshold)
    test_metrics["blocking"] = blocking_summary["test"]

    checkpoint_path = run_dir / "best_model.pth"
    checkpoint = {
        "model_state_dict": best_state_dict,
        "vocab": train_dataset.vocab.to_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "max_len": args.max_len,
            "attributes": attributes,
            "split_strategy": args.split_strategy,
            "seed": args.seed,
            "monitor_metric": args.monitor_metric,
            "threshold_mode": args.threshold_mode,
            "precision_target": args.precision_target,
            "data_path": args.data_path,
            "split_path": str(split_path),
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "blocking_keys": blocking_keys,
            "blocking_mode": args.blocking_mode,
        },
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "split_summary": split_summary,
        "blocking_summary": blocking_summary,
    }
    torch.save(checkpoint, checkpoint_path)

    history_df = pd.DataFrame(history_rows)
    history_path = run_dir / "metrics_history.tsv"
    history_df.to_csv(history_path, sep="\t", index=False)

    if best_threshold_sweep is not None:
        threshold_sweep_path = run_dir / "val_threshold_sweep.tsv"
        best_threshold_sweep.to_csv(threshold_sweep_path, sep="\t", index=False)

    summary_payload = {
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "monitor_metric": args.monitor_metric,
        "best_monitor_value": best_metric_value,
        "best_val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "split_summary": split_summary,
        "blocking_keys": blocking_keys,
        "blocking_mode": args.blocking_mode,
        "blocking_summary": blocking_summary,
        "paths": {
            "checkpoint": str(checkpoint_path),
            "split": str(split_path),
            "history_tsv": str(history_path),
        },
    }
    summary_path = run_dir / "best_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    split_summary_path = run_dir / "split_summary.json"
    with open(split_summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    test_cm_path = run_dir / "test_confusion_matrix.tsv"
    _save_confusion_matrix(test_cm_path, test_metrics)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
