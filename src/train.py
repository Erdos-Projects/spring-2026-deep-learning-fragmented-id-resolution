"""
Training loop for the Siamese matching network.

Features:
  - Binary cross-entropy loss with label smoothing
  - Learning rate scheduling (cosine annealing)
  - Early stopping on validation loss
  - Evaluation metrics: Precision, Recall, F1, AUC, calibration
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .model import SiameseMatchingNetwork


def train_one_epoch(
    model: SiameseMatchingNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input1 = batch["input1"].to(device)
        input2 = batch["input2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input1, input2)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: SiameseMatchingNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the model and return a dict of metrics.

    Returns:
        dict with keys: loss, accuracy, precision, recall, f1, auc, fpr, fnr
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probs = []
    all_labels = []

    for batch in dataloader:
        input1 = batch["input1"].to(device)
        input2 = batch["input2"].to(device)
        labels = batch["label"].to(device)

        logits = model(input1, input2)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    acc = accuracy_score(all_labels, preds)

    # FPR and FNR
    n_neg = (all_labels == 0).sum()
    n_pos = (all_labels == 1).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    fpr = fp / max(n_neg, 1)
    fnr = fn / max(n_pos, 1)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
    }


def train_model(
    model: SiameseMatchingNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 1.0,
) -> Dict[str, list]:
    """
    Full training loop with ReduceLROnPlateau scheduling and early stopping.

    Returns:
        history: dict with train/val metrics per epoch
    """
    model = model.to(device)

    # ── Log class distribution (no pos_weight — mild imbalance doesn't need it,
    #    and it creates a flat loss landscape that causes training collapse) ──
    n_pos, n_neg = 0, 0
    for batch in train_loader:
        labels = batch["label"]
        n_pos += labels.sum().item()
        n_neg += (1 - labels).sum().item()
    print(f"Class distribution: {int(n_pos)} pos, {int(n_neg)} neg (ratio={n_neg/max(n_pos,1):.2f})")

    criterion = nn.BCEWithLogitsLoss()

    # ── Apply label smoothing to targets during training ──
    smooth_pos = 1.0 - label_smoothing
    smooth_neg = label_smoothing

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train with label smoothing ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input1 = batch["input1"].to(device)
            input2 = batch["input2"].to(device)
            labels = batch["label"].to(device)
            # Apply label smoothing
            smoothed = labels * smooth_pos + (1 - labels) * smooth_neg

            optimizer.zero_grad()
            logits = model(input1, input2)
            loss = criterion(logits, smoothed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"LR: {lr_now:.6f}"
        )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return history
