"""
Training and evaluation functions for Siamese CNN model.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda or cpu)
    
    Returns:
        Tuple of (average_loss, accuracy, f1_score)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for seq1, seq2, labels in tqdm(loader, desc="Training", leave=False):
        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(seq1, seq2).squeeze()
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Predictions
        preds = (logits > 0.5).float()
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to run on (cuda or cpu)
    
    Returns:
        Tuple of (loss, accuracy, f1_score, labels, predictions, probabilities)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seq1, seq2, labels in tqdm(loader, desc="Evaluating", leave=False):
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
            
            # Forward pass
            logits = model(seq1, seq2).squeeze()
            loss = criterion(logits, labels)
            running_loss += loss.item()
            
            # Predictions
            preds = (logits > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(logits.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, acc, f1, all_labels, all_preds, all_probs


def print_evaluation_report(labels, preds, probs=None):
    """
    Print detailed evaluation report.
    
    Args:
        labels: True labels
        preds: Predicted labels
        probs: Prediction probabilities (optional)
    """
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(labels, preds, 
                               target_names=['Non-Duplicate', 'Duplicate'],
                               digits=4))
    
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual  Neg   {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"        Pos   {cm[1,0]:6d} {cm[1,1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(labels, preds)
    
    print(f"\nAdditional Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return cm, precision, recall, specificity, f1
