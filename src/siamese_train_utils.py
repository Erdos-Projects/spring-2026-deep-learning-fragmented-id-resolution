"""
Training and evaluation functions for Siamese CNN model.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_epoch(model, loader, criterion, optimizer, device, sample_weights=None):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda or cpu)
        sample_weights: Optional numpy array of sample weights (for hard negative weighting).
                       Length must match total samples in loader.
                       Default is None (all samples weighted equally as 1.0).
    
    Returns:
        Tuple of (average_loss, accuracy, f1_score)
    
    Example with hard negative weights:
        from hard_negative_weights import compute_hard_negative_weights
        
        weights = compute_hard_negative_weights(train_df, hard_negative_weight=2.0)
        train_epoch(model, loader, criterion, optimizer, device, sample_weights=weights)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    weight_idx = 0  # Index to track position in sample_weights array
    
    for seq1, seq2, labels in tqdm(loader, desc="Training", leave=False):
        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(seq1, seq2).squeeze()
        
        # Calculate loss with optional sample weights
        if sample_weights is not None:
            # Get weights for this batch
            batch_size = labels.size(0)
            batch_weights = torch.tensor(
                sample_weights[weight_idx:weight_idx + batch_size],
                dtype=torch.float32,
                device=device
            )
            weight_idx += batch_size
            
            # Calculate per-sample loss and apply weights
            loss_per_sample = criterion(logits, labels)
            # If loss is already reduced, we need to expand it
            if loss_per_sample.dim() == 0:
                # Loss was already reduced (scalar), so we weight it by mean batch weight
                loss = loss_per_sample * batch_weights.mean()
            else:
                # Loss is per-sample, apply weights element-wise
                loss = (loss_per_sample * batch_weights).mean()
        else:
            # Standard unweighted loss
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


def train_with_hard_negatives(model, train_loader, val_loader, train_df, 
                              criterion, optimizer, scheduler, device, epochs,
                              hard_negative_weight=2.0, similarity_threshold=0.7):
    """
    Simplified training wrapper that handles hard negative weighting automatically.
    
    This function:
    1. Computes hard negative weights from training data
    2. Trains the model using weighted loss
    3. Evaluates on validation set
    4. Saves best checkpoint
    
    Args:
        model: SiameseCNN model to train
        train_loader: DataLoader for training pairs
        val_loader: DataLoader for validation pairs
        train_df: DataFrame containing training pair data (needed for hard negative detection)
        criterion: Loss function (BCELoss)
        optimizer: Adam or other optimizer
        scheduler: Learning rate scheduler (e.g., ReduceLROnPlateau)
        device: cuda or cpu
        epochs: Number of epochs to train
        hard_negative_weight: Weight multiplier for hard negatives (default 2.0)
        similarity_threshold: Threshold for near-match detection (default 0.7)
    
    Returns:
        Tuple of (model, history_dict, best_val_f1, best_epoch)
    
    Example:
        from hard_negative_weights import compute_hard_negative_weights
        
        # Computed once before training
        model, history, best_f1, best_epoch = train_with_hard_negatives(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,  # Raw DataFrame with pairs
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=30,
            hard_negative_weight=2.0,
            similarity_threshold=0.7
        )
    """
    from hard_negative_weights import compute_hard_negative_weights
    
    # Step 1: Compute hard negative weights
    print("\n" + "="*65)
    print("TRAINING WITH HARD NEGATIVE WEIGHTING")
    print("="*65)
    weights, is_hard = compute_hard_negative_weights(
        data=train_df,
        hard_negative_weight=hard_negative_weight,
        similarity_threshold=similarity_threshold,
        return_mask=True
    )
    
    # Step 2: Training loop
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train with weighted samples
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            sample_weights=weights  # Pass the computed weights
        )
        
        # Validate (no weights needed)
        val_loss, val_acc, val_f1, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            print(f"🎯 NEW BEST F1: {best_val_f1:.4f}")
    
    print(f"\n{'='*65}")
    print(f"Training complete! Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"{'='*65}\n")
    
    return model, history, best_val_f1, best_epoch
