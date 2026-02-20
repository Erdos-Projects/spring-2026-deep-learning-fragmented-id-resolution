import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import NCVotersDataset
from model import SiameseNetwork
import numpy as np
from tqdm import tqdm

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x1, x2, labels in tqdm(loader, desc="Training"):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        
        logits = model(x1, x2).squeeze()
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Accuracy
        preds = torch.sigmoid(logits) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x1, x2, labels in tqdm(loader, desc="Validating"):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            
            logits = model(x1, x2).squeeze()
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), correct / total

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    MAX_LEN = 80
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    full_dataset = NCVotersDataset(
        data_path='data/processed/ncvoters_prepared.tsv',
        dpl_path='data/raw/ncvoters_DPL.tsv',
        ndpl_path='data/raw/ncvoters_NDPL.tsv',
        max_len=MAX_LEN
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    vocab_size = full_dataset.vocab.vocab_size
    model = SiameseNetwork(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Saved Best Model!")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
