import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # We'll use a linear layer to project the bidirectional output to a manageable size if needed
        # But for now, we'll just use the hidden state size * 2
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # x: [batch_size, seq_len]
        
        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embedding_dim]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [n_layers * 2, batch_size, hidden_dim]
        
        # We can use the last hidden state, or a pooling mechanism.
        # Max pooling over time often works well for capturing "did this feature occur?"
        # global_max_pooling
        encoded = torch.max(outputs, dim=1)[0] # [batch_size, hidden_dim * 2]
        
        return encoded

class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64):
        super(SiameseNetwork, self).__init__()
        
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        
        # Classifier
        # Input is |h1 - h2| (element-wise absolute difference)
        # Size is encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Output raw logit (use BCEWithLogitsLoss)
        )

    def forward(self, x1, x2):
        # Encode both inputs using the SAME encoder (shared weights)
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Similarity function: Absolute Difference
        # Captures "distance" between the two representations
        diff = torch.abs(h1 - h2)
        
        # Classify
        logits = self.classifier(diff)
        
        return logits

if __name__ == "__main__":
    # Simple test
    vocab_size = 50
    model = SiameseNetwork(vocab_size=vocab_size)
    
    # Mock data [batch_size, seq_len]
    x1 = torch.randint(0, vocab_size, (32, 20))
    x2 = torch.randint(0, vocab_size, (32, 20))
    
    output = model(x1, x2)
    print(f"Model architecture: {model}")
    print(f"Output shape: {output.shape}") # Should be [32, 1]
