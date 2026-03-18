"""
Siamese CNN model for duplicate detection.
"""

import torch
import torch.nn as nn
from model import CharCNNEncoder


class SiameseCNN(nn.Module):
    """
    End-to-end Siamese network: embeds text with CNN, then classifies.
    """
    def __init__(self, vocab_size, embed_dim=32, num_filters=128, 
                 kernel_sizes=(3, 4, 5), dropout=0.3, cnn_output_dim=128):
        super(SiameseCNN, self).__init__()
        
        # Shared CNN encoder (processes both persons with same weights)
        self.encoder = CharCNNEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            output_dim=cnn_output_dim
        )
        
        # Classifier on top of concatenated features
        # Input: [emb1, emb2, |emb1-emb2|, emb1*emb2] = 4 * cnn_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, seq1, seq2):
        # Encode both sequences with SHARED encoder
        emb1 = self.encoder(seq1)
        emb2 = self.encoder(seq2)
        
        # Compute similarity features
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        
        # Concatenate all features
        features = torch.cat([emb1, emb2, diff, prod], dim=1)
        
        # Classify
        logits = self.classifier(features)
        return logits
