"""
Siamese network for record pair matching.

Architecture:
  1. Character-level embedding → CNN/BiLSTM encoder → fixed-size record embedding
  2. Pairwise comparison: |emb1 - emb2|, emb1 * emb2, cosine similarity
  3. MLP classifier → P(same person)

This addresses the teammate's concern about embedding categorical/textual
identity data: we learn character-level representations that naturally handle
typos, abbreviations, and nickname variations — which pretrained LMs like BERT
are NOT designed for.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNEncoder(nn.Module):
    """
    Character-level CNN encoder for identity strings.

    Produces a fixed-size embedding from a variable-length character sequence.
    The CNN captures local patterns (n-grams) which are ideal for detecting
    typos, abbreviations, and suffix variations.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        output_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, ks, padding=ks // 2)
            for ks in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        total_filters = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(total_filters, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) character indices
        Returns:
            (batch, output_dim) embedding vector
        """
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)          # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))           # (batch, num_filters, seq_len)
            c = c.max(dim=2).values         # (batch, num_filters) — global max pool
            conv_outs.append(c)

        cat = torch.cat(conv_outs, dim=1)   # (batch, total_filters)
        cat = self.dropout(cat)
        out = self.layer_norm(self.fc(cat))  # (batch, output_dim)
        return out


class SiameseMatchingNetwork(nn.Module):
    """
    Siamese network that compares two record embeddings and outputs
    a calibrated match probability.

    The comparison head uses:
      - Element-wise absolute difference |e1 - e2|
      - Element-wise product e1 * e2
      - Cosine similarity
    This multi-view comparison is more expressive than cosine alone.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 4, 5),
        encoder_output_dim: int = 128,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared encoder (weight-tied Siamese arms)
        self.encoder = CharCNNEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            output_dim=encoder_output_dim,
        )

        # Comparison head input: |diff| + product + cosine = 2*dim + 1
        comparison_dim = 2 * encoder_output_dim + 1

        self.classifier = nn.Sequential(
            nn.Linear(comparison_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all layers to improve gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.LongTensor) -> torch.Tensor:
        """Encode a single record (for candidate retrieval / ANN search)."""
        return self.encoder(x)

    def forward(
        self,
        input1: torch.LongTensor,
        input2: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            input1: (batch, seq_len) char indices for record 1
            input2: (batch, seq_len) char indices for record 2
        Returns:
            (batch, 1) logit (pass through sigmoid for probability)
        """
        e1 = self.encoder(input1)   # (batch, dim)
        e2 = self.encoder(input2)   # (batch, dim)

        # Multi-view comparison
        diff = torch.abs(e1 - e2)
        prod = e1 * e2
        cos_sim = F.cosine_similarity(e1, e2, dim=1).unsqueeze(1)

        combined = torch.cat([diff, prod, cos_sim], dim=1)
        logit = self.classifier(combined)
        return logit

    def predict_proba(
        self,
        input1: torch.LongTensor,
        input2: torch.LongTensor,
    ) -> torch.Tensor:
        """Return calibrated match probability."""
        logit = self.forward(input1, input2)
        return torch.sigmoid(logit)
