import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout, output_dim):
        super(CharCNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        output = self.fc(cat)
        output = self.layer_norm(output)
        return output
