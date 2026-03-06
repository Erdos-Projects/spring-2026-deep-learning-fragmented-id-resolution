import torch
import torch.nn as nn


def _parse_kernel_sizes(value):
    if isinstance(value, (list, tuple)):
        sizes = [int(v) for v in value]
    elif isinstance(value, str):
        sizes = [int(v.strip()) for v in value.split(",") if v.strip()]
    else:
        raise ValueError("cnn_kernel_sizes must be a list, tuple, or comma-separated string.")
    if not sizes:
        raise ValueError("cnn_kernel_sizes cannot be empty.")
    return sizes


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        return torch.max(outputs, dim=1)[0]


class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, cnn_channels=64, cnn_kernel_sizes=(3, 4, 5), dropout=0.1):
        super().__init__()
        kernel_sizes = _parse_kernel_sizes(cnn_kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=cnn_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = cnn_channels * len(kernel_sizes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            conv_out = self.activation(conv(embedded))
            pooled.append(torch.max(conv_out, dim=2)[0])
        encoded = torch.cat(pooled, dim=1)
        return self.dropout(encoded)


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        encoder_type="bilstm",
        cnn_channels=64,
        cnn_kernel_sizes=(3, 4, 5),
        classifier_hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
        elif encoder_type == "charcnn":
            self.encoder = CharCNNEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                cnn_channels=cnn_channels,
                cnn_kernel_sizes=cnn_kernel_sizes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        diff = torch.abs(h1 - h2)
        return self.classifier(diff)


if __name__ == "__main__":
    vocab_size = 50
    x1 = torch.randint(0, vocab_size, (8, 20))
    x2 = torch.randint(0, vocab_size, (8, 20))

    bilstm = SiameseNetwork(vocab_size=vocab_size, encoder_type="bilstm")
    cnn = SiameseNetwork(vocab_size=vocab_size, encoder_type="charcnn")
    print("BiLSTM output shape:", bilstm(x1, x2).shape)
    print("CharCNN output shape:", cnn(x1, x2).shape)
