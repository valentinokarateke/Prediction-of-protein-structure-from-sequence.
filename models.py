import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PairwiseDistanceHead(nn.Module):
    #predicts pairwise distances from per-residue representations
    #concatenates [z_i || z_j] -> mlp -> scalar distance, then symmetrizes

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, Z, mask=None):
        B, N, d = Z.shape

        Zi = Z.unsqueeze(2).expand(B, N, N, d)
        Zj = Z.unsqueeze(1).expand(B, N, N, d)
        pairs = torch.cat([Zi, Zj], dim=-1)

        dist = self.mlp(pairs).squeeze(-1)

        #symmetrize
        dist = (dist + dist.transpose(1, 2)) / 2.0

        #zero diagonal
        diag_mask = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(diag_mask, 0.0)

        return dist


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.name = "Transformer"

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=0
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.embed_dim, config.max_seq_len, config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )

        self.distance_head = PairwiseDistanceHead(
            config.embed_dim, config.head_hidden_dim
        )

    def forward(self, tokens, mask=None):
        padding_mask = (tokens == 0)

        x = self.embedding(tokens)
        x = self.pos_encoding(x)

        Z = self.encoder(x, src_key_padding_mask=padding_mask)

        dist = self.distance_head(Z, mask)
        return dist


class CNNModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.name = "CNN"

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=0
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.embed_dim, config.max_seq_len, config.dropout
        )

        #conv layers with residual connections
        layers = []
        in_channels = config.embed_dim
        for i in range(config.cnn_layers):
            out_channels = config.cnn_channels
            padding = config.cnn_kernel_size // 2
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, config.cnn_kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ))
            in_channels = out_channels
        self.conv_layers = nn.ModuleList(layers)

        if config.cnn_channels != config.embed_dim:
            self.proj = nn.Linear(config.cnn_channels, config.embed_dim)
        else:
            self.proj = nn.Identity()

        self.input_proj = (
            nn.Conv1d(config.embed_dim, config.cnn_channels, 1)
            if config.embed_dim != config.cnn_channels
            else nn.Identity()
        )

        self.distance_head = PairwiseDistanceHead(
            config.embed_dim, config.head_hidden_dim
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = self.pos_encoding(x)

        h = x.transpose(1, 2)
        h = self.input_proj(h)

        for conv_layer in self.conv_layers:
            residual = h
            h = conv_layer(h)
            h = h + residual

        Z = h.transpose(1, 2)
        Z = self.proj(Z)

        dist = self.distance_head(Z, mask)
        return dist


class BiLSTMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.name = "BiLSTM"

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=0
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.embed_dim, config.max_seq_len, config.dropout
        )

        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )

        #project forward+backward to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.distance_head = PairwiseDistanceHead(
            config.embed_dim, config.head_hidden_dim
        )

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = self.pos_encoding(x)

        lengths = (tokens != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=tokens.size(1)
        )

        Z = self.projection(h)

        dist = self.distance_head(Z, mask)
        return dist


def build_model(architecture: str, config: ModelConfig = None) -> nn.Module:
    if config is None:
        config = ModelConfig()

    architecture = architecture.lower()
    if architecture == 'transformer':
        return TransformerModel(config)
    elif architecture == 'cnn':
        return CNNModel(config)
    elif architecture == 'bilstm':
        return BiLSTMModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Choose from: transformer, cnn, bilstm")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = ModelConfig()
    for arch in ['transformer', 'cnn', 'bilstm']:
        model = build_model(arch, config)
        n_params = count_parameters(model)
        print(f"{model.name:12s}: {n_params:>10,} parameters")

        tokens = torch.randint(1, 21, (2, 50))
        mask = torch.ones(2, 50, 50)
        dist = model(tokens, mask)
        print(f"  Input:  tokens {tokens.shape}")
        print(f"  Output: dist   {dist.shape}")
        print()