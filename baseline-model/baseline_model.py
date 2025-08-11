import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Config ---
VOCAB_SIZE = 100  
EMBED_DIM = 16
NUM_HEADS = 2
SEQ_LEN = 10
NUM_LAYERS = 2
FF_DIM = 32

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, embed_dim).unsqueeze(0)
        angle_rates = 1 / (10000 ** (2 * (i // 2) / embed_dim))
        angle_rads = pos * angle_rates

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- Baseline Transformer Model ---
class BaselineTransformer(nn.Module):
    def __init__(self,vocab_size=VOCAB_SIZE,emb_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=FF_DIM,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        logits = self.fc_out(x)
        return logits