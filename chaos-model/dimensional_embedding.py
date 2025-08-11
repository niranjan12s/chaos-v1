import torch
import torch.nn as nn
import json

class DimensionalEmbedding(nn.Module):
    def __init__(self, vocab_size, num_dims, dim_vec_size):
        super().__init__()
        self.token_dim_weights = nn.Embedding(vocab_size, num_dims)
        self.dimension_vectors = nn.Parameter(torch.randn(num_dims, dim_vec_size))

    def forward(self, token_ids):
        # token_ids: [batch, seq_len]
        weights = self.token_dim_weights(token_ids)           # [b, s, num_dims]
        # produces embedding [b, s, dim_vec_size]
        scalar_emb = torch.matmul(weights, self.dimension_vectors)  # collapse
        return scalar_emb, weights

