class ChaosTransformer(nn.Module):
    def __init__(self, vocab_size, num_dims, dim_vec_size, emb_dim, n_heads, n_layers, ff_hidden, max_len=500):
        super().__init__()
        self.dim_emb = DimensionalEmbedding(vocab_size, num_dims, dim_vec_size)
        self.collapse = DimInteractionCollapse(num_dims)
        self.token_dim_to_emb = nn.Linear(dim_vec_size, emb_dim)

        self.positional_encoding = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        scalar_emb, dim_weights = self.dim_emb(x)               # [b, s, vec]
        x = self.token_dim_to_emb(scalar_emb)                    # [b, s, emb_dim]
        x = self.positional_encoding(x)
        x = self.encoder(x)
        logits = self.fc_out(x)
        return logits, dim_weights