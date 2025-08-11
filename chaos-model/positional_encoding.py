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