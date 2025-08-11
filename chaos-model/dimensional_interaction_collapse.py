class DimInteractionCollapse(nn.Module):
    def __init__(self, num_dimensions):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dim_weights):
        # dim_weights: [b, s, num_dims]
        M = torch.matmul(dim_weights.unsqueeze(3), dim_weights.unsqueeze(2))
        # M: [b, s, num_dims, num_dims]
        attn = self.softmax(M)
        v = attn.mean(dim=-1)  # [b, s, num_dims]
        return v
