import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GNNEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        return h
