import torch
from torch import nn


class GraphCritic(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_embedding)
