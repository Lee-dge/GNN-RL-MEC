import torch
from torch import nn


class DeviceActor(nn.Module):
    def __init__(self, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, device_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(device_embeddings)
