from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from gnnrl_mec.models.actor import DeviceActor
from gnnrl_mec.models.critic import GraphCritic
from gnnrl_mec.models.gnn_encoder import GNNEncoder


class GraphPPOPolicy(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_gnn_layers: int,
        num_actions: int,
        num_devices: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        self.encoder = GNNEncoder(node_dim, hidden_dim, num_gnn_layers, dropout)
        self.actor = DeviceActor(hidden_dim, num_actions)
        self.critic = GraphCritic(hidden_dim)

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs.x, obs.edge_index)
        device_embeddings = h[: self.num_devices]
        pooled = h.mean(dim=0)
        logits = self.actor(device_embeddings)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def act_deterministic(self, obs):
        logits, value = self.forward(obs)
        action = torch.argmax(logits, dim=-1)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def evaluate_actions(self, obs, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().mean()
        return log_prob, entropy, value
