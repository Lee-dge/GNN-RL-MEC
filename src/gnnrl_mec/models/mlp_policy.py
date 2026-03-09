from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class MLPPPOPolicy(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_devices: int,
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        in_dim = num_nodes * node_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_devices * num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.num_actions = num_actions

    def _flatten(self, obs) -> torch.Tensor:
        return obs.x.reshape(1, -1)

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self._flatten(obs)).squeeze(0)
        logits = self.actor(h).reshape(self.num_devices, self.num_actions)
        value = self.critic(h).squeeze(-1)
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
