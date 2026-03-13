from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta, Categorical, Dirichlet


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


class MLPContinuousPPOPolicy(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        hidden_dim: int,
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
        self.actor = nn.Linear(hidden_dim, num_devices * 2)
        self.critic = nn.Linear(hidden_dim, 1)

    def _flatten(self, obs) -> torch.Tensor:
        return obs.x.reshape(1, -1)

    def _distribution(self, obs):
        h = self.encoder(self._flatten(obs)).squeeze(0)
        raw = self.actor(h).reshape(self.num_devices, 2)
        alpha = torch.nn.functional.softplus(raw[:, 0]) + 1.0
        beta = torch.nn.functional.softplus(raw[:, 1]) + 1.0
        dist = Beta(alpha, beta)
        value = self.critic(h).squeeze(-1)
        return dist, value

    def act(self, obs):
        dist, value = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def act_deterministic(self, obs):
        dist, value = self._distribution(obs)
        action = dist.mean
        log_prob = dist.log_prob(torch.clamp(action, 1e-5, 1.0 - 1e-5)).sum()
        return action, log_prob, value

    def evaluate_actions(self, obs, action: torch.Tensor):
        dist, value = self._distribution(obs)
        safe_action = torch.clamp(action, 1e-5, 1.0 - 1e-5)
        log_prob = dist.log_prob(safe_action).sum()
        entropy = dist.entropy().mean()
        return log_prob, entropy, value


class MLPMixContinuousPPOPolicy(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        hidden_dim: int,
        num_devices: int,
        num_servers: int,
    ) -> None:
        super().__init__()
        self.num_devices = num_devices
        self.action_dim = num_servers + 1
        in_dim = num_nodes * node_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_devices * self.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def _flatten(self, obs) -> torch.Tensor:
        return obs.x.reshape(1, -1)

    def _distribution(self, obs):
        h = self.encoder(self._flatten(obs)).squeeze(0)
        raw = self.actor(h).reshape(self.num_devices, self.action_dim)
        concentration = torch.nn.functional.softplus(raw) + 1.0
        dist = Dirichlet(concentration)
        value = self.critic(h).squeeze(-1)
        return dist, value

    def act(self, obs):
        dist, value = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def act_deterministic(self, obs):
        dist, value = self._distribution(obs)
        action = dist.mean
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def evaluate_actions(self, obs, action: torch.Tensor):
        dist, value = self._distribution(obs)
        safe_action = torch.clamp(action, 1e-8, 1.0)
        safe_action = safe_action / safe_action.sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(safe_action).sum()
        entropy = dist.entropy().mean()
        return log_prob, entropy, value
