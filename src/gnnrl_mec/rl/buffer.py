from dataclasses import dataclass

import torch


@dataclass
class Transition:
    obs: object
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: float
    done: bool
    value: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.items: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def clear(self) -> None:
        self.items.clear()
