from dataclasses import dataclass


@dataclass
class ServerSpec:
    bandwidth: float
    max_cpu_frequency: float
