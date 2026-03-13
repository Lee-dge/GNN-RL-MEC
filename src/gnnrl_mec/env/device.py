from dataclasses import dataclass


@dataclass
class DeviceSpec:
    priority: float
    cpu_frequency: float
    max_process_delay: float
    max_load_capacity: float
    max_process_load_per_slot: float
    energy_consume_per_ms: float
    transmission_power: float
    channel_gain: float


@dataclass
class DeviceState:
    spec: DeviceSpec
    total_task_load: float = 0.0
    current_task_load: float = 0.0
    current_delay: float = 0.0
    current_energy: float = 0.0
    finished: bool = True
