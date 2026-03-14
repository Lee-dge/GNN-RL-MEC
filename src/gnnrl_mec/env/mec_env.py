from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math

import numpy as np

from gnnrl_mec.env.device import DeviceSpec, DeviceState
from gnnrl_mec.env.graph_builder import build_bipartite_graph
from gnnrl_mec.env.reward import compute_reward
from gnnrl_mec.env.server import ServerSpec


class MECEnv:
    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.action_type = str(config.get("action_type", "discrete")).lower()
        self.max_links_per_device = int(config.get("max_links_per_device", len(config["servers"])))
        self.server_dynamics = config.get("server_dynamics", {})
        self.link_dynamics = config.get("link_dynamics", {})
        self.episode_length = int(config["episode_length"])
        self.time_weight = float(config["time_weight"])
        self.energy_weight = float(config["energy_weight"])
        self.failure_penalty = float(config["failure_penalty"])
        self.task_cfg = config["task"]
        self.wireless_cfg = config["wireless"]
        self.device_specs = self._expand_device_specs(config["devices"])
        self.server_specs = self._expand_server_specs(config["servers"])
        self.num_devices = len(self.device_specs)
        self.num_servers = len(self.server_specs)
        self.continuous_target_server_index = int(config.get("continuous_target_server_index", 0))
        if self.continuous_target_server_index < 0 or self.continuous_target_server_index >= self.num_servers:
            raise ValueError("continuous_target_server_index out of range")
        self.current_server_bandwidth = np.zeros(self.num_servers, dtype=np.float64)
        self.current_server_cpu = np.zeros(self.num_servers, dtype=np.float64)
        self.link_gain_matrix = np.ones((self.num_devices, self.num_servers), dtype=np.float64)
        self.server_recent_load = np.zeros(self.num_servers, dtype=np.float64)
        self.t = 0
        self.total_failures = 0
        self.devices: list[DeviceState] = []
        self.reset()

    def _expand_device_specs(self, entries: list[dict[str, Any]]) -> list[DeviceSpec]:
        specs: list[DeviceSpec] = []
        for entry in entries:
            count = int(entry["count"])
            spec = DeviceSpec(
                priority=float(entry["priority"]),
                cpu_frequency=float(entry["cpu_frequency"]),
                max_process_delay=float(entry["max_process_delay"]),
                max_load_capacity=float(entry["max_load_capacity"]),
                max_process_load_per_slot=float(entry["max_process_load_per_slot"]),
                energy_consume_per_ms=float(entry["energy_consume_per_ms"]),
                transmission_power=float(entry["transmission_power"]),
                channel_gain=float(entry["channel_gain"]),
            )
            for _ in range(count):
                specs.append(spec)
        return specs

    def _expand_server_specs(self, entries: list[dict[str, Any]]) -> list[ServerSpec]:
        specs: list[ServerSpec] = []
        for entry in entries:
            count = int(entry["count"])
            spec = ServerSpec(
                bandwidth=float(entry["bandwidth"]),
                max_cpu_frequency=float(entry["max_cpu_frequency"]),
            )
            for _ in range(count):
                specs.append(spec)
        return specs

    def reset(self):
        self.t = 0
        self.total_failures = 0
        self.devices = [DeviceState(spec=spec) for spec in self.device_specs]
        self._sample_server_state(init=True)
        self._sample_link_state(init=True)
        for device in self.devices:
            device.total_task_load = min(self._sample_task_load(), device.spec.max_load_capacity)
            device.current_task_load = min(device.total_task_load, device.spec.max_process_load_per_slot)
        return self._get_observation()

    def _sample_server_state(self, init: bool = False) -> None:
        enabled = bool(self.server_dynamics.get("enabled", False))
        bw_jitter = float(self.server_dynamics.get("bandwidth_jitter", 0.0))
        cpu_jitter = float(self.server_dynamics.get("cpu_jitter", 0.0))
        for idx, spec in enumerate(self.server_specs):
            if enabled:
                bw_scale = max(0.1, 1.0 + self.rng.normal(0.0, bw_jitter))
                cpu_scale = max(0.1, 1.0 + self.rng.normal(0.0, cpu_jitter))
            else:
                bw_scale = 1.0
                cpu_scale = 1.0
            self.current_server_bandwidth[idx] = spec.bandwidth * bw_scale
            self.current_server_cpu[idx] = spec.max_cpu_frequency * cpu_scale
        if init:
            self.server_recent_load[:] = 0.0

    def _sample_link_state(self, init: bool = False) -> None:
        enabled = bool(self.link_dynamics.get("enabled", False))
        if init or not enabled:
            low = float(self.link_dynamics.get("init_low", 0.6))
            high = float(self.link_dynamics.get("init_high", 1.4))
            self.link_gain_matrix = self.rng.uniform(low, high, size=(self.num_devices, self.num_servers))
            return
        noise = float(self.link_dynamics.get("noise_std", 0.08))
        self.link_gain_matrix += self.rng.normal(0.0, noise, size=self.link_gain_matrix.shape)
        self.link_gain_matrix = np.clip(self.link_gain_matrix, 0.2, 2.5)

    def _sample_task_load(self) -> float:
        load = self.rng.normal(
            self.task_cfg["avg_load_bits"],
            self.task_cfg["load_std_bits"],
        )
        return max(0.0, float(load))

    def _local_process(self, device: DeviceState, data_size: float) -> tuple[float, float]:
        cpu_cycle_per_bit = float(self.task_cfg["avg_cpu_cycle_per_bit"])
        coeff = float(self.wireless_cfg["effective_capacitance_coefficient"])
        time_ms = data_size * cpu_cycle_per_bit / device.spec.cpu_frequency * 1e-9 * 1000.0
        energy_per_cycle = coeff * math.pow(device.spec.cpu_frequency * 1e9, 2)
        energy_mj = data_size * cpu_cycle_per_bit * energy_per_cycle * 1000.0
        return time_ms, energy_mj

    def _remote_process(
        self,
        assignments: dict[int, list[tuple[int, float]]],
    ) -> dict[int, tuple[float, float]]:
        remote_result: dict[int, tuple[float, float]] = {}
        cpu_cycle_per_bit = float(self.task_cfg["avg_cpu_cycle_per_bit"])
        channel_noise = float(self.wireless_cfg["gaussian_channel_noise"])
        for server_idx, tasks in assignments.items():
            if not tasks:
                continue
            priority_sum = sum(self.devices[idx].spec.priority for idx, _ in tasks)
            for device_idx, remote_load in tasks:
                if remote_load <= 0.0:
                    continue
                device = self.devices[device_idx]
                share = device.spec.priority / max(priority_sum, 1e-6)
                bandwidth = self.current_server_bandwidth[server_idx] * share
                link_gain = device.spec.channel_gain * self.link_gain_matrix[device_idx][server_idx]
                upload_rate = bandwidth * 1e6 * math.log2(
                    1.0 + (
                        device.spec.transmission_power
                        * link_gain
                        / max(bandwidth * channel_noise, 1e-6)
                    )
                )
                upload_time = remote_load / max(upload_rate, 1e-6) * 1000.0
                upload_energy = device.spec.energy_consume_per_ms * upload_time
                cpu_frequency = self.current_server_cpu[server_idx] * share
                process_time = remote_load * cpu_cycle_per_bit / max(cpu_frequency, 1e-6) * 1e-9 * 1000.0
                prev_delay, prev_energy = remote_result.get(device_idx, (0.0, 0.0))
                remote_result[device_idx] = (max(prev_delay, upload_time + process_time), prev_energy + upload_energy)
        return remote_result

    def _step_discrete(self, actions: list[int]):
        if len(actions) != self.num_devices:
            raise ValueError(f"Expected {self.num_devices} device actions, got {len(actions)}")

        assignments = {server_idx: [] for server_idx in range(self.num_servers)}
        local_flags = []
        for device_idx, action in enumerate(actions):
            local = int(action) == 0
            local_flags.append(local)
            if not local:
                server_idx = int(action) - 1
                if server_idx < 0 or server_idx >= self.num_servers:
                    raise ValueError(f"Invalid server action {action} for device {device_idx}")
                assignments[server_idx].append((device_idx, self.devices[device_idx].current_task_load))

        remote_result = self._remote_process(assignments)
        self._update_server_recent_load(assignments)
        step_failures = 0
        total_delay = 0.0
        total_energy = 0.0
        total_offload_ratio = 0.0
        for device_idx, device in enumerate(self.devices):
            if local_flags[device_idx]:
                delay, energy = self._local_process(device, device.current_task_load)
            else:
                delay, energy = remote_result[device_idx]
                total_offload_ratio += 1.0
            device.current_delay = delay
            device.current_energy = energy
            device.finished = delay <= device.spec.max_process_delay
            if not device.finished:
                step_failures += 1
            total_delay += delay
            total_energy += energy

        self.total_failures += step_failures
        reward = compute_reward(
            total_delay=total_delay,
            total_energy=total_energy,
            failures=step_failures,
            time_weight=self.time_weight,
            energy_weight=self.energy_weight,
            failure_penalty=self.failure_penalty,
        )

        self.t += 1
        done = self.t >= self.episode_length
        info = {
            "t": self.t,
            "failures": step_failures,
            "total_failures": self.total_failures,
            "total_delay": total_delay,
            "total_energy": total_energy,
            "offload_ratio_mean": total_offload_ratio / max(self.num_devices, 1),
        }
        if not done:
            self._advance_state()
        obs = self._get_observation()
        return obs, reward, done, info

    def _step_continuous(self, actions: list[float]):
        if len(actions) != self.num_devices:
            raise ValueError(f"Expected {self.num_devices} device actions, got {len(actions)}")
        assignments = {server_idx: [] for server_idx in range(self.num_servers)}
        total_offload_ratio = 0.0
        for device_idx, action in enumerate(actions):
            ratio = float(np.clip(action, 0.0, 1.0))
            total_offload_ratio += ratio
            remote_load = ratio * self.devices[device_idx].current_task_load
            if remote_load > 0.0:
                assignments[self.continuous_target_server_index].append((device_idx, remote_load))

        remote_result = self._remote_process(assignments)
        self._update_server_recent_load(assignments)
        step_failures = 0
        total_delay = 0.0
        total_energy = 0.0
        for device_idx, device in enumerate(self.devices):
            ratio = float(np.clip(actions[device_idx], 0.0, 1.0))
            local_load = (1.0 - ratio) * device.current_task_load
            local_delay, local_energy = self._local_process(device, local_load)
            remote_delay, remote_energy = remote_result.get(device_idx, (0.0, 0.0))
            delay = max(local_delay, remote_delay)
            energy = local_energy + remote_energy
            device.current_delay = delay
            device.current_energy = energy
            device.finished = delay <= device.spec.max_process_delay
            if not device.finished:
                step_failures += 1
            total_delay += delay
            total_energy += energy

        self.total_failures += step_failures
        reward = compute_reward(
            total_delay=total_delay,
            total_energy=total_energy,
            failures=step_failures,
            time_weight=self.time_weight,
            energy_weight=self.energy_weight,
            failure_penalty=self.failure_penalty,
        )
        self.t += 1
        done = self.t >= self.episode_length
        info = {
            "t": self.t,
            "failures": step_failures,
            "total_failures": self.total_failures,
            "total_delay": total_delay,
            "total_energy": total_energy,
            "offload_ratio_mean": total_offload_ratio / max(self.num_devices, 1),
        }
        if not done:
            self._advance_state()
        obs = self._get_observation()
        return obs, reward, done, info

    def _step_continuous_mix(self, actions: list[list[float]]):
        if len(actions) != self.num_devices:
            raise ValueError(f"Expected {self.num_devices} device actions, got {len(actions)}")
        assignments = {server_idx: [] for server_idx in range(self.num_servers)}
        local_ratios = []
        for device_idx, action in enumerate(actions):
            if len(action) != self.num_servers + 1:
                raise ValueError(
                    f"Expected action dim {self.num_servers + 1} for device {device_idx}, got {len(action)}"
                )
            vec = np.array(action, dtype=np.float64)
            vec = np.clip(vec, 0.0, None)
            vec_sum = float(vec.sum())
            if vec_sum <= 1e-12:
                vec = np.zeros_like(vec)
                vec[0] = 1.0
            else:
                vec = vec / vec_sum
            local_ratios.append(float(vec[0]))
            task_load = self.devices[device_idx].current_task_load
            for server_idx in range(self.num_servers):
                remote_ratio = float(vec[server_idx + 1])
                remote_load = remote_ratio * task_load
                if remote_load > 0.0:
                    assignments[server_idx].append((device_idx, remote_load))

        remote_result = self._remote_process(assignments)
        self._update_server_recent_load(assignments)
        step_failures = 0
        total_delay = 0.0
        total_energy = 0.0
        total_offload_ratio = 0.0
        for device_idx, device in enumerate(self.devices):
            local_ratio = local_ratios[device_idx]
            local_load = local_ratio * device.current_task_load
            local_delay, local_energy = self._local_process(device, local_load)
            remote_delay, remote_energy = remote_result.get(device_idx, (0.0, 0.0))
            delay = max(local_delay, remote_delay)
            energy = local_energy + remote_energy
            device.current_delay = delay
            device.current_energy = energy
            device.finished = delay <= device.spec.max_process_delay
            if not device.finished:
                step_failures += 1
            total_delay += delay
            total_energy += energy
            total_offload_ratio += 1.0 - local_ratio

        self.total_failures += step_failures
        reward = compute_reward(
            total_delay=total_delay,
            total_energy=total_energy,
            failures=step_failures,
            time_weight=self.time_weight,
            energy_weight=self.energy_weight,
            failure_penalty=self.failure_penalty,
        )
        self.t += 1
        done = self.t >= self.episode_length
        info = {
            "t": self.t,
            "failures": step_failures,
            "total_failures": self.total_failures,
            "total_delay": total_delay,
            "total_energy": total_energy,
            "offload_ratio_mean": total_offload_ratio / max(self.num_devices, 1),
        }
        if not done:
            self._advance_state()
        obs = self._get_observation()
        return obs, reward, done, info

    def step(self, actions):
        if self.action_type == "continuous_mix":
            return self._step_continuous_mix(actions)
        if self.action_type == "continuous_ratio":
            return self._step_continuous(actions)
        return self._step_discrete(actions)

    def _advance_state(self) -> None:
        self._sample_server_state()
        self._sample_link_state()
        for device in self.devices:
            new_load = self._sample_task_load()
            remaining = 0.0 if device.finished else device.current_task_load
            total_load = min(remaining + new_load, device.spec.max_load_capacity)
            device.total_task_load = total_load
            device.current_task_load = min(total_load, device.spec.max_process_load_per_slot)
            device.current_delay = 0.0
            device.current_energy = 0.0
            device.finished = True

    def _update_server_recent_load(self, assignments: dict[int, list[tuple[int, float]]]) -> None:
        for server_idx in range(self.num_servers):
            total_load = sum(load for _, load in assignments.get(server_idx, []))
            self.server_recent_load[server_idx] = total_load

    def _get_observation(self):
        device_features = []
        for device in self.devices:
            best_link = float(np.max(self.link_gain_matrix[len(device_features)]))
            device_features.append(
                [
                    1.0,
                    0.0,
                    device.spec.priority / 3.0,
                    device.spec.cpu_frequency / 3.0,
                    device.total_task_load / max(device.spec.max_load_capacity, 1.0),
                    device.current_task_load / max(device.spec.max_process_load_per_slot, 1.0),
                    device.spec.max_process_delay,
                    best_link / 2.5,
                ]
            )
        server_features = []
        for server_idx, _server in enumerate(self.server_specs):
            avg_server_link = float(np.mean(self.link_gain_matrix[:, server_idx]))
            load_ratio = self.server_recent_load[server_idx] / max(1.0, float(self.task_cfg["avg_load_bits"]) * self.num_devices)
            server_features.append(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    load_ratio,
                    self.current_server_bandwidth[server_idx] / 20.0,
                    self.current_server_cpu[server_idx] / 10.0,
                ]
            )
        return build_bipartite_graph(
            device_features,
            server_features,
            edge_scores=self.link_gain_matrix.tolist(),
            max_links_per_device=self.max_links_per_device,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "episode_length": self.episode_length,
            "action_type": self.action_type,
            "num_devices": self.num_devices,
            "num_servers": self.num_servers,
            "device_spec": asdict(self.device_specs[0]),
            "server_spec": asdict(self.server_specs[0]),
        }
