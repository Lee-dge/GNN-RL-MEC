from __future__ import annotations

from pathlib import Path
import csv

import torch

from gnnrl_mec.env.mec_env import MECEnv
from gnnrl_mec.models.factory import build_policy
from gnnrl_mec.utils.logger import dump_json


def evaluate_policy(cfg: dict, checkpoint_path: Path, episodes: int) -> dict:
    device = torch.device(cfg["device"])
    env = MECEnv(cfg["env"], seed=int(cfg["seed"]) + 10_000)
    obs = env.reset()
    policy = build_policy(cfg, obs=obs, num_devices=env.num_devices, num_servers=env.num_servers).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    policy.load_state_dict(state_dict)
    policy.eval()

    ep_rewards = []
    ep_failures = []
    ep_delays = []
    ep_energies = []
    ep_offload_ratios = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        reward_sum = 0.0
        failure_sum = 0
        delay_sum = 0.0
        energy_sum = 0.0
        offload_ratio_sum = 0.0
        while not done:
            obs = obs.to(device)
            with torch.no_grad():
                action, _, _ = policy.act_deterministic(obs)
            obs, reward, done, info = env.step(action.cpu().tolist())
            reward_sum += reward
            failure_sum += info["failures"]
            delay_sum += info["total_delay"]
            energy_sum += info["total_energy"]
            offload_ratio_sum += info.get("offload_ratio_mean", 0.0)
        ep_rewards.append(reward_sum)
        ep_failures.append(failure_sum / max(env.episode_length * env.num_devices, 1))
        ep_delays.append(delay_sum / max(env.episode_length, 1))
        ep_energies.append(energy_sum / max(env.episode_length, 1))
        ep_offload_ratios.append(offload_ratio_sum / max(env.episode_length, 1))

    return {
        "episodes": episodes,
        "reward_mean": float(sum(ep_rewards) / max(len(ep_rewards), 1)),
        "failure_rate_mean": float(sum(ep_failures) / max(len(ep_failures), 1)),
        "delay_mean": float(sum(ep_delays) / max(len(ep_delays), 1)),
        "energy_mean": float(sum(ep_energies) / max(len(ep_energies), 1)),
        "offload_ratio_mean": float(sum(ep_offload_ratios) / max(len(ep_offload_ratios), 1)),
    }


def save_evaluation_csv(output_csv: Path, rows: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "model",
        "seed",
        "reward_mean",
        "failure_rate_mean",
        "delay_mean",
        "energy_mean",
        "offload_ratio_mean",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_evaluation_json(output_json: Path, payload: dict) -> None:
    dump_json(output_json, payload)
