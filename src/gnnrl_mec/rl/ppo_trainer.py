from __future__ import annotations

from pathlib import Path
import csv
from datetime import datetime

import torch
from torch import nn

from gnnrl_mec.env.mec_env import MECEnv
from gnnrl_mec.models.factory import build_policy
from gnnrl_mec.rl.rollout import collect_rollout
from gnnrl_mec.utils.logger import dump_json
from gnnrl_mec.utils.seed import set_seed


def _compute_returns_and_advantages(buffer, gamma: float, gae_lambda: float):
    rewards = [item.reward for item in buffer.items]
    dones = [item.done for item in buffer.items]
    values = [float(item.value.item()) for item in buffer.items]
    returns = []
    advantages = []
    gae = 0.0
    next_value = 0.0
    for idx in reversed(range(len(rewards))):
        mask = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[idx])
        next_value = values[idx]
    adv = torch.tensor(advantages, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.tensor(returns, dtype=torch.float32)
    return ret, adv


def _new_run_dir(cfg: dict, project_root: Path, run_name: str | None) -> Path:
    exp_cfg = cfg.get("experiment", {})
    output_root = project_root / exp_cfg.get("output_dir", "outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"].get("name", "gnn").lower()
    seed = int(cfg["seed"])
    prefix = run_name if run_name else exp_cfg.get("name", "baseline")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_root / f"{prefix}_{model_name}_seed{seed}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_training(cfg: dict, project_root: Path, run_name: str | None = None) -> Path:
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    env = MECEnv(cfg["env"], seed=int(cfg["seed"]))
    obs = env.reset()
    policy = build_policy(cfg, obs=obs, num_devices=env.num_devices, num_servers=env.num_servers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg["train"]["learning_rate"]))
    output_dir = _new_run_dir(cfg, project_root, run_name=run_name)
    history = []

    for update in range(int(cfg["train"]["total_updates"])):
        buffer, stats = collect_rollout(
            env=env,
            policy=policy,
            steps=int(cfg["train"]["steps_per_update"]),
            device=device,
        )
        returns, advantages = _compute_returns_and_advantages(
            buffer,
            gamma=float(cfg["train"]["gamma"]),
            gae_lambda=float(cfg["train"]["gae_lambda"]),
        )

        for _ in range(int(cfg["train"]["epochs"])):
            for idx, transition in enumerate(buffer.items):
                obs_batch = transition.obs.to(device)
                action_batch = transition.action.to(device)
                old_log_prob = transition.log_prob.to(device)
                return_t = returns[idx].to(device)
                advantage_t = advantages[idx].to(device)

                new_log_prob, entropy, value = policy.evaluate_actions(obs_batch, action_batch)
                ratio = torch.exp(new_log_prob - old_log_prob)
                clip_ratio = float(cfg["train"]["clip_ratio"])
                surr1 = ratio * advantage_t
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_t
                policy_loss = -torch.min(surr1, surr2)
                value_loss = nn.functional.mse_loss(value, return_t)
                entropy_loss = -float(cfg["train"]["entropy_coef"]) * entropy
                loss = policy_loss + float(cfg["train"]["value_coef"]) * value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["train"]["max_grad_norm"]))
                optimizer.step()

        history.append({"update": update, **stats})
        if update % 10 == 0 or update == int(cfg["train"]["total_updates"]) - 1:
            print(
                f"update={update:03d} avg_reward={stats['avg_reward']:.4f} "
                f"avg_failures={stats['avg_failures']:.4f} "
                f"avg_delay={stats['avg_delay']:.4f} avg_energy={stats['avg_energy']:.4f} "
                f"avg_offload={stats['avg_offload_ratio']:.4f} episodes={stats['episodes']}"
            )

    checkpoint = {
        "state_dict": policy.state_dict(),
        "model_name": cfg["model"].get("name", "gnn").lower(),
        "seed": int(cfg["seed"]),
        "env": env.describe(),
    }
    torch.save(checkpoint, output_dir / "policy.pt")
    dump_json(output_dir / "train_history.json", {"history": history, "env": env.describe()})
    dump_json(output_dir / "run_config.json", cfg)
    with open(output_dir / "train_metrics.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["update", "avg_reward", "avg_failures", "avg_delay", "avg_energy", "avg_offload_ratio", "episodes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"saved model to {output_dir / 'policy.pt'}")
    return output_dir
