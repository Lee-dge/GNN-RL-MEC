from __future__ import annotations

import torch

from gnnrl_mec.rl.buffer import RolloutBuffer, Transition


def collect_rollout(env, policy, steps: int, device: torch.device) -> tuple[RolloutBuffer, dict]:
    buffer = RolloutBuffer()
    obs = env.reset()
    total_reward = 0.0
    total_failures = 0
    total_delay = 0.0
    total_energy = 0.0
    total_offload_ratio = 0.0
    episodes = 0
    for _ in range(steps):
        obs = obs.to(device)
        with torch.no_grad():
            action, log_prob, value = policy.act(obs)
        next_obs, reward, done, info = env.step(action.cpu().tolist())
        buffer.add(
            Transition(
                obs=obs.cpu(),
                action=action.cpu(),
                log_prob=log_prob.cpu(),
                reward=reward,
                done=done,
                value=value.cpu(),
            )
        )
        total_reward += reward
        total_failures += info["failures"]
        total_delay += info["total_delay"]
        total_energy += info["total_energy"]
        total_offload_ratio += info.get("offload_ratio_mean", 0.0)
        obs = env.reset() if done else next_obs
        episodes += int(done)
    stats = {
        "avg_reward": total_reward / max(steps, 1),
        "avg_failures": total_failures / max(steps, 1),
        "avg_delay": total_delay / max(steps, 1),
        "avg_energy": total_energy / max(steps, 1),
        "avg_offload_ratio": total_offload_ratio / max(steps, 1),
        "episodes": episodes,
    }
    return buffer, stats
