# GNN-RL MEC

An independent research project for mobile edge computing task offloading with graph neural networks and reinforcement learning.

## Scope

This project rebuilds the MEC simulation pipeline from scratch and only uses `COSIMv7.5` as a modeling reference. The first milestone is a runnable `GNN + PPO` baseline:

- graph state encoder with `torch_geometric`
- discrete offloading policy: local or one MEC server
- PPO training loop in PyTorch
- small MEC simulator aligned with your thesis topic

## Project Layout

```text
GNN-RL-mec/
├─ configs/
├─ scripts/
├─ src/gnnrl_mec/
└─ tests/
```

## Quick Start

Use your existing conda environment:

```powershell
conda activate gnnrl
cd D:\GNN-RL-mec
pip install -r requirements.txt
python scripts\sanity_check.py
python scripts\train.py --config configs\base.yaml
python scripts\evaluate.py --config configs\base.yaml --checkpoint outputs\<run_dir>\policy.pt
python scripts\run_baselines.py --config configs\base.yaml --seeds 42 43 44 --models gnn mlp
python scripts\train.py --config configs\base.yaml --model gnn --action-type continuous_ratio --run-name cont_v1
python scripts\run_baselines.py --config configs\base.yaml --models gnn mlp --action-type continuous_ratio --seeds 42 43 44 --run-name cont_v1
python scripts\train.py --config configs\base.yaml --model gnn --action-type continuous_mix --run-name mix_v1
python scripts\run_baselines.py --config configs\base.yaml --models gnn mlp --action-type continuous_mix --seeds 42 43 44 --run-name mix_v1
python scripts\run_baselines.py --config configs/gnn_advantage.yaml --models gnn mlp --action-type continuous_mix --seeds 40 41 42 43 44 --run-name gnn_adv_v1
```

## Current Baseline

- Environment: stochastic MEC queueing and offloading simulator
- Graph: bipartite graph between devices and servers
- Policy: GNN encoder + per-device categorical actor
- Continuous variant: per-device Beta actor for offloading ratio in [0, 1]
- Value: pooled graph critic
- Algorithm: PPO
- Metrics during train: reward, failures, delay, energy
- Batch benchmark utility: multi-seed + model comparison + CSV/PNG summary

## Continuous Offloading Mode

Set `env.action_type` to `continuous_ratio` (or pass `--action-type continuous_ratio`).

- Action definition: each device outputs one offloading ratio `alpha in [0,1]`
- Local load: `(1 - alpha) * current_task_load`
- Remote load: `alpha * current_task_load`
- Current implementation sends remote load to `env.continuous_target_server_index` for stable first-stage experiments

Set `env.action_type` to `continuous_mix` for multi-destination allocation.

- Action definition: each device outputs one vector over `[local, server_1, ..., server_M]`
- The vector is normalized to sum to 1 and represents load split ratios
- This mode jointly models offloading ratio and server assignment in one action

`configs/gnn_advantage.yaml` adds a harder setting for relational learning:

- dynamic server bandwidth/CPU jitter across time slots
- dynamic device-server link quality matrix
- sparse per-device connectivity (`max_links_per_device`)

## Next Research Steps

1. Add continuous offloading ratio control.
2. Add stronger baselines such as MLP-PPO, DQN, and heuristic policies.
3. Add ablation on graph depth, reward design, and topology scale.
4. Add evaluation plots and thesis-ready experiment scripts.
