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
```

## Current Baseline

- Environment: stochastic MEC queueing and offloading simulator
- Graph: bipartite graph between devices and servers
- Policy: GNN encoder + per-device categorical actor
- Value: pooled graph critic
- Algorithm: PPO
- Metrics during train: reward, failures, delay, energy
- Batch benchmark utility: multi-seed + model comparison + CSV/PNG summary

## Next Research Steps

1. Add continuous offloading ratio control.
2. Add stronger baselines such as MLP-PPO, DQN, and heuristic policies.
3. Add ablation on graph depth, reward design, and topology scale.
4. Add evaluation plots and thesis-ready experiment scripts.
