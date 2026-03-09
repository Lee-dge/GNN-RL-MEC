from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnnrl_mec.config import load_config
from gnnrl_mec.env.mec_env import MECEnv
from gnnrl_mec.models.policy import GraphPPOPolicy


def main() -> None:
    cfg = load_config(ROOT / "configs/base.yaml")
    env = MECEnv(cfg["env"], seed=cfg["seed"])
    obs = env.reset()
    policy = GraphPPOPolicy(
        node_dim=obs.x.size(-1),
        hidden_dim=cfg["model"]["hidden_dim"],
        num_gnn_layers=cfg["model"]["gnn_layers"],
        num_actions=env.num_servers + 1,
        num_devices=env.num_devices,
        dropout=cfg["model"]["dropout"],
    )
    action, log_prob, value = policy.act(obs)
    next_obs, reward, done, info = env.step(action.tolist())
    print("nodes:", obs.num_nodes)
    print("device_action_shape:", tuple(action.shape))
    print("reward:", round(reward, 4))
    print("done:", done)
    print("value:", round(float(value.item()), 4))
    print("failures:", info["failures"])


if __name__ == "__main__":
    main()
