from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnnrl_mec.config import load_config
from gnnrl_mec.env.mec_env import MECEnv


def test_env_reset_and_step():
    cfg = load_config(ROOT / "configs/base.yaml")
    env = MECEnv(cfg["env"], seed=cfg["seed"])
    obs = env.reset()
    assert obs.num_nodes == env.num_devices + env.num_servers
    next_obs, reward, done, info = env.step([0] * env.num_devices)
    assert next_obs.num_nodes == obs.num_nodes
    assert isinstance(reward, float)
    assert "failures" in info
