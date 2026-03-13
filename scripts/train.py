from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnnrl_mec.config import load_config
from gnnrl_mec.rl.ppo_trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["gnn", "mlp"], default=None)
    parser.add_argument("--action-type", type=str, choices=["discrete", "continuous_ratio", "continuous_mix"], default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.action_type is not None:
        cfg["env"]["action_type"] = args.action_type
    output_dir = run_training(cfg, project_root=ROOT, run_name=args.run_name)
    print(f"run_dir={output_dir}")


if __name__ == "__main__":
    main()
