from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnnrl_mec.config import load_config
from gnnrl_mec.experiments.evaluator import evaluate_policy, save_evaluation_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["gnn", "mlp"], default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.model is not None:
        cfg["model"]["name"] = args.model
    episodes = args.episodes if args.episodes is not None else int(cfg.get("experiment", {}).get("eval_episodes", 20))
    ckpt_path = Path(args.checkpoint)
    result = evaluate_policy(cfg, checkpoint_path=ckpt_path, episodes=episodes)
    output_json = ckpt_path.parent / "eval_metrics.json"
    save_evaluation_json(output_json, result)
    print(result)
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
