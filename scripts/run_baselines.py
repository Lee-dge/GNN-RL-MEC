from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnnrl_mec.config import load_config
from gnnrl_mec.experiments.evaluator import evaluate_policy, save_evaluation_csv, save_evaluation_json
from gnnrl_mec.rl.ppo_trainer import run_training


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, var**0.5


def _plot_summary(rows: list[dict], output_dir: Path) -> None:
    metrics = [
        ("reward_mean", "Reward"),
        ("failure_rate_mean", "Failure Rate"),
        ("delay_mean", "Delay"),
        ("energy_mean", "Energy"),
    ]
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (metric_key, title) in zip(axes.flatten(), metrics):
        means = []
        errs = []
        for model in models:
            values = [float(row[metric_key]) for row in rows if row["model"] == model]
            mean, std = _mean_std(values)
            means.append(mean)
            errs.append(std)
        ax.bar(models, means, yerr=errs, capsize=4)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_compare.png", dpi=160)
    plt.close(fig)


def _write_aggregate(rows: list[dict], output_dir: Path) -> None:
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model = row["model"]
        by_model.setdefault(
            model,
            {
                "reward_mean": [],
                "failure_rate_mean": [],
                "delay_mean": [],
                "energy_mean": [],
                "offload_ratio_mean": [],
            },
        )
        for key in by_model[model]:
            by_model[model][key].append(float(row[key]))
    result = {}
    for model, metrics in by_model.items():
        result[model] = {}
        for key, values in metrics.items():
            mean, std = _mean_std(values)
            result[model][f"{key}_mean"] = mean
            result[model][f"{key}_std"] = std
    save_evaluation_json(output_dir / "aggregate.json", result)


def _save_run_table(rows: list[dict], output_dir: Path) -> None:
    output_csv = output_dir / "runs.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "model", "seed", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--models", type=str, nargs="+", default=["gnn", "mlp"])
    parser.add_argument("--action-type", type=str, choices=["discrete", "continuous_ratio", "continuous_mix"], default="discrete")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    base_cfg = load_config(ROOT / args.config)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else int(base_cfg.get("experiment", {}).get("eval_episodes", 20))
    summary_output = ROOT / base_cfg.get("experiment", {}).get("output_dir", "outputs") / f"{args.run_name}_summary"
    summary_output.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    run_rows = []
    for model_name in args.models:
        for seed in args.seeds:
            cfg = load_config(ROOT / args.config)
            cfg["seed"] = int(seed)
            cfg["model"]["name"] = model_name
            cfg["env"]["action_type"] = args.action_type
            if args.total_updates is not None:
                cfg["train"]["total_updates"] = int(args.total_updates)
            if args.steps_per_update is not None:
                cfg["train"]["steps_per_update"] = int(args.steps_per_update)
            if args.epochs is not None:
                cfg["train"]["epochs"] = int(args.epochs)
            run_dir = run_training(cfg, project_root=ROOT, run_name=args.run_name)
            result = evaluate_policy(cfg, checkpoint_path=run_dir / "policy.pt", episodes=eval_episodes)
            eval_row = {
                "run_name": args.run_name,
                "model": model_name,
                "seed": int(seed),
                "reward_mean": result["reward_mean"],
                "failure_rate_mean": result["failure_rate_mean"],
                "delay_mean": result["delay_mean"],
                "energy_mean": result["energy_mean"],
                "offload_ratio_mean": result["offload_ratio_mean"],
            }
            eval_rows.append(eval_row)
            run_rows.append(
                {
                    "run_name": args.run_name,
                    "model": model_name,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                }
            )
            print(
                f"model={model_name} seed={seed} reward={result['reward_mean']:.4f} "
                f"fail={result['failure_rate_mean']:.4f}"
            )

    save_evaluation_csv(summary_output / "evaluation.csv", eval_rows)
    _save_run_table(run_rows, summary_output)
    _write_aggregate(eval_rows, summary_output)
    _plot_summary(eval_rows, summary_output)
    print(f"saved summary to {summary_output}")


if __name__ == "__main__":
    main()
