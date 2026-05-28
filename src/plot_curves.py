#!/usr/bin/env python3
"""Plot training curves and comparison figures for the RL project."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_METRICS = [
    "episode_reward",
    "moving_avg_reward",
    "loss",
    "epsilon",
    "avg_q_value",
    "eval_reward",
    "eval_hit_at_1",
]

COMPARISON_METRICS = [
    "hit_at_1",
    "avg_worker_reward",
    "avg_requester_reward",
    "avg_total_reward",
    "project_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves and evaluation comparisons.")
    parser.add_argument("--logs_dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--tables_dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--train_logs", nargs="*", type=Path, default=[])
    parser.add_argument("--baseline_csv", type=Path, default=Path("outputs/tables/baseline_results_test.csv"))
    parser.add_argument("--eval_csv", type=Path, default=Path("outputs/tables/eval_results_test.csv"))
    parser.add_argument("--metrics", nargs="*", default=TRAIN_METRICS)
    parser.add_argument("--comparison_metrics", nargs="*", default=COMPARISON_METRICS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_train_logs(args: argparse.Namespace) -> list[Path]:
    if args.train_logs:
        return args.train_logs
    if not args.logs_dir.exists():
        return []
    return sorted(path for path in args.logs_dir.glob("*.csv") if not path.name.endswith("_config.csv"))


def plot_training_curve(log_path: Path, output_dir: Path, metrics: list[str]) -> list[Path]:
    rows = read_csv(log_path)
    if not rows:
        return []
    output_paths: list[Path] = []
    episodes = [int(float(row["episode"])) for row in rows]
    run_name = log_path.stem

    for metric in metrics:
        values = [parse_float(row.get(metric)) for row in rows]
        points = [(x, y) for x, y in zip(episodes, values) if y is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.title(f"{run_name}: {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = output_dir / "training" / f"{run_name}_{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180)
        plt.close()
        output_paths.append(path)
    return output_paths


def normalize_baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["method"] = row.get("baseline", "")
        item["source"] = "baseline"
        normalized.append(item)
    return normalized


def normalize_eval_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["method"] = row.get("run_name", row.get("checkpoint", "model"))
        item["source"] = "dqn"
        normalized.append(item)
    return normalized


def plot_comparison(
    baseline_csv: Path,
    eval_csv: Path,
    output_dir: Path,
    metrics: list[str],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    if baseline_csv.exists():
        rows.extend(normalize_baseline_rows(read_csv(baseline_csv)))
    if eval_csv.exists():
        rows.extend(normalize_eval_rows(read_csv(eval_csv)))
    if not rows:
        return []

    output_paths: list[Path] = []
    methods = [row["method"] for row in rows]
    colors = ["#6b7280" if row["source"] == "baseline" else "#2563eb" for row in rows]

    for metric in metrics:
        values = [parse_float(row.get(metric)) for row in rows]
        if all(value is None for value in values):
            continue
        ys = [0.0 if value is None else value for value in values]
        plt.figure(figsize=(max(9, len(methods) * 0.8), 5))
        plt.bar(methods, ys, color=colors)
        plt.ylabel(metric)
        plt.title(f"Method comparison: {metric}")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = output_dir / "comparison" / f"comparison_{metric}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180)
        plt.close()
        output_paths.append(path)
    return output_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for log_path in find_train_logs(args):
        generated.extend(plot_training_curve(log_path, args.output_dir, args.metrics))

    generated.extend(
        plot_comparison(
            baseline_csv=args.baseline_csv,
            eval_csv=args.eval_csv,
            output_dir=args.output_dir,
            metrics=args.comparison_metrics,
        )
    )

    if generated:
        print("Generated figures:")
        for path in generated:
            print(path)
    else:
        print("No figures generated. Check whether log/evaluation CSV files exist.")


if __name__ == "__main__":
    main()
