#!/usr/bin/env python3
"""Evaluate rule-based baselines on generated recommendation features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
    raise SystemExit(
        "Missing dependency: numpy. Please run `conda env create -f environment.yml` "
        "and then `conda activate rl-course` before running src/baselines.py."
    ) from exc


DEFAULT_BASELINES = [
    "random",
    "popular",
    "award",
    "deadline",
    "category_match",
    "oracle_worker",
    "oracle_requester",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline recommendation policies.")
    parser.add_argument("--features_dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
        choices=DEFAULT_BASELINES,
        help="Baseline policies to evaluate.",
    )
    return parser.parse_args()


def load_feature_names(features_dir: Path) -> list[str]:
    path = features_dir / "feature_names.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(features_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = features_dir / f"{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def feature_index(feature_names: list[str], name: str) -> int:
    try:
        return feature_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Feature `{name}` not found in feature_names.json") from exc


def argmax_with_random_tie(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Argmax per row with random tie-breaking."""
    max_scores = scores.max(axis=1, keepdims=True)
    ties = np.isclose(scores, max_scores)
    random_scores = rng.random(scores.shape)
    random_scores[~ties] = -1.0
    return random_scores.argmax(axis=1)


def choose_actions(
    baseline: str,
    data: dict[str, np.ndarray],
    feature_names: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    states = data["states"]
    num_events, candidate_size, _ = states.shape

    if baseline == "random":
        return rng.integers(low=0, high=candidate_size, size=num_events)

    if baseline == "popular":
        idx = feature_index(feature_names, "project_current_entries_log_norm")
        return argmax_with_random_tie(states[:, :, idx], rng)

    if baseline == "award":
        idx = feature_index(feature_names, "project_award_log_norm")
        return argmax_with_random_tie(states[:, :, idx], rng)

    if baseline == "deadline":
        idx = feature_index(feature_names, "project_time_left_ratio")
        return argmax_with_random_tie(-states[:, :, idx], rng)

    if baseline == "category_match":
        category_idx = feature_index(feature_names, "category_match")
        industry_idx = feature_index(feature_names, "industry_match")
        category_pref_idx = feature_index(feature_names, "worker_category_preference")
        industry_pref_idx = feature_index(feature_names, "worker_industry_preference")
        scores = (
            1.0 * states[:, :, category_idx]
            + 1.0 * states[:, :, industry_idx]
            + 0.5 * states[:, :, category_pref_idx]
            + 0.5 * states[:, :, industry_pref_idx]
        )
        return argmax_with_random_tie(scores, rng)

    if baseline == "oracle_worker":
        return argmax_with_random_tie(data["worker_rewards"], rng)

    if baseline == "oracle_requester":
        return argmax_with_random_tie(data["requester_rewards"], rng)

    raise ValueError(f"Unknown baseline: {baseline}")


def gather_by_action(values: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), actions]


def evaluate_actions(
    baseline: str,
    split: str,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    labels = data["labels"]
    worker_rewards = data["worker_rewards"]
    requester_rewards = data["requester_rewards"]
    states = data["states"]

    chosen_worker_rewards = gather_by_action(worker_rewards, actions)
    chosen_requester_rewards = gather_by_action(requester_rewards, actions)
    hit = actions == labels

    category_match_idx = feature_index(feature_names, "category_match")
    industry_match_idx = feature_index(feature_names, "industry_match")
    worker_quality_idx = feature_index(feature_names, "worker_quality")
    score_proxy = chosen_worker_rewards

    chosen_category_match = gather_by_action(states[:, :, category_match_idx], actions)
    chosen_industry_match = gather_by_action(states[:, :, industry_match_idx], actions)
    chosen_worker_quality = gather_by_action(states[:, :, worker_quality_idx], actions)

    chosen_projects = gather_by_action(data["candidate_project_ids"], actions)
    project_coverage = len(set(chosen_projects.tolist())) / max(
        len(set(data["candidate_project_ids"].reshape(-1).tolist())), 1
    )

    return {
        "split": split,
        "baseline": baseline,
        "num_events": int(labels.shape[0]),
        "hit_at_1": float(hit.mean()),
        "avg_worker_reward": float(chosen_worker_rewards.mean()),
        "avg_requester_reward": float(chosen_requester_rewards.mean()),
        "avg_total_reward": float((chosen_worker_rewards + chosen_requester_rewards).mean()),
        "category_match_rate": float(chosen_category_match.mean()),
        "industry_match_rate": float(chosen_industry_match.mean()),
        "avg_worker_quality": float(chosen_worker_quality.mean()),
        "project_coverage": float(project_coverage),
        "unique_recommended_projects": int(len(set(chosen_projects.tolist()))),
        "reward_std": float(score_proxy.std()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "split",
        "baseline",
        "hit_at_1",
        "avg_worker_reward",
        "avg_requester_reward",
        "category_match_rate",
        "industry_match_rate",
        "project_coverage",
    ]
    widths = {col: max(len(col), *(len(format_value(row[col])) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(format_value(row[col]).ljust(widths[col]) for col in columns))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    feature_names = load_feature_names(args.features_dir)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    rows: list[dict[str, Any]] = []
    for split in splits:
        data = load_npz(args.features_dir, split)
        for baseline in args.baselines:
            rng = np.random.default_rng(args.seed)
            actions = choose_actions(baseline, data, feature_names, rng)
            rows.append(evaluate_actions(baseline, split, data, actions, feature_names))

    suffix = args.split
    csv_path = args.output_dir / f"baseline_results_{suffix}.csv"
    json_path = args.output_dir / f"baseline_results_{suffix}.json"
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    print_table(rows)
    print(f"\nWrote baseline CSV to {csv_path}")
    print(f"Wrote baseline JSON to {json_path}")


if __name__ == "__main__":
    main()
