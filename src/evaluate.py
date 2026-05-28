#!/usr/bin/env python3
"""Evaluate trained DQN checkpoints on feature splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN checkpoints.")
    parser.add_argument("--features_dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Optional explicit checkpoint paths. If omitted, evaluate *_best.pt.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output_name", type=str, default="")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_names(features_dir: Path) -> list[str]:
    return json.loads((features_dir / "feature_names.json").read_text(encoding="utf-8"))


def feature_index(feature_names: list[str], name: str) -> int:
    try:
        return feature_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Feature `{name}` not found in feature_names.json") from exc


def find_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return args.checkpoints
    paths = sorted(args.checkpoint_dir.glob("*_best.pt"))
    if not paths:
        paths = sorted(args.checkpoint_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoint files found in {args.checkpoint_dir}")
    return paths


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def infer_model_config(checkpoint: dict[str, Any], feature_dim: int) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    model_type = config.get("model", "dqn")
    hidden_dim = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.0))
    reward_type = config.get("reward_type", "unknown")
    run_name = config.get("run_name", "")
    return {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "reward_type": reward_type,
        "run_name": run_name,
    }


def predict_actions(
    model: torch.nn.Module,
    states: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    actions: list[np.ndarray] = []
    max_q_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            batch = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
            q_values = model(batch)
            max_q, action = torch.max(q_values, dim=1)
            actions.append(action.cpu().numpy())
            max_q_values.append(max_q.cpu().numpy())
    return np.concatenate(actions), np.concatenate(max_q_values)


def gather_by_action(values: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), actions]


def evaluate_predictions(
    *,
    checkpoint_path: Path,
    split: str,
    data: dict[str, np.ndarray],
    feature_names: list[str],
    actions: np.ndarray,
    max_q_values: np.ndarray,
    model_type: str,
    reward_type: str,
    run_name: str,
) -> dict[str, Any]:
    labels = data["labels"]
    states = data["states"]
    worker_rewards = data["worker_rewards"]
    requester_rewards = data["requester_rewards"]
    candidate_project_ids = data["candidate_project_ids"]

    chosen_worker_rewards = gather_by_action(worker_rewards, actions)
    chosen_requester_rewards = gather_by_action(requester_rewards, actions)
    chosen_projects = gather_by_action(candidate_project_ids, actions)

    category_match_idx = feature_index(feature_names, "category_match")
    industry_match_idx = feature_index(feature_names, "industry_match")
    worker_quality_idx = feature_index(feature_names, "worker_quality")
    chosen_category_match = gather_by_action(states[:, :, category_match_idx], actions)
    chosen_industry_match = gather_by_action(states[:, :, industry_match_idx], actions)
    chosen_worker_quality = gather_by_action(states[:, :, worker_quality_idx], actions)

    project_coverage = len(set(chosen_projects.tolist())) / max(
        len(set(candidate_project_ids.reshape(-1).tolist())), 1
    )
    hit = actions == labels

    return {
        "split": split,
        "run_name": run_name or checkpoint_path.stem,
        "checkpoint": str(checkpoint_path),
        "model": model_type,
        "reward_type": reward_type,
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
        "avg_max_q": float(max_q_values.mean()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["split", "run_name", "model", "reward_type", "hit_at_1", "avg_worker_reward", "avg_requester_reward"]
    widths = {col: max(len(col), *(len(format_value(row[col])) for row in rows)) for col in cols}
    header = "  ".join(col.ljust(widths[col]) for col in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(format_value(row[col]).ljust(widths[col]) for col in cols))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    feature_names = load_feature_names(args.features_dir)
    data_npz = np.load(args.features_dir / f"{args.split}_features.npz")
    data = {key: data_npz[key] for key in data_npz.files}
    feature_dim = int(data["states"].shape[-1])

    rows: list[dict[str, Any]] = []
    for checkpoint_path in find_checkpoints(args):
        checkpoint = load_checkpoint(checkpoint_path, device)
        cfg = infer_model_config(checkpoint, feature_dim)
        model = build_model(
            model_type=cfg["model_type"],
            feature_dim=feature_dim,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        actions, max_q_values = predict_actions(model, data["states"], args.batch_size, device)
        rows.append(
            evaluate_predictions(
                checkpoint_path=checkpoint_path,
                split=args.split,
                data=data,
                feature_names=feature_names,
                actions=actions,
                max_q_values=max_q_values,
                model_type=cfg["model_type"],
                reward_type=cfg["reward_type"],
                run_name=cfg["run_name"],
            )
        )

    output_name = args.output_name or f"eval_results_{args.split}"
    csv_path = args.output_dir / f"{output_name}.csv"
    json_path = args.output_dir / f"{output_name}.json"
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    print_table(rows)
    print(f"\nWrote evaluation CSV to {csv_path}")
    print(f"Wrote evaluation JSON to {json_path}")


if __name__ == "__main__":
    main()

