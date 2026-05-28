#!/usr/bin/env python3
"""Train DQN variants for crowdsourcing task recommendation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from env import CrowdRecommendationEnv  # noqa: E402
from models import build_model, count_parameters  # noqa: E402


@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        candidate_size: int,
        feature_dim: int,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.candidate_size = candidate_size
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, candidate_size, feature_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, candidate_size, feature_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return TransitionBatch(
            states=torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.long, device=device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            next_states=torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return self.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN models for RL recommendation.")
    parser.add_argument("--features_dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=["dqn", "double_dqn", "dueling_dqn"], default="dqn")
    parser.add_argument("--reward_type", choices=["worker", "requester", "sum"], default="worker")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--min_buffer_size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=30000)
    parser.add_argument("--target_update_interval", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_max_events", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional output name. Defaults to {model}_{reward_type}_seed{seed}.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    ratio = min(step / decay_steps, 1.0)
    return start + ratio * (end - start)


def select_action(
    model: nn.Module,
    state: np.ndarray,
    epsilon: float,
    candidate_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[int, float]:
    if rng.random() < epsilon:
        return int(rng.integers(0, candidate_size)), float("nan")
    with torch.no_grad():
        state_tensor = torch.as_tensor(state[None, :, :], dtype=torch.float32, device=device)
        q_values = model(state_tensor).squeeze(0)
        action = int(torch.argmax(q_values).item())
        avg_q = float(q_values.mean().item())
    return action, avg_q


def compute_loss(
    *,
    model: nn.Module,
    target_model: nn.Module,
    batch: TransitionBatch,
    model_type: str,
    gamma: float,
    criterion: nn.Module,
) -> tuple[torch.Tensor, float]:
    q_values = model(batch.states)
    current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if model_type == "double_dqn":
            next_actions = torch.argmax(model(batch.next_states), dim=1)
            next_q = target_model(batch.next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = target_model(batch.next_states).max(dim=1).values
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * next_q

    loss = criterion(current_q, target_q)
    return loss, float(q_values.mean().detach().item())


def evaluate_policy(
    model: nn.Module,
    feature_path: Path,
    reward_type: str,
    device: torch.device,
    max_events: int = 20000,
) -> dict[str, float]:
    data = np.load(feature_path)
    states = data["states"]
    labels = data["labels"]
    worker_rewards = data["worker_rewards"]
    requester_rewards = data["requester_rewards"]
    candidate_project_ids = data["candidate_project_ids"]
    n = len(labels) if max_events <= 0 else min(len(labels), max_events)
    if n == 0:
        return {
            "eval_hit_at_1": 0.0,
            "eval_worker_reward": 0.0,
            "eval_requester_reward": 0.0,
            "eval_reward": 0.0,
            "eval_project_coverage": 0.0,
        }

    model.eval()
    actions: list[np.ndarray] = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_states = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
            q_values = model(batch_states)
            actions.append(torch.argmax(q_values, dim=1).cpu().numpy())
    chosen_actions = np.concatenate(actions)

    rows = np.arange(n)
    chosen_worker_rewards = worker_rewards[rows, chosen_actions]
    chosen_requester_rewards = requester_rewards[rows, chosen_actions]
    if reward_type == "worker":
        chosen_rewards = chosen_worker_rewards
    elif reward_type == "requester":
        chosen_rewards = chosen_requester_rewards
    else:
        chosen_rewards = chosen_worker_rewards + chosen_requester_rewards

    chosen_projects = candidate_project_ids[rows, chosen_actions]
    project_coverage = len(set(chosen_projects.tolist())) / max(
        len(set(candidate_project_ids[:n].reshape(-1).tolist())), 1
    )
    model.train()
    return {
        "eval_hit_at_1": float((chosen_actions == labels[:n]).mean()),
        "eval_worker_reward": float(chosen_worker_rewards.mean()),
        "eval_requester_reward": float(chosen_requester_rewards.mean()),
        "eval_reward": float(chosen_rewards.mean()),
        "eval_project_coverage": float(project_coverage),
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


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or f"{args.model}_{args.reward_type}_seed{args.seed}"
    checkpoints_dir = args.output_dir / "checkpoints"
    logs_dir = args.output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.features_dir / "train_features.npz"
    valid_path = args.features_dir / "valid_features.npz"
    env = CrowdRecommendationEnv(
        feature_path=train_path,
        reward_type=args.reward_type,
        episode_length=args.episode_length,
        shuffle=True,
        seed=args.seed,
    )

    model = build_model(
        model_type=args.model,
        feature_dim=env.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    target_model = build_model(
        model_type=args.model,
        feature_dim=env.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        candidate_size=env.candidate_size,
        feature_dim=env.feature_dim,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    recent_rewards: deque[float] = deque(maxlen=20)
    log_rows: list[dict[str, Any]] = []
    global_step = 0
    best_eval_reward = -math.inf

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "run_name": run_name,
            "device": str(device),
            "feature_dim": env.feature_dim,
            "candidate_size": env.candidate_size,
            "parameter_count": count_parameters(model),
        }
    )
    write_json(logs_dir / f"{run_name}_config.json", config)
    tqdm.write(
        f"Training {run_name}: device={device}, events={len(env)}, "
        f"state=({env.candidate_size}, {env.feature_dim}), params={count_parameters(model)}"
    )

    progress = trange(1, args.episodes + 1, desc=run_name, unit="episode")
    for episode in progress:
        state = env.reset()
        episode_reward = 0.0
        episode_worker_reward = 0.0
        episode_requester_reward = 0.0
        episode_hits = 0
        episode_steps = 0
        losses: list[float] = []
        q_values_seen: list[float] = []

        for _ in range(args.episode_length):
            epsilon = epsilon_by_step(
                global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            action, avg_q = select_action(
                model=model,
                state=state,
                epsilon=epsilon,
                candidate_size=env.candidate_size,
                device=device,
                rng=rng,
            )
            if not math.isnan(avg_q):
                q_values_seen.append(avg_q)

            result = env.step(action)
            buffer.add(state, action, result.reward, result.next_state, result.done)
            state = result.next_state
            episode_reward += result.reward
            episode_worker_reward += result.info["worker_reward"]
            episode_requester_reward += result.info["requester_reward"]
            episode_hits += result.info["hit"]
            episode_steps += 1

            if len(buffer) >= args.min_buffer_size:
                batch = buffer.sample(args.batch_size, device)
                loss, avg_train_q = compute_loss(
                    model=model,
                    target_model=target_model,
                    batch=batch,
                    model_type=args.model,
                    gamma=args.gamma,
                    criterion=criterion,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(float(loss.item()))
                q_values_seen.append(avg_train_q)

            global_step += 1
            if global_step % args.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            if result.done:
                break

        recent_rewards.append(episode_reward)
        log_row: dict[str, Any] = {
            "episode": episode,
            "global_step": global_step,
            "episode_reward": episode_reward,
            "moving_avg_reward": float(np.mean(recent_rewards)),
            "episode_worker_reward": episode_worker_reward,
            "episode_requester_reward": episode_requester_reward,
            "hit_rate": episode_hits / max(episode_steps, 1),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": epsilon_by_step(
                global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            ),
            "avg_q_value": float(np.mean(q_values_seen)) if q_values_seen else 0.0,
            "buffer_size": len(buffer),
        }

        if episode % args.eval_interval == 0 or episode == args.episodes:
            eval_metrics = evaluate_policy(
                model=model,
                feature_path=valid_path,
                reward_type=args.reward_type,
                device=device,
                max_events=args.eval_max_events,
            )
            log_row.update(eval_metrics)
            if eval_metrics["eval_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_reward"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "eval_metrics": eval_metrics,
                    },
                    checkpoints_dir / f"{run_name}_best.pt",
                )

        log_rows.append(log_row)
        progress.set_postfix(
            {
                "reward": f"{episode_reward:.3f}",
                "ma": f"{log_row['moving_avg_reward']:.3f}",
                "loss": f"{log_row['loss']:.5f}",
                "eps": f"{log_row['epsilon']:.3f}",
                "hit": f"{log_row['hit_rate']:.3f}",
                "eval": f"{log_row.get('eval_reward', float('nan')):.3f}",
            }
        )
        write_csv(logs_dir / f"{run_name}.csv", log_rows)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_eval_reward": best_eval_reward,
        },
        checkpoints_dir / f"{run_name}_last.pt",
    )
    tqdm.write(f"Wrote log to {logs_dir / f'{run_name}.csv'}")
    tqdm.write(f"Wrote checkpoints to {checkpoints_dir}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
