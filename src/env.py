#!/usr/bin/env python3
"""Lightweight RL environment for crowdsourcing task recommendation.

The environment consumes feature files produced by src/features.py. It follows
the common reset/step API but does not require gym/gymnasium as a dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
    raise SystemExit(
        "Missing dependency: numpy. Please run `conda env create -f environment.yml` "
        "and then `conda activate rl-course` before running src/env.py."
    ) from exc


@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class CrowdRecommendationEnv:
    """Sequential offline recommendation environment.

    One step corresponds to one worker-arrival event. The state is a matrix with
    shape [candidate_size, feature_dim]. The action is an integer index selecting
    one candidate project.
    """

    def __init__(
        self,
        feature_path: str | Path,
        reward_type: str = "worker",
        episode_length: int = 1000,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.feature_path = Path(feature_path)
        self.reward_type = reward_type
        self.episode_length = episode_length
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        if reward_type not in {"worker", "requester", "sum"}:
            raise ValueError("reward_type must be one of: worker, requester, sum")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {self.feature_path}")

        data = np.load(self.feature_path)
        self.states = data["states"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.worker_rewards = data["worker_rewards"].astype(np.float32)
        self.requester_rewards = data["requester_rewards"].astype(np.float32)
        self.event_ids = data["event_ids"].astype(np.int64)
        self.worker_ids = data["worker_ids"].astype(np.int64)
        self.positive_project_ids = data["positive_project_ids"].astype(np.int64)
        self.candidate_project_ids = data["candidate_project_ids"].astype(np.int64)

        self.num_events, self.candidate_size, self.feature_dim = self.states.shape
        self.indices = np.arange(self.num_events)
        self.position = 0
        self.episode_step = 0
        self.current_index = 0

    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial state."""
        if self.shuffle:
            self.rng.shuffle(self.indices)
            self.position = 0
        elif self.position >= self.num_events:
            self.position = 0

        self.episode_step = 0
        self.current_index = int(self.indices[self.position])
        return self.states[self.current_index]

    def step(self, action: int) -> StepResult:
        """Apply an action and move to the next event."""
        if action < 0 or action >= self.candidate_size:
            raise ValueError(f"action must be in [0, {self.candidate_size}), got {action}")

        idx = self.current_index
        reward = self._reward(idx, action)
        info = self._info(idx, action)

        self.position += 1
        self.episode_step += 1
        done = self.episode_step >= self.episode_length or self.position >= self.num_events

        if done:
            next_state = np.zeros((self.candidate_size, self.feature_dim), dtype=np.float32)
        else:
            self.current_index = int(self.indices[self.position])
            next_state = self.states[self.current_index]

        return StepResult(next_state=next_state, reward=reward, done=done, info=info)

    def _reward(self, event_index: int, action: int) -> float:
        worker_reward = float(self.worker_rewards[event_index, action])
        requester_reward = float(self.requester_rewards[event_index, action])
        if self.reward_type == "worker":
            return worker_reward
        if self.reward_type == "requester":
            return requester_reward
        return worker_reward + requester_reward

    def _info(self, event_index: int, action: int) -> dict[str, Any]:
        label = int(self.labels[event_index])
        chosen_project_id = int(self.candidate_project_ids[event_index, action])
        positive_project_id = int(self.positive_project_ids[event_index])
        return {
            "event_id": int(self.event_ids[event_index]),
            "worker_id": int(self.worker_ids[event_index]),
            "action": int(action),
            "label": label,
            "hit": int(action == label),
            "chosen_project_id": chosen_project_id,
            "positive_project_id": positive_project_id,
            "worker_reward": float(self.worker_rewards[event_index, action]),
            "requester_reward": float(self.requester_rewards[event_index, action]),
        }

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.candidate_size))

    def greedy_label_action(self) -> int:
        """Return the logged positive action for debugging/oracle checks."""
        return int(self.labels[self.current_index])

    def get_state(self) -> np.ndarray:
        return self.states[self.current_index]

    def __len__(self) -> int:
        return self.num_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the recommendation environment.")
    parser.add_argument("--feature_path", type=Path, default=Path("outputs/features/train_features.npz"))
    parser.add_argument("--reward_type", choices=["worker", "requester", "sum"], default="worker")
    parser.add_argument("--episode_length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = CrowdRecommendationEnv(
        feature_path=args.feature_path,
        reward_type=args.reward_type,
        episode_length=args.episode_length,
        shuffle=False,
        seed=args.seed,
    )
    state = env.reset()
    print(
        f"Loaded {len(env)} events from {args.feature_path}; "
        f"state_shape={state.shape}; reward_type={args.reward_type}"
    )
    total_reward = 0.0
    for step in range(args.steps):
        action = env.sample_action()
        result = env.step(action)
        total_reward += result.reward
        print(
            f"step={step} action={action} reward={result.reward:.4f} "
            f"hit={result.info['hit']} done={result.done}"
        )
        if result.done:
            env.reset()
    print(f"sample_total_reward={total_reward:.4f}")


if __name__ == "__main__":
    main()

