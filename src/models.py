#!/usr/bin/env python3
"""Neural network models for DQN-based recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
    raise SystemExit(
        "Missing dependency: torch. Please run `conda env create -f environment.yml` "
        "and then `conda activate rl-course` before running src/models.py."
    ) from exc


class CandidateQNetwork(nn.Module):
    """Shared MLP over candidate features.

    Input shape:
      [batch_size, candidate_size, feature_dim]

    Output shape:
      [batch_size, candidate_size]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [B, K, F], got {tuple(states.shape)}")
        batch_size, candidate_size, feature_dim = states.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {feature_dim}")
        flat = states.reshape(batch_size * candidate_size, feature_dim)
        encoded = self.encoder(flat)
        q_values = self.q_head(encoded).reshape(batch_size, candidate_size)
        return q_values


class DuelingCandidateQNetwork(nn.Module):
    """Dueling DQN for candidate recommendation.

    The advantage branch is computed per candidate. The value branch uses a
    permutation-invariant mean pooling over encoded candidate features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.advantage_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(f"states must have shape [B, K, F], got {tuple(states.shape)}")
        batch_size, candidate_size, feature_dim = states.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {feature_dim}")

        flat = states.reshape(batch_size * candidate_size, feature_dim)
        encoded = self.encoder(flat).reshape(batch_size, candidate_size, 64)
        advantages = self.advantage_head(encoded).squeeze(-1)
        pooled = encoded.mean(dim=1)
        values = self.value_head(pooled)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


def build_model(
    model_type: str,
    feature_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for DQN model variants.

    `double_dqn` uses the same network architecture as `dqn`; the difference is
    in the target computation during training.
    """

    if model_type in {"dqn", "double_dqn"}:
        return CandidateQNetwork(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == "dueling_dqn":
        return DuelingCandidateQNetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError("model_type must be one of: dqn, double_dqn, dueling_dqn")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def infer_feature_dim(feature_path: Path) -> int:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
        raise SystemExit("Missing dependency: numpy.") from exc
    data = np.load(feature_path)
    return int(data["states"].shape[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test DQN model forward pass.")
    parser.add_argument("--model", choices=["dqn", "double_dqn", "dueling_dqn"], default="dqn")
    parser.add_argument("--feature_dim", type=int, default=0)
    parser.add_argument("--feature_path", type=Path, default=Path("outputs/features/train_features.npz"))
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dim = args.feature_dim or infer_feature_dim(args.feature_path)
    model = build_model(
        model_type=args.model,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    states = torch.randn(args.batch_size, args.candidate_size, feature_dim)
    q_values = model(states)
    print(f"model={args.model}")
    print(f"feature_dim={feature_dim}")
    print(f"parameters={count_parameters(model)}")
    print(f"input_shape={tuple(states.shape)}")
    print(f"q_shape={tuple(q_values.shape)}")


if __name__ == "__main__":
    main()

