"""
Step 2: MLP for item recommendation.

Architecture:
  Input (2152) → Linear → BN → ReLU → Dropout
               → Linear → BN → ReLU → Dropout
               → Linear → BN → ReLU → Dropout
               → Linear (207) → output logits

Training: reward-weighted cross-entropy
  loss = -R * log(P(y | X))
"""

import torch
import torch.nn as nn


class ItemRecommender(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def reward_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rewards: torch.Tensor,
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    """
    Cross-entropy weighted by the globally normalized reward:
      weight = (R - R_min_global) / (R_max_global - R_min_global)
    """
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    weights = (rewards - r_min) / (r_max - r_min)
    weights = weights.clamp(0.0, 1.0)
    return (weights * ce).mean()
