"""
Model definitions for LoL item recommendation.

Architectures:
  - mlp         : ItemRecommender — flat MLP baseline
  - transformer : TransformerRecommender — per-player tokens + FiLM conditioning

Use build_model(config, input_dim, output_dim) to instantiate either arch.
Config is a plain dict saved inside every checkpoint under the key "arch_config".
"""

import torch
import torch.nn as nn


# ── MLP baseline ──────────────────────────────────────────────────────────────

class ItemRecommender(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list[int], dropout: float = 0.3):
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


# ── Transformer components ────────────────────────────────────────────────────

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Applies scale and shift to x conditioned on a context vector.
    Uses residual formulation: out = (1 + gamma) * x + beta
    """
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(condition_dim, 2 * feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) or (B, D)
        # condition: (B, condition_dim)
        gamma, beta = self.linear(condition).chunk(2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with FiLM conditioning after attention."""
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float, condition_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.film  = FiLM(condition_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention + residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop(attn_out)
        # FiLM conditioning on global context
        x = self.film(x, condition)
        # Pre-norm FFN + residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerRecommender(nn.Module):
    """
    Treats each of the 10 players as a token.
    Global features (game_time, gold_diff, buyer runes) condition
    each transformer block via FiLM.
    Output is computed from the buyer's token.

    Feature layout (matches preprocess.py):
      x[0:2]           → global: game_time_min, team_gold_diff
      x[2 : 2+10*448]  → 10 player blocks of 448 dims each
        player block:  [8 numeric | 208 items | 172 champs | 6 tags | 5 roles | 13 stats | 36 runes]
        is_buyer flag  → dim 7 within each player block
        rune features  → dims 412:448 within each player block
    """
    N_PLAYERS  = 10
    GLOBAL_DIM = 2    # game_time_min, team_gold_diff
    IS_BUYER_IDX = 7  # offset within each player block
    RUNE_START = 412  # 8 + 208 + 172 + 6 + 5 + 13
    RUNE_DIM   = 36

    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, ffn_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.player_dim = (input_dim - self.GLOBAL_DIM) // self.N_PLAYERS  # 448
        condition_dim   = self.GLOBAL_DIM + self.RUNE_DIM  # 38

        self.projection = nn.Linear(self.player_dim, d_model)

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout, d_model)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        global_feats = x[:, :self.GLOBAL_DIM]                                         # (B, 2)
        player_feats = x[:, self.GLOBAL_DIM:].reshape(B, self.N_PLAYERS, self.player_dim)  # (B, 10, 448)

        # Identify buyer token
        buyer_idx   = player_feats[:, :, self.IS_BUYER_IDX].argmax(dim=1)            # (B,)
        buyer_runes = player_feats[torch.arange(B), buyer_idx, self.RUNE_START:]     # (B, 36)

        # Build FiLM condition: [game_time, gold_diff, buyer_runes]
        condition = self.condition_mlp(
            torch.cat([global_feats, buyer_runes], dim=-1)                            # (B, 38)
        )                                                                              # (B, d_model)

        # Project players to d_model and run transformer
        tokens = self.projection(player_feats)                                        # (B, 10, d_model)
        for block in self.blocks:
            tokens = block(tokens, condition)

        # Output from buyer token
        buyer_token = tokens[torch.arange(B), buyer_idx]                             # (B, d_model)
        return self.head(self.norm_out(buyer_token))


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(arch_config: dict, input_dim: int, output_dim: int) -> nn.Module:
    """
    Instantiate a model from an arch_config dict.
    The dict is saved inside every checkpoint under 'arch_config'.

    MLP config:
        {"arch": "mlp", "hidden_dims": [1024, 512, 256], "dropout": 0.1}

    Transformer config:
        {"arch": "transformer", "d_model": 256, "n_heads": 8,
         "n_layers": 4, "ffn_dim": 512, "dropout": 0.1}
    """
    arch = arch_config.get("arch", "mlp")
    if arch == "mlp":
        return ItemRecommender(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=arch_config.get("hidden_dims", [1024, 512, 256]),
            dropout=arch_config.get("dropout", 0.1),
        )
    elif arch == "transformer":
        return TransformerRecommender(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=arch_config.get("d_model", 256),
            n_heads=arch_config.get("n_heads", 8),
            n_layers=arch_config.get("n_layers", 4),
            ffn_dim=arch_config.get("ffn_dim", 512),
            dropout=arch_config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'mlp' or 'transformer'.")


# ── Loss ──────────────────────────────────────────────────────────────────────

def reward_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rewards: torch.Tensor,
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    ce      = nn.functional.cross_entropy(logits, targets, reduction="none")
    weights = ((rewards - r_min) / (r_max - r_min)).clamp(0.0, 1.0)
    return (weights * ce).mean()
