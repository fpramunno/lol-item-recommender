"""
Step 3: Training loop with wandb logging.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import logging
import itertools
import wandb

from model import ItemRecommender, reward_weighted_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Wandb configuration ───────────────────────────────────────────────────────

WANDB_PROJECT = "lol-item-recommender"   # project name on wandb
WANDB_ENTITY  = "francescopio"
WANDB_RUN_NAME = "mlp-baseline"         # run name — change for each experiment

# ── Hyperparameters ───────────────────────────────────────────────────────────

HIDDEN_DIMS   = [1024, 512, 256]
DROPOUT       = 0.3
BATCH_SIZE    = 512
TOTAL_STEPS   = 50000
LR            = 1e-3
VAL_SPLIT     = 0.1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FOG_MASK_PROB = 0.5   # probability of masking each enemy player

# Baselines to beat (computed by baseline.py)
BASELINE_RANDOM        = {"top1": 0.005, "top3": 0.014, "top5": 0.024}
BASELINE_MOST_FREQUENT = {"top1": 0.047, "top3": 0.119, "top5": 0.178}


# ── Fog of War masking ────────────────────────────────────────────────────────

def apply_fog_of_war(X_batch: torch.Tensor, input_dim: int, mask_prob: float = 0.5) -> torch.Tensor:
    """
    Zeros out the features of enemy players (indices 5-9) with probability mask_prob.
    Simulates fog of war: at inference time you may not see enemy items/gold/stats.
    """
    per_player = (input_dim - 2) // 10
    B = X_batch.shape[0]
    for i in range(5, 10):  # enemy players: 0-indexed 5..9
        p_start = 2 + i * per_player
        p_end   = p_start + per_player
        mask = torch.rand(B, device=X_batch.device) < mask_prob
        X_batch = X_batch.clone()
        X_batch[mask, p_start:p_end] = 0.0
    return X_batch


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, R: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.R = torch.tensor(R, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.R[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    log.info(f"Device: {DEVICE}")

    # Load data
    log.info("Loading data...")
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    R = np.load(PROCESSED_DIR / "R.npy")

    input_dim  = X.shape[1]
    output_dim = int(y.max()) + 1
    r_min = float(R.min())
    r_max = float(R.max())
    log.info(f"Input dim: {input_dim}, Output dim (n_items): {output_dim}")
    log.info(f"Total samples: {len(X)}")
    log.info(f"Global R range: [{r_min:.3f}, {r_max:.3f}]")

    # Train/val split
    dataset = LoLDataset(X, y, R)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info(f"Train: {train_size} | Val: {val_size}")

    # Model
    model = ItemRecommender(input_dim, output_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # ── Wandb init ────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "total_steps": TOTAL_STEPS,
            "lr": LR,
            "val_split": VAL_SPLIT,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "n_params": n_params,
            "n_samples": len(X),
            "device": DEVICE,
        },
        # What to monitor automatically
        monitor_gym=False,
        save_code=True,   # saves train.py and model.py in the run
    )

    # Wandb tracks model gradients and weights every 500 steps
    wandb.watch(model, log="gradients", log_freq=500)

    # Log baselines as horizontal reference lines
    wandb.log({
        "baseline/random_top1":        BASELINE_RANDOM["top1"],
        "baseline/random_top3":        BASELINE_RANDOM["top3"],
        "baseline/random_top5":        BASELINE_RANDOM["top5"],
        "baseline/mostfreq_top1":      BASELINE_MOST_FREQUENT["top1"],
        "baseline/mostfreq_top3":      BASELINE_MOST_FREQUENT["top3"],
        "baseline/mostfreq_top5":      BASELINE_MOST_FREQUENT["top5"],
    }, step=0)

    best_val_loss = float("inf")
    global_step = 0
    LOG_TRAIN_EVERY = 50
    LOG_VAL_EVERY   = 1000

    # Training accumulators
    running_loss    = 0.0
    running_correct = 0
    running_steps   = 0

    model.train()
    for X_batch, y_batch, R_batch in itertools.islice(
        itertools.cycle(train_loader), TOTAL_STEPS
    ):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            R_batch = R_batch.to(DEVICE)

            optimizer.zero_grad()
            X_batch = apply_fog_of_war(X_batch, input_dim, FOG_MASK_PROB)
            logits = model(X_batch)
            loss = reward_weighted_loss(logits, y_batch, R_batch, r_min, r_max)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item()
            running_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            running_steps   += 1
            global_step     += 1

            # ── Log training every 50 steps ───────────────────────────────────
            if global_step % LOG_TRAIN_EVERY == 0:
                train_loss = running_loss / running_steps
                train_acc  = running_correct / (running_steps * BATCH_SIZE)
                lr_current = optimizer.param_groups[0]["lr"]

                log.info(f"Step {global_step} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}")
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc":  train_acc,
                    "train/lr":   lr_current,
                }, step=global_step)

                running_loss = running_correct = running_steps = 0

            # ── Validation every 1000 steps ───────────────────────────────────
            if global_step % LOG_VAL_EVERY == 0:
                model.eval()
                val_loss = 0.0
                top1 = top3 = top5 = 0
                reward_num = reward_den = 0.0

                with torch.no_grad():
                    for X_val, y_val, R_val in val_loader:
                        X_val = X_val.to(DEVICE)
                        y_val = y_val.to(DEVICE)
                        R_val = R_val.to(DEVICE)

                        logits_val = model(X_val)
                        val_loss += reward_weighted_loss(logits_val, y_val, R_val, r_min, r_max).item()

                        topk = logits_val.topk(5, dim=1).indices
                        correct1 = (topk[:, :1] == y_val.unsqueeze(1)).any(dim=1)
                        correct3 = (topk[:, :3] == y_val.unsqueeze(1)).any(dim=1)
                        correct5 = (topk[:, :5] == y_val.unsqueeze(1)).any(dim=1)
                        top1 += correct1.sum().item()
                        top3 += correct3.sum().item()
                        top5 += correct5.sum().item()

                        reward_num += (R_val * correct1.float()).sum().item()
                        reward_den += R_val.abs().sum().item()

                val_loss  /= len(val_loader)
                top1_acc   = top1 / val_size
                top3_acc   = top3 / val_size
                top5_acc   = top5 / val_size
                reward_acc = reward_num / reward_den if reward_den > 0 else 0.0

                scheduler.step(val_loss)
                model.train()

                log.info(
                    f"Step {global_step} [VAL] | "
                    f"val_loss={val_loss:.4f} "
                    f"top1={top1_acc:.3f} top3={top3_acc:.3f} top5={top5_acc:.3f} "
                    f"reward_acc={reward_acc:.3f}"
                )

                wandb.log({
                    "val/loss":       val_loss,
                    "val/top1":       top1_acc,
                    "val/top3":       top3_acc,
                    "val/top5":       top5_acc,
                    "val/reward_acc": reward_acc,
                    "vs_baseline/top1_vs_mostfreq": top1_acc - BASELINE_MOST_FREQUENT["top1"],
                    "vs_baseline/top3_vs_mostfreq": top3_acc - BASELINE_MOST_FREQUENT["top3"],
                    "vs_baseline/top5_vs_mostfreq": top5_acc - BASELINE_MOST_FREQUENT["top5"],
                }, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    path = CHECKPOINT_DIR / "best_model.pt"
                    torch.save({
                        "step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "top1_acc": top1_acc,
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                    }, path)
                    log.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
                    wandb.save(str(path))

    wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    train()
