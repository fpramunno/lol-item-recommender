"""
Step 3: Training loop with wandb logging.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import itertools
import wandb
import yaml

from model import build_model, reward_weighted_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR    = Path("data/processed/shards")
CHECKPOINTS_ROOT = Path("checkpoints")

# ── Wandb configuration ───────────────────────────────────────────────────────

WANDB_PROJECT  = "lol-item-recommender"
WANDB_ENTITY   = "francescopio"
WANDB_RUN_NAME = "mlp-baseline"

# ── Hyperparameters ───────────────────────────────────────────────────────────

HIDDEN_DIMS   = [1024, 512, 256]
DROPOUT       = 0.3
BATCH_SIZE    = 512
TOTAL_STEPS   = 50000
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FOG_MASK_PROB = 0.5

BASELINE_RANDOM        = {"top1": 0.005, "top3": 0.014, "top5": 0.024}
BASELINE_MOST_FREQUENT = {"top1": 0.047, "top3": 0.119, "top5": 0.178}


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShardDataset(Dataset):
    """Loads one shard (X_NNN.npy, y_NNN.npy, R_NNN.npy) fully into RAM."""
    def __init__(self, shard_dir: Path, shard_id: int):
        self.X = torch.tensor(np.load(shard_dir / f"X_{shard_id:03d}.npy"), dtype=torch.float32)
        self.y = torch.tensor(np.load(shard_dir / f"y_{shard_id:03d}.npy"), dtype=torch.long)
        self.R = torch.tensor(np.load(shard_dir / f"R_{shard_id:03d}.npy"), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.R[idx]


# ── Fog of War masking ────────────────────────────────────────────────────────

def apply_fog_of_war(X_batch: torch.Tensor, input_dim: int, mask_prob: float) -> torch.Tensor:
    per_player = (input_dim - 2) // 10
    B = X_batch.shape[0]
    X_batch = X_batch.clone()
    for i in range(5, 10):
        p_start = 2 + i * per_player
        p_end   = p_start + per_player
        mask = torch.rand(B, device=X_batch.device) < mask_prob
        X_batch[mask, p_start:p_end] = 0.0
    return X_batch


# ── Training ──────────────────────────────────────────────────────────────────

def _build_run_name(arch_config: dict) -> str:
    arch = arch_config["arch"]
    if arch == "transformer":
        return (f"transformer-d{arch_config['d_model']}"
                f"-h{arch_config['n_heads']}"
                f"-l{arch_config['n_layers']}"
                f"-ffn{arch_config['ffn_dim']}")
    else:
        dims = "-".join(str(d) for d in arch_config["hidden_dims"])
        return f"mlp-{dims}"


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str,   default=None,   help="Path to YAML config file (e.g. configs/transformer.yaml)")
    parser.add_argument("--resume",        type=str,   default=None,   help="Path to checkpoint to resume from")
    parser.add_argument("--device",        type=str,   default=DEVICE)
    parser.add_argument("--num-workers",   type=int,   default=0)
    # CLI overrides (override YAML values when provided)
    parser.add_argument("--arch",          type=str,   default=None,   choices=["mlp", "transformer"])
    parser.add_argument("--hidden-dims",   type=int,   nargs="+",      default=None)
    parser.add_argument("--d-model",       type=int,   default=None)
    parser.add_argument("--n-heads",       type=int,   default=None)
    parser.add_argument("--n-layers",      type=int,   default=None)
    parser.add_argument("--ffn-dim",       type=int,   default=None)
    parser.add_argument("--dropout",       type=float, default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--total-steps",   type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--fog-mask-prob", type=float, default=None)
    parser.add_argument("--lr-patience",   type=int,   default=None)
    args = parser.parse_args()

    # Load config: YAML base, then CLI overrides on top
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        log.info(f"Loaded config: {args.config}")
    else:
        log.warning("No --config provided, using built-in defaults.")

    def _get(key, default):
        cli_val = getattr(args, key.replace("-", "_"), None)
        return cli_val if cli_val is not None else cfg.get(key, default)

    arch_config = {
        "arch":        _get("arch",        "mlp"),
        "hidden_dims": _get("hidden_dims", HIDDEN_DIMS),
        "d_model":     _get("d_model",     256),
        "n_heads":     _get("n_heads",     8),
        "n_layers":    _get("n_layers",    4),
        "ffn_dim":     _get("ffn_dim",     512),
        "dropout":     _get("dropout",     DROPOUT),
    }
    dropout       = arch_config["dropout"]
    batch_size    = _get("batch_size",    BATCH_SIZE)
    total_steps   = _get("total_steps",   TOTAL_STEPS)
    lr            = _get("lr",            LR)
    fog_mask_prob = _get("fog_mask_prob", FOG_MASK_PROB)
    lr_patience   = _get("lr_patience",   3)
    num_workers   = args.num_workers
    device        = args.device
    hidden_dims   = arch_config["hidden_dims"]

    from datetime import datetime
    run_time       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_name    = Path(args.config).stem if args.config else arch_config["arch"]
    CHECKPOINT_DIR = CHECKPOINTS_ROOT / config_name / run_time
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {device} | Checkpoint dir: {CHECKPOINT_DIR}")

    # Discover shards
    shard_files = sorted(PROCESSED_DIR.glob("X_*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {PROCESSED_DIR}. Run: python shard_dataset.py")
    n_shards = len(shard_files)
    log.info(f"Found {n_shards} shards in {PROCESSED_DIR}")

    # Last 2 shards → validation, rest → training
    train_ids = list(range(n_shards - 2))
    val_ids   = [n_shards - 2, n_shards - 1]

    # Get dims and R range from shard 0
    X0 = np.load(PROCESSED_DIR / f"X_{0:03d}.npy")
    input_dim  = X0.shape[1]
    output_dim = int(max(np.load(PROCESSED_DIR / f"y_{i:03d}.npy").max() for i in range(n_shards))) + 1
    r_min      = float(min(np.load(PROCESSED_DIR / f"R_{i:03d}.npy").min() for i in range(n_shards)))
    r_max      = float(max(np.load(PROCESSED_DIR / f"R_{i:03d}.npy").max() for i in range(n_shards)))
    del X0
    log.info(f"Input dim: {input_dim} | Output dim: {output_dim} | R: [{r_min:.3f}, {r_max:.3f}]")

    # Validation size (count only, don't load into RAM yet)
    from torch.utils.data import ConcatDataset
    val_size = sum(len(np.load(PROCESSED_DIR / f"y_{sid:03d}.npy")) for sid in val_ids)
    log.info(f"Train shards: {len(train_ids)} | Val samples: {val_size}")

    # Model
    model     = build_model(arch_config, input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, factor=0.5)
    n_params  = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # Resume
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            # search legacy checkpoints root for backwards compatibility
            resume_path = CHECKPOINTS_ROOT / args.resume
        ckpt = torch.load(resume_path, map_location=device)
        # If resuming, use the arch from checkpoint (ignore CLI arch flags)
        if "arch_config" in ckpt:
            arch_config = ckpt["arch_config"]
            model = build_model(arch_config, input_dim, output_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = ckpt.get("step", 0)
        log.info(f"Resumed from {args.resume} at step {global_step} (arch={arch_config['arch']})")

    # Wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=_build_run_name(arch_config),
        config={
            "hidden_dims":   hidden_dims,
            "dropout":       dropout,
            "batch_size":    batch_size,
            "total_steps":   total_steps,
            "lr":            lr,
            "fog_mask_prob": fog_mask_prob,
            "input_dim":     input_dim,
            "output_dim":    output_dim,
            "n_params":      n_params,
            "n_shards":      n_shards,
            "device":        device,
        },
        monitor_gym=False,
        save_code=True,
    )
    wandb.log({
        "baseline/random_top1":    BASELINE_RANDOM["top1"],
        "baseline/random_top3":    BASELINE_RANDOM["top3"],
        "baseline/random_top5":    BASELINE_RANDOM["top5"],
        "baseline/mostfreq_top1":  BASELINE_MOST_FREQUENT["top1"],
        "baseline/mostfreq_top3":  BASELINE_MOST_FREQUENT["top3"],
        "baseline/mostfreq_top5":  BASELINE_MOST_FREQUENT["top5"],
    }, step=0)

    best_val_loss   = float("inf")
    shard_epoch     = 0
    LOG_TRAIN_EVERY = 50
    running_loss = running_correct = running_steps = 0

    def safe_wandb_log(data, step):
        try:
            wandb.log(data, step=step)
        except Exception as e:
            log.warning(f"wandb.log failed (step={step}): {e}")

    def run_validation():
        model.eval()
        val_loss = 0.0
        top1 = top3 = top5 = 0
        reward_num = reward_den = 0.0
        n_batches = 0
        with torch.no_grad():
            for sid in val_ids:
                X_mm = np.load(PROCESSED_DIR / f"X_{sid:03d}.npy", mmap_mode="r")
                y_mm = np.load(PROCESSED_DIR / f"y_{sid:03d}.npy", mmap_mode="r")
                R_mm = np.load(PROCESSED_DIR / f"R_{sid:03d}.npy", mmap_mode="r")
                n = len(y_mm)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    xv = torch.tensor(np.array(X_mm[start:end]), dtype=torch.float32).to(device)
                    yv = torch.tensor(np.array(y_mm[start:end]), dtype=torch.long).to(device)
                    rv = torch.tensor(np.array(R_mm[start:end]), dtype=torch.float32).to(device)
                    xv          = apply_fog_of_war(xv, input_dim, fog_mask_prob)
                    logits_val  = model(xv)
                    val_loss   += reward_weighted_loss(logits_val, yv, rv, r_min, r_max).item()
                    topk        = logits_val.topk(5, dim=1).indices
                    c1 = (topk[:, :1] == yv.unsqueeze(1)).any(dim=1)
                    c3 = (topk[:, :3] == yv.unsqueeze(1)).any(dim=1)
                    c5 = (topk[:, :5] == yv.unsqueeze(1)).any(dim=1)
                    top1 += c1.sum().item()
                    top3 += c3.sum().item()
                    top5 += c5.sum().item()
                    reward_num += (rv * c1.float()).sum().item()
                    reward_den += rv.abs().sum().item()
                    n_batches  += 1
                del X_mm, y_mm, R_mm
        val_loss  /= n_batches
        top1_acc   = top1 / val_size
        top3_acc   = top3 / val_size
        top5_acc   = top5 / val_size
        reward_acc = reward_num / reward_den if reward_den > 0 else 0.0
        model.train()
        return val_loss, top1_acc, top3_acc, top5_acc, reward_acc

    model.train()
    stop_training = False

    for shard_epoch in itertools.count(1):
        if stop_training:
            break
        log.info(f"--- Shard epoch {shard_epoch} ---")
        for sid in np.random.permutation(train_ids):
            if stop_training:
                break
            X_mm = np.load(PROCESSED_DIR / f"X_{sid:03d}.npy", mmap_mode="r")
            y_mm = np.load(PROCESSED_DIR / f"y_{sid:03d}.npy", mmap_mode="r")
            R_mm = np.load(PROCESSED_DIR / f"R_{sid:03d}.npy", mmap_mode="r")
            n = len(y_mm)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = torch.tensor(np.array(X_mm[start:end]), dtype=torch.float32).to(device)
                y_batch = torch.tensor(np.array(y_mm[start:end]), dtype=torch.long).to(device)
                R_batch = torch.tensor(np.array(R_mm[start:end]), dtype=torch.float32).to(device)
                if global_step >= total_steps:
                    stop_training = True
                    break

                optimizer.zero_grad()
                X_batch = apply_fog_of_war(X_batch, input_dim, fog_mask_prob)
                logits  = model(X_batch)
                loss    = reward_weighted_loss(logits, y_batch, R_batch, r_min, r_max)
                loss.backward()
                optimizer.step()

                running_loss    += loss.item()
                running_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                running_steps   += 1
                global_step     += 1

                if global_step % LOG_TRAIN_EVERY == 0:
                    train_loss = running_loss / running_steps
                    train_acc  = running_correct / (running_steps * batch_size)
                    log.info(f"Step {global_step} | loss={train_loss:.4f} acc={train_acc:.3f}")
                    safe_wandb_log({
                        "train/loss": train_loss,
                        "train/acc":  train_acc,
                        "train/lr":   optimizer.param_groups[0]["lr"],
                    }, step=global_step)
                    running_loss = running_correct = running_steps = 0

                if global_step % 5000 == 0:
                    ckpt_path = CHECKPOINT_DIR / f"ckpt_step{global_step}.pt"
                    torch.save({
                        "step": global_step, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "input_dim": input_dim, "output_dim": output_dim,
                        "arch_config": arch_config,
                    }, ckpt_path)
                    log.info(f"  Checkpoint saved: {ckpt_path.name}")

                    val_loss, top1_acc, top3_acc, top5_acc, reward_acc = run_validation()
                    scheduler.step(val_loss)
                    lr_now = optimizer.param_groups[0]["lr"]
                    log.info(f"Step {global_step} [VAL] | loss={val_loss:.4f} top1={top1_acc:.3f} top3={top3_acc:.3f} top5={top5_acc:.3f} lr={lr_now:.2e}")
                    safe_wandb_log({
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
                            "step": global_step, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "val_loss": val_loss, "top1_acc": top1_acc,
                            "input_dim": input_dim, "output_dim": output_dim,
                            "arch_config": arch_config,
                        }, path)
                        log.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
                        try:
                            wandb.save(str(path))
                        except Exception as e:
                            log.warning(f"wandb.save failed: {e}")

    wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    train()
