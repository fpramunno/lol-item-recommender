"""
Evaluates the best model on the validation shards.
Usage:
    python evaluate.py
    python evaluate.py --fog-mask-prob 0.0   # disable fog of war
    python evaluate.py --val-shards 2        # number of last shards to use as validation
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import logging

from model import build_model, reward_weighted_loss
from train import ShardDataset, apply_fog_of_war

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR  = Path("data/processed/shards")
CHECKPOINT_DIR = Path("checkpoints")

BASELINE_RANDOM        = {"top1": 0.005, "top3": 0.014, "top5": 0.024}
BASELINE_MOST_FREQUENT = {"top1": 0.047, "top3": 0.119, "top5": 0.178}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str,   default="best_model.pt")
    parser.add_argument("--val-shards",    type=int,   default=2, help="Number of last shards to use as validation")
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--fog-mask-prob", type=float, default=0.5)
    parser.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers",   type=int,   default=0)
    args = parser.parse_args()

    shard_files = sorted(PROCESSED_DIR.glob("X_*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {PROCESSED_DIR}")
    n_shards = len(shard_files)
    val_ids  = list(range(n_shards - args.val_shards, n_shards))
    log.info(f"Validation shards: {val_ids}")

    ckpt_path = CHECKPOINT_DIR / args.checkpoint
    ckpt        = torch.load(ckpt_path, map_location=args.device)
    input_dim   = ckpt["input_dim"]
    output_dim  = ckpt["output_dim"]
    arch_config = ckpt.get("arch_config", {"arch": "mlp", "hidden_dims": [1024, 512, 256], "dropout": 0.0})
    log.info(f"Loaded checkpoint: {ckpt_path.name} (step={ckpt.get('step', '?')}, arch={arch_config['arch']})")

    model = build_model(arch_config, input_dim, output_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    r_min = float(min(np.load(PROCESSED_DIR / f"R_{i:03d}.npy").min() for i in val_ids))
    r_max = float(max(np.load(PROCESSED_DIR / f"R_{i:03d}.npy").max() for i in val_ids))

    val_ds     = ConcatDataset([ShardDataset(PROCESSED_DIR, sid) for sid in val_ids])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_size   = len(val_ds)
    log.info(f"Val samples: {val_size} | fog_mask_prob: {args.fog_mask_prob} | device: {args.device}")

    val_loss = 0.0
    top1 = top3 = top5 = 0
    reward_num = reward_den = 0.0

    with torch.no_grad():
        for xv, yv, rv in val_loader:
            xv, yv, rv = xv.to(args.device), yv.to(args.device), rv.to(args.device)
            if args.fog_mask_prob > 0:
                xv = apply_fog_of_war(xv, input_dim, args.fog_mask_prob)
            logits = model(xv)
            val_loss += reward_weighted_loss(logits, yv, rv, r_min, r_max).item()
            topk = logits.topk(5, dim=1).indices
            c1 = (topk[:, :1] == yv.unsqueeze(1)).any(dim=1)
            c3 = (topk[:, :3] == yv.unsqueeze(1)).any(dim=1)
            c5 = (topk[:, :5] == yv.unsqueeze(1)).any(dim=1)
            top1 += c1.sum().item()
            top3 += c3.sum().item()
            top5 += c5.sum().item()
            reward_num += (rv * c1.float()).sum().item()
            reward_den += rv.abs().sum().item()

    val_loss  /= len(val_loader)
    top1_acc   = top1 / val_size
    top3_acc   = top3 / val_size
    top5_acc   = top5 / val_size
    reward_acc = reward_num / reward_den if reward_den > 0 else 0.0

    print("\n" + "="*55)
    print(f"  Checkpoint : {ckpt_path.name}  (step {ckpt.get('step', '?')})")
    print(f"  Val samples: {val_size:,}")
    print(f"  Fog of war : {args.fog_mask_prob}")
    print("="*55)
    print(f"  Loss       : {val_loss:.4f}")
    print(f"  Top-1 acc  : {top1_acc:.3f}  (random={BASELINE_RANDOM['top1']:.3f}, mostfreq={BASELINE_MOST_FREQUENT['top1']:.3f})")
    print(f"  Top-3 acc  : {top3_acc:.3f}  (random={BASELINE_RANDOM['top3']:.3f}, mostfreq={BASELINE_MOST_FREQUENT['top3']:.3f})")
    print(f"  Top-5 acc  : {top5_acc:.3f}  (random={BASELINE_RANDOM['top5']:.3f}, mostfreq={BASELINE_MOST_FREQUENT['top5']:.3f})")
    print(f"  Reward acc : {reward_acc:.3f}")
    print(f"  vs mostfreq: top1 {top1_acc - BASELINE_MOST_FREQUENT['top1']:+.3f} | top3 {top3_acc - BASELINE_MOST_FREQUENT['top3']:+.3f} | top5 {top5_acc - BASELINE_MOST_FREQUENT['top5']:+.3f}")
    print("="*55)


if __name__ == "__main__":
    main()
