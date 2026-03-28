"""
Splits data/processed/X.npy, y.npy, R.npy into smaller shards.
Run once after preprocess.py:
    python shard_dataset.py --shard-size 500000
"""

import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-size", type=int, default=500_000, help="Samples per shard (default: 500k ~8.5GB)")
    args = parser.parse_args()

    log.info("Loading X, y, R (memmap)...")
    X = np.load(PROCESSED_DIR / "X.npy", mmap_mode="r")
    y = np.load(PROCESSED_DIR / "y.npy", mmap_mode="r")
    R = np.load(PROCESSED_DIR / "R.npy", mmap_mode="r")
    n = len(y)
    log.info(f"Total samples: {n} | X shape: {X.shape}")

    shard_dir = PROCESSED_DIR / "shards"
    shard_dir.mkdir(exist_ok=True)

    n_shards = (n + args.shard_size - 1) // args.shard_size
    log.info(f"Creating {n_shards} shards of ~{args.shard_size} samples each → {shard_dir}/")

    for i in range(n_shards):
        start = i * args.shard_size
        end   = min(start + args.shard_size, n)
        np.save(shard_dir / f"X_{i:03d}.npy", np.array(X[start:end]))
        np.save(shard_dir / f"y_{i:03d}.npy", np.array(y[start:end]))
        np.save(shard_dir / f"R_{i:03d}.npy", np.array(R[start:end]))
        log.info(f"  Shard {i:03d}: samples {start}–{end} ({end - start} samples)")

    log.info("Done.")


if __name__ == "__main__":
    main()
