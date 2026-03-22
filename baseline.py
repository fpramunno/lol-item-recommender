"""
Computes the baselines to beat:
  - Random: picks a random item from the 207 available
  - Most frequent: always picks the most purchased item
"""

import numpy as np
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def main():
    y = np.load(PROCESSED_DIR / "y.npy")
    n = len(y)
    n_classes = int(y.max()) + 1

    log.info(f"Total samples: {n}")
    log.info(f"Unique items: {n_classes}")

    # ── Baseline 1: Random ────────────────────────────────────────────────────
    random_top1 = 1 / n_classes
    random_top3 = 3 / n_classes
    random_top5 = 5 / n_classes
    log.info(f"\nBaseline RANDOM:")
    log.info(f"  Top-1: {random_top1:.3f} ({random_top1*100:.1f}%)")
    log.info(f"  Top-3: {random_top3:.3f} ({random_top3*100:.1f}%)")
    log.info(f"  Top-5: {random_top5:.3f} ({random_top5*100:.1f}%)")

    # ── Baseline 2: Most Frequent ─────────────────────────────────────────────
    counts = Counter(y.tolist())
    most_common = [item for item, _ in counts.most_common(5)]

    top1_acc = counts[most_common[0]] / n
    top3_acc = sum(counts[i] for i in most_common[:3]) / n
    top5_acc = sum(counts[i] for i in most_common[:5]) / n

    log.info(f"\nBaseline MOST FREQUENT:")
    log.info(f"  Most purchased item (idx {most_common[0]}): {counts[most_common[0]]} times ({top1_acc*100:.1f}%)")
    log.info(f"  Top-1: {top1_acc:.3f} ({top1_acc*100:.1f}%)")
    log.info(f"  Top-3: {top3_acc:.3f} ({top3_acc*100:.1f}%)")
    log.info(f"  Top-5: {top5_acc:.3f} ({top5_acc*100:.1f}%)")

    log.info(f"\n{'─'*50}")
    log.info(f"Your model must exceed these values to be useful.")


if __name__ == "__main__":
    main()
