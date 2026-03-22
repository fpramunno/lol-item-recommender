"""
Computes the min and max of each numerical feature from the real dataset.
Saves the statistics in data/processed/feature_stats.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLES_FILE = Path("data/dataset/samples.jsonl")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

PLAYER_FEATURES = ["level", "kills", "deaths", "assists", "cs", "gold_spent", "gold_current"]
GLOBAL_FEATURES = ["game_time_min", "team_gold_diff"]

def main():
    stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})

    log.info("Analyzing dataset...")
    n = 0
    with SAMPLES_FILE.open() as f:
        for line in tqdm(f, desc="Processing samples"):
            s = json.loads(line)
            X = s["X"]
            n += 1

            # Global features
            for feat in GLOBAL_FEATURES:
                v = X[feat]
                stats[feat]["min"] = min(stats[feat]["min"], v)
                stats[feat]["max"] = max(stats[feat]["max"], v)

            # Per-player features
            for p in X["players"]:
                for feat in PLAYER_FEATURES:
                    v = p[feat] if p[feat] is not None else 0
                    stats[feat]["min"] = min(stats[feat]["min"], v)
                    stats[feat]["max"] = max(stats[feat]["max"], v)

    log.info(f"Samples analyzed: {n}")
    log.info("\nStatistics per feature:")
    for feat, s in sorted(stats.items()):
        log.info(f"  {feat:20s} min={s['min']:>10.1f}  max={s['max']:>10.1f}")

    out = {feat: {"min": float(s["min"]), "max": float(s["max"])}
           for feat, s in stats.items()}
    (OUTPUT_DIR / "feature_stats.json").write_text(json.dumps(out, indent=2))
    log.info(f"\nSaved to {OUTPUT_DIR}/feature_stats.json")

if __name__ == "__main__":
    main()
