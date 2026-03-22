"""
Step 1: Preprocessing
Converts samples.jsonl into numpy tensors ready for training.

Structure of the X vector for each sample:
  [2 global features] + [10 players × (8 + N_items + N_champs + 6 + 5 + 13)] total features

Per-player features:
  - level, kills, deaths, assists, cs, gold_spent, gold_current, is_buyer → 8
  - binary vector of owned items → N_items (~207)
  - one-hot champion → N_champs (~170)
  - champion tags (Fighter/Tank/Mage/Assassin/Marksman/Support) → 6
  - role one-hot (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY) → 5
  - normalized champion base stats → 13
"""

import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLES_FILE = Path("data/dataset/samples.jsonl")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Normalization constants (computed by compute_stats.py) ────────────────────

STATS_FILE = Path("data/processed/feature_stats.json")

def load_norm_stats() -> dict:
    if not STATS_FILE.exists():
        raise FileNotFoundError("Run compute_stats.py first")
    return json.loads(STATS_FILE.read_text())

NORM = load_norm_stats()


# ── Champion data from Data Dragon ────────────────────────────────────────────

CHAMPION_TAGS  = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
ROLES          = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
CHAMP_STAT_KEYS = [
    "hp", "hpperlevel", "mp", "mpperlevel", "movespeed",
    "armor", "spellblock", "attackrange",
    "hpregen", "mpregen", "crit", "attackdamage", "attackspeedmod",
]

def load_champion_data() -> tuple[dict, dict, dict]:
    """
    Returns:
      champion_to_idx : {champion_id(int) → one-hot index}
      champ_tags      : {champion_id(int) → np.array(6,) binary}
      champ_stats_norm: {champion_id(int) → np.array(13,) normalized [-1,1]}
    """
    cache = Path("data/champion_features.json")
    if cache.exists():
        raw = json.loads(cache.read_text())
        champion_to_idx  = {int(k): v for k, v in raw["champion_to_idx"].items()}
        champ_tags       = {int(k): np.array(v, dtype=np.float32) for k, v in raw["champ_tags"].items()}
        champ_stats_norm = {int(k): np.array(v, dtype=np.float32) for k, v in raw["champ_stats_norm"].items()}
        return champion_to_idx, champ_tags, champ_stats_norm

    log.info("Downloading champion data from Data Dragon...")
    import urllib.request
    with urllib.request.urlopen("https://ddragon.leagueoflegends.com/api/versions.json") as r:
        latest = json.loads(r.read())[0]
    with urllib.request.urlopen(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json") as r:
        raw_champs = json.loads(r.read())["data"]

    # Collect all champion IDs and their raw stats
    champ_list = []
    for cdata in raw_champs.values():
        cid = int(cdata["key"])
        stats_raw = [cdata["stats"].get(k, 0.0) for k in CHAMP_STAT_KEYS]
        tags = [1.0 if t in cdata.get("tags", []) else 0.0 for t in CHAMPION_TAGS]
        champ_list.append((cid, stats_raw, tags))

    # Sort by ID for a deterministic index
    champ_list.sort(key=lambda x: x[0])
    champion_to_idx = {cid: idx for idx, (cid, _, _) in enumerate(champ_list)}

    # Normalize base stats using min/max across all champions
    stats_matrix = np.array([s for _, s, _ in champ_list], dtype=np.float32)
    stat_mins = stats_matrix.min(axis=0)
    stat_maxs = stats_matrix.max(axis=0)

    champ_tags = {}
    champ_stats_norm = {}
    for cid, stats_raw, tags in champ_list:
        champ_tags[cid] = np.array(tags, dtype=np.float32)
        s = np.array(stats_raw, dtype=np.float32)
        std = stat_maxs - stat_mins
        std[std == 0] = 1.0
        s_norm = (s - stat_mins) / std         # [0, 1]
        s_norm = (s_norm - 0.5) / 0.5          # [-1, 1]
        champ_stats_norm[cid] = s_norm

    # Save cache
    cache.write_text(json.dumps({
        "champion_to_idx":  {str(k): v for k, v in champion_to_idx.items()},
        "champ_tags":       {str(k): v.tolist() for k, v in champ_tags.items()},
        "champ_stats_norm": {str(k): v.tolist() for k, v in champ_stats_norm.items()},
    }))
    log.info(f"Champion data saved: {len(champion_to_idx)} champions")
    return champion_to_idx, champ_tags, champ_stats_norm


# ── Step 1: build the item_id → index mapping ─────────────────────────────────

def build_item_index(samples_file: Path) -> dict[int, int]:
    log.info("Building item index...")
    items = set()
    with samples_file.open() as f:
        for line in f:
            s = json.loads(line)
            items.add(s["y"])
            for p in s["X"]["players"]:
                items.update(p["items"])

    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(items))}
    log.info(f"Unique items: {len(item_to_idx)}")
    return item_to_idx


# ── Step 2: convert a sample into a numeric vector ────────────────────────────

def norm(value: float, min_val: float, max_val: float) -> float:
    """Normalizes a value to the range [-1, 1] via a two-step z-score:
       1. (x - mean) / std  where mean=min, std=max-min  → [0, 1]
       2. (x - 0.5) / 0.5                               → [-1, 1]
    """
    mean = min_val
    std = max_val - min_val
    if std == 0.0:
        return 0.0
    normalized = (value - mean) / std
    return (normalized - 0.5) / 0.5


def encode_sample(
    sample: dict,
    item_to_idx: dict,
    champion_to_idx: dict,
    champ_tags: dict,
    champ_stats_norm: dict,
) -> tuple[np.ndarray, int, float]:
    """
    Returns:
      x: float32 vector
      y: index of the purchased item
      r: reward float
    """
    n_items  = len(item_to_idx)
    n_champs = len(champion_to_idx)
    X = sample["X"]
    players = sorted(sample["X"]["players"], key=lambda p: p["participant_id"])

    # ── Global features (2) ───────────────────────────────────────────────────
    global_feats = np.array([
        norm(X["game_time_min"],    NORM["game_time_min"]["min"],    NORM["game_time_min"]["max"]),
        norm(X["team_gold_diff"],   NORM["team_gold_diff"]["min"],   NORM["team_gold_diff"]["max"]),
    ], dtype=np.float32)

    # ── Per-player features ───────────────────────────────────────────────────
    player_feats = []
    for p in players:
        # 8 numeric features
        numeric = np.array([
            norm(p["level"],                NORM["level"]["min"],       NORM["level"]["max"]),
            norm(p["kills"],                NORM["kills"]["min"],       NORM["kills"]["max"]),
            norm(p["deaths"],               NORM["deaths"]["min"],      NORM["deaths"]["max"]),
            norm(p["assists"],              NORM["assists"]["min"],     NORM["assists"]["max"]),
            norm(p["cs"],                   NORM["cs"]["min"],          NORM["cs"]["max"]),
            norm(p["gold_spent"],           NORM["gold_spent"]["min"],  NORM["gold_spent"]["max"]),
            norm(p["gold_current"] or 0,    NORM["gold_current"]["min"],NORM["gold_current"]["max"]),
            float(p["is_buyer"]),
        ], dtype=np.float32)

        # Binary item vector
        item_vec = np.zeros(n_items, dtype=np.float32)
        for item_id in p["items"]:
            if item_id in item_to_idx:
                item_vec[item_to_idx[item_id]] = 1.0

        # One-hot champion
        champ_vec = np.zeros(n_champs, dtype=np.float32)
        cid = p.get("champion_id", 0)
        if cid in champion_to_idx:
            champ_vec[champion_to_idx[cid]] = 1.0

        # Champion tags (6)
        tags_vec = champ_tags.get(cid, np.zeros(len(CHAMPION_TAGS), dtype=np.float32))

        # Role one-hot (5)
        role_vec = np.zeros(len(ROLES), dtype=np.float32)
        role = p.get("role", "")
        if role in ROLES:
            role_vec[ROLES.index(role)] = 1.0

        # Normalized champion base stats (13)
        stats_vec = champ_stats_norm.get(cid, np.zeros(len(CHAMP_STAT_KEYS), dtype=np.float32))

        player_feats.append(np.concatenate([numeric, item_vec, champ_vec, tags_vec, role_vec, stats_vec]))

    x = np.concatenate([global_feats] + player_feats)

    y = item_to_idx[sample["y"]]
    r = float(sample["R"])

    return x, y, r


# ── Step 3: process the entire dataset ────────────────────────────────────────

def preprocess(samples_file: Path, output_dir: Path):
    item_to_idx = build_item_index(samples_file)

    # Save the mapping for use at inference time
    (output_dir / "item_to_idx.json").write_text(json.dumps(item_to_idx))
    log.info("item_to_idx.json saved")

    # Load champion data
    champion_to_idx, champ_tags, champ_stats_norm = load_champion_data()
    (output_dir / "champion_to_idx.json").write_text(json.dumps({str(k): v for k, v in champion_to_idx.items()}))
    log.info(f"champion_to_idx.json saved: {len(champion_to_idx)} champions")

    # Dimensions
    n_items  = len(item_to_idx)
    n_champs = len(champion_to_idx)
    feat_dim = 2 + 10 * (8 + n_items + n_champs + len(CHAMPION_TAGS) + len(ROLES) + len(CHAMP_STAT_KEYS))
    log.info(f"X vector dimension: {feat_dim} (items={n_items}, champs={n_champs})")

    # Count rows for pre-allocation
    log.info("Counting samples...")
    n_samples = sum(1 for _ in samples_file.open())
    log.info(f"Total samples: {n_samples}")

    # Pre-allocate arrays
    X_all = np.zeros((n_samples, feat_dim), dtype=np.float32)
    y_all = np.zeros(n_samples, dtype=np.int64)
    R_all = np.zeros(n_samples, dtype=np.float32)

    log.info("Encoding samples...")
    errors = 0
    with samples_file.open() as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                log.info(f"  {i}/{n_samples}")
            try:
                sample = json.loads(line)
                x, y, r = encode_sample(sample, item_to_idx, champion_to_idx, champ_tags, champ_stats_norm)
                X_all[i] = x
                y_all[i] = y
                R_all[i] = r
            except Exception as e:
                errors += 1

    if errors:
        log.warning(f"Samples with errors (skipped): {errors}")

    # Save
    np.save(output_dir / "X.npy", X_all)
    np.save(output_dir / "y.npy", y_all)
    np.save(output_dir / "R.npy", R_all)

    log.info(f"Saved to {output_dir}/")
    log.info(f"  X shape: {X_all.shape}")
    log.info(f"  y shape: {y_all.shape}")
    log.info(f"  R shape: {R_all.shape}")
    log.info(f"  R range: [{R_all.min():.3f}, {R_all.max():.3f}]")


if __name__ == "__main__":
    preprocess(SAMPLES_FILE, OUTPUT_DIR)
