"""
Step 1: Preprocessing
Converts samples.jsonl into numpy tensors ready for training.

Structure of the X vector for each sample:
  [2 global features] + [10 players × (8 + N_items + N_champs + 6 + 5 + 13 + 18 + 5 + 5 + 3 + 3 + 3)]

Per-player features:
  - level, kills, deaths, assists, cs, gold_spent, gold_current, is_buyer → 8
  - binary vector of owned items → N_items (~207)
  - one-hot champion → N_champs (~170)
  - champion tags (Fighter/Tank/Mage/Assassin/Marksman/Support) → 6
  - role one-hot (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY) → 5
  - normalized champion base stats → 13
  - keystone one-hot → 18
  - primary_tree one-hot → 5
  - secondary_tree one-hot → 5
  - stat_offense one-hot → 3
  - stat_flex one-hot → 3
  - stat_defense one-hot → 3
"""

import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLES_FILE = Path("data/dataset/samples.jsonl")
OUTPUT_DIR   = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Normalization constants ────────────────────────────────────────────────────

STATS_FILE = Path("data/processed/feature_stats.json")

def load_norm_stats() -> dict:
    if not STATS_FILE.exists():
        raise FileNotFoundError("Run compute_stats.py first")
    return json.loads(STATS_FILE.read_text())

NORM = load_norm_stats()


# ── Rune / role lookup dicts (O(1) instead of list.index) ─────────────────────

TREE_IDS         = [8000, 8100, 8200, 8300, 8400]
KEYSTONE_IDS     = [8005, 8008, 8010, 8021, 8112, 8124, 8128, 9923,
                    8214, 8229, 8230, 8351, 8360, 8369, 8437, 8439, 8465]
STAT_OFFENSE_IDS = [5005, 5007, 5008]
STAT_FLEX_IDS    = [5001, 5007, 5008]
STAT_DEFENSE_IDS = [5001, 5011, 5013]
CHAMPION_TAGS    = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
ROLES            = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
CHAMP_STAT_KEYS  = [
    "hp", "hpperlevel", "mp", "mpperlevel", "movespeed",
    "armor", "spellblock", "attackrange",
    "hpregen", "mpregen", "crit", "attackdamage", "attackspeedmod",
]

# Precomputed O(1) lookups
TREE_IDX         = {v: i for i, v in enumerate(TREE_IDS)}
KEYSTONE_IDX     = {v: i for i, v in enumerate(KEYSTONE_IDS)}
STAT_OFFENSE_IDX = {v: i for i, v in enumerate(STAT_OFFENSE_IDS)}
STAT_FLEX_IDX    = {v: i for i, v in enumerate(STAT_FLEX_IDS)}
STAT_DEFENSE_IDX = {v: i for i, v in enumerate(STAT_DEFENSE_IDS)}
ROLE_IDX         = {v: i for i, v in enumerate(ROLES)}

N_TREE     = len(TREE_IDS)
N_KEYSTONE = len(KEYSTONE_IDS)
N_STAT_OFF = len(STAT_OFFENSE_IDS)
N_STAT_FLX = len(STAT_FLEX_IDS)
N_STAT_DEF = len(STAT_DEFENSE_IDS)
N_TAGS     = len(CHAMPION_TAGS)
N_ROLES    = len(ROLES)
N_STATS    = len(CHAMP_STAT_KEYS)
RUNE_DIM   = N_KEYSTONE + N_TREE + N_TREE + N_STAT_OFF + N_STAT_FLX + N_STAT_DEF


# ── Champion data ──────────────────────────────────────────────────────────────

def load_champion_data() -> tuple[dict, dict, dict]:
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

    champ_list = []
    for cdata in raw_champs.values():
        cid = int(cdata["key"])
        stats_raw = [cdata["stats"].get(k, 0.0) for k in CHAMP_STAT_KEYS]
        tags = [1.0 if t in cdata.get("tags", []) else 0.0 for t in CHAMPION_TAGS]
        champ_list.append((cid, stats_raw, tags))

    champ_list.sort(key=lambda x: x[0])
    champion_to_idx = {cid: idx for idx, (cid, _, _) in enumerate(champ_list)}

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
        s_norm = (s - stat_mins) / std
        s_norm = (s_norm - 0.5) / 0.5
        champ_stats_norm[cid] = s_norm

    cache.write_text(json.dumps({
        "champion_to_idx":  {str(k): v for k, v in champion_to_idx.items()},
        "champ_tags":       {str(k): v.tolist() for k, v in champ_tags.items()},
        "champ_stats_norm": {str(k): v.tolist() for k, v in champ_stats_norm.items()},
    }))
    log.info(f"Champion data saved: {len(champion_to_idx)} champions")
    return champion_to_idx, champ_tags, champ_stats_norm


# ── Item index ─────────────────────────────────────────────────────────────────

def build_item_index(samples_file: Path) -> tuple[dict[int, int], int]:
    """Returns (item_to_idx, n_samples) in a single pass."""
    log.info("Building item index and counting samples...")
    items = set()
    n = 0
    with samples_file.open() as f:
        for line in tqdm(f, desc="scanning", unit="sample"):
            s = json.loads(line)
            items.add(s["y"])
            for p in s["X"]["players"]:
                items.update(p["items"])
            n += 1
    item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(items))}
    log.info(f"Unique items: {len(item_to_idx)} | Total samples: {n}")
    return item_to_idx, n


# ── Encoding ───────────────────────────────────────────────────────────────────

def norm(value: float, min_val: float, max_val: float) -> float:
    std = max_val - min_val
    if std == 0.0:
        return 0.0
    return ((value - min_val) / std - 0.5) / 0.5


def encode_sample(
    sample: dict,
    item_to_idx: dict,
    champion_to_idx: dict,
    champ_tags: dict,
    champ_stats_norm: dict,
    x_buf: np.ndarray,   # pre-allocated buffer — written in place
) -> tuple[int, float]:
    """Encodes sample directly into x_buf. Returns (y_idx, r)."""
    N  = NORM
    X  = sample["X"]
    players = sorted(X["players"], key=lambda p: p["participant_id"])

    n_items  = len(item_to_idx)
    n_champs = len(champion_to_idx)
    per_player = 8 + n_items + n_champs + N_TAGS + N_ROLES + N_STATS + RUNE_DIM

    x_buf[0] = norm(X["game_time_min"],  N["game_time_min"]["min"],  N["game_time_min"]["max"])
    x_buf[1] = norm(X["team_gold_diff"], N["team_gold_diff"]["min"], N["team_gold_diff"]["max"])

    offset = 2
    for p in players:
        # 8 numeric
        x_buf[offset]     = norm(p["level"],            N["level"]["min"],        N["level"]["max"])
        x_buf[offset + 1] = norm(p["kills"],            N["kills"]["min"],        N["kills"]["max"])
        x_buf[offset + 2] = norm(p["deaths"],           N["deaths"]["min"],       N["deaths"]["max"])
        x_buf[offset + 3] = norm(p["assists"],          N["assists"]["min"],      N["assists"]["max"])
        x_buf[offset + 4] = norm(p["cs"],               N["cs"]["min"],           N["cs"]["max"])
        x_buf[offset + 5] = norm(p["gold_spent"],       N["gold_spent"]["min"],   N["gold_spent"]["max"])
        x_buf[offset + 6] = norm(p["gold_current"] or 0, N["gold_current"]["min"], N["gold_current"]["max"])
        x_buf[offset + 7] = float(p["is_buyer"])
        o = offset + 8

        # Items
        for item_id in p["items"]:
            idx = item_to_idx.get(item_id)
            if idx is not None:
                x_buf[o + idx] = 1.0
        o += n_items

        # Champion one-hot
        cid = p.get("champion_id", 0)
        cidx = champion_to_idx.get(cid)
        if cidx is not None:
            x_buf[o + cidx] = 1.0
        o += n_champs

        # Tags
        tags = champ_tags.get(cid)
        if tags is not None:
            x_buf[o:o + N_TAGS] = tags
        o += N_TAGS

        # Role
        ridx = ROLE_IDX.get(p.get("role", ""))
        if ridx is not None:
            x_buf[o + ridx] = 1.0
        o += N_ROLES

        # Base stats
        stats = champ_stats_norm.get(cid)
        if stats is not None:
            x_buf[o:o + N_STATS] = stats
        o += N_STATS

        # Keystone
        kidx = KEYSTONE_IDX.get(p.get("keystone", 0))
        if kidx is not None:
            x_buf[o + kidx] = 1.0
        o += N_KEYSTONE

        # Primary tree
        pidx = TREE_IDX.get(p.get("primary_tree", 0))
        if pidx is not None:
            x_buf[o + pidx] = 1.0
        o += N_TREE

        # Secondary tree
        sidx = TREE_IDX.get(p.get("secondary_tree", 0))
        if sidx is not None:
            x_buf[o + sidx] = 1.0
        o += N_TREE

        # Stat shards
        oidx = STAT_OFFENSE_IDX.get(p.get("stat_offense", 0))
        if oidx is not None:
            x_buf[o + oidx] = 1.0
        o += N_STAT_OFF

        fidx = STAT_FLEX_IDX.get(p.get("stat_flex", 0))
        if fidx is not None:
            x_buf[o + fidx] = 1.0
        o += N_STAT_FLX

        didx = STAT_DEFENSE_IDX.get(p.get("stat_defense", 0))
        if didx is not None:
            x_buf[o + didx] = 1.0

        offset += per_player

    return item_to_idx[sample["y"]], float(sample["R"])


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess(samples_file: Path, output_dir: Path):
    item_to_idx, n_samples = build_item_index(samples_file)
    (output_dir / "item_to_idx.json").write_text(json.dumps(item_to_idx))
    log.info("item_to_idx.json saved")

    champion_to_idx, champ_tags, champ_stats_norm = load_champion_data()
    (output_dir / "champion_to_idx.json").write_text(
        json.dumps({str(k): v for k, v in champion_to_idx.items()})
    )
    log.info(f"champion_to_idx.json saved: {len(champion_to_idx)} champions")

    n_items  = len(item_to_idx)
    n_champs = len(champion_to_idx)
    feat_dim = 2 + 10 * (8 + n_items + n_champs + N_TAGS + N_ROLES + N_STATS + RUNE_DIM)
    log.info(f"X vector dimension: {feat_dim} (items={n_items}, champs={n_champs})")

    SHARD_SIZE = 500_000  # samples per shard (~8.5 GB each)

    log.info(f"Encoding and saving shards (shard_size={SHARD_SIZE})...")
    x_buf  = np.zeros(feat_dim, dtype=np.float32)
    errors = 0
    shard  = 0
    X_buf, y_buf, R_buf = [], [], []
    total  = 0

    def flush_shard(idx, Xb, yb, Rb):
        np.save(output_dir / f"X_{idx:03d}.npy", np.array(Xb, dtype=np.float32))
        np.save(output_dir / f"y_{idx:03d}.npy", np.array(yb, dtype=np.int64))
        np.save(output_dir / f"R_{idx:03d}.npy", np.array(Rb, dtype=np.float32))
        log.info(f"  Shard {idx:03d} saved — {len(Xb)} samples")

    with samples_file.open() as f:
        for line in tqdm(f, total=n_samples, desc="encoding", unit="sample"):
            try:
                sample = json.loads(line)
                x_buf[:] = 0.0
                y, r = encode_sample(sample, item_to_idx, champion_to_idx, champ_tags, champ_stats_norm, x_buf)
                X_buf.append(x_buf.copy())
                y_buf.append(y)
                R_buf.append(r)
                total += 1
                if len(X_buf) >= SHARD_SIZE:
                    flush_shard(shard, X_buf, y_buf, R_buf)
                    shard += 1
                    X_buf, y_buf, R_buf = [], [], []
            except Exception:
                errors += 1

    if X_buf:
        flush_shard(shard, X_buf, y_buf, R_buf)
        shard += 1

    if errors:
        log.warning(f"Samples with errors: {errors}")
    log.info(f"Done — {total} samples in {shard} shards → {output_dir}/")


if __name__ == "__main__":
    preprocess(SAMPLES_FILE, OUTPUT_DIR)
