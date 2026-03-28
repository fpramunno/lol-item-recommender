"""
Extracts samples (X, y, R) from the downloaded JSON timelines.
Output: data/dataset/samples.jsonl  — one sample per line.
"""

import bisect
import json
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TIMELINES_DIR = Path("data/timelines")
RESULTS_FILE = Path("data/match_results.json")
PARTICIPANTS_FILE = Path("data/match_participants.json")
OUTPUT_FILE = Path("data/dataset/samples.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Items to exclude: wards, potions, trinkets, cheap components
EXCLUDED_ITEMS = {
    2003, 2031, 2033, 2055,          # potions
    3340, 3363, 3364,                 # trinkets
    2301, 2302, 2303,                 # support quest items
}
MIN_ITEM_GOLD = 900      # excludes sub-900g components
DELTA_GOLD_MINUTES = 4   # window to calculate ΔGold after purchase
MIN_GAME_DURATION = 15   # minutes — exclude fast surrenders
ALPHA = 0.3              # weight of ΔGold in the reward


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_frame_at(frame_timestamps: list[int], frames: list, timestamp_ms: int) -> dict | None:
    """Returns the last frame at or before the given timestamp (binary search)."""
    idx = bisect.bisect_right(frame_timestamps, timestamp_ms) - 1
    return frames[idx] if idx >= 0 else None


def build_all_inventories(events: list) -> dict[int, tuple[list[int], list[list[int]]]]:
    """
    Single pass → {pid: (sorted_timestamps, inventories_at_each_ts)}
    Use bisect on sorted_timestamps for O(log n) lookup.
    """
    inventories: dict[int, list[int]] = {}
    ts_list:     dict[int, list[int]] = {}
    inv_list:    dict[int, list[list[int]]] = {}

    for ev in events:
        pid = ev.get("participantId")
        if pid is None:
            continue
        etype   = ev["type"]
        item_id = ev.get("itemId")
        ts      = ev["timestamp"]

        if pid not in inventories:
            inventories[pid] = []
            ts_list[pid]     = []
            inv_list[pid]    = []

        if etype == "ITEM_PURCHASED" and item_id is not None:
            inventories[pid].append(item_id)
            ts_list[pid].append(ts)
            inv_list[pid].append(list(inventories[pid]))
        elif etype in ("ITEM_SOLD", "ITEM_UNDO") and item_id is not None:
            if item_id in inventories[pid]:
                inventories[pid].remove(item_id)

    return {pid: (ts_list[pid], inv_list[pid]) for pid in inventories}


def build_kda_timeline(events: list) -> list[tuple[int, dict]]:
    """
    Single pass: returns [(timestamp, kda_snapshot), ...] sorted by timestamp.
    """
    kda: dict[int, dict] = {}
    timeline = []

    for ev in events:
        if ev["type"] != "CHAMPION_KILL":
            continue
        killer  = ev.get("killerId", 0)
        victim  = ev.get("victimId", 0)
        assists = ev.get("assistingParticipantIds", [])
        for pid in [killer, victim] + assists:
            if pid not in kda:
                kda[pid] = {"kills": 0, "deaths": 0, "assists": 0}
        kda[killer]["kills"]  += 1
        kda[victim]["deaths"] += 1
        for a in assists:
            kda[a]["assists"] += 1
        timeline.append((ev["timestamp"], {k: dict(v) for k, v in kda.items()}))

    return timeline


def get_kda_at(kda_timeline: list, up_to_ms: int) -> dict[int, dict]:
    """Binary search to find KDA snapshot at a given timestamp."""
    timestamps = [t for t, _ in kda_timeline]
    idx = bisect.bisect_right(timestamps, up_to_ms) - 1
    if idx < 0:
        return {}
    return kda_timeline[idx][1]


def gold_spent_from_inventory(inventory: list[int], item_costs: dict) -> int:
    return sum(item_costs.get(i, 0) for i in inventory)


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_timeline(
    match_id: str,
    timeline: dict,
    match_results: dict,
    match_participants: dict,
    item_costs: dict,
) -> list[dict]:
    """Returns a list of samples (X, y, R) extracted from a match."""
    samples = []
    info = timeline.get("info", {})
    frames = info.get("frames", [])
    if not frames:
        return []

    last_ts_min = frames[-1]["timestamp"] / 60000
    if last_ts_min < MIN_GAME_DURATION:
        return []

    result = match_results.get(match_id)
    winner_team = result["winner"] if result else None

    events = [ev for frame in frames for ev in frame.get("events", [])]

    # Single-pass precomputation
    inv_snapshots = build_all_inventories(events)
    kda_timeline  = build_kda_timeline(events)

    participant_info = info.get("participants", [])
    participant_team = {
        p["participantId"]: (100 if p["participantId"] <= 5 else 200)
        for p in participant_info
    }

    pdata = match_participants.get(match_id, {})
    participant_champion = {int(k): v["champion_id"] for k, v in pdata.items()}
    participant_role     = {int(k): v["team_position"] for k, v in pdata.items()}
    participant_runes    = {
        int(k): {
            "keystone":       v.get("keystone", 0),
            "primary_tree":   v.get("primary_tree", 0),
            "secondary_tree": v.get("secondary_tree", 0),
            "stat_offense":   v.get("stat_offense", 0),
            "stat_flex":      v.get("stat_flex", 0),
            "stat_defense":   v.get("stat_defense", 0),
        }
        for k, v in pdata.items()
    }

    # Frames are already sorted by timestamp — precompute once for bisect
    frame_timestamps = [f["timestamp"] for f in frames]

    for ev in events:
        if ev["type"] != "ITEM_PURCHASED":
            continue

        participant_id = ev.get("participantId")
        item_id        = ev.get("itemId")
        timestamp_ms   = ev["timestamp"]

        if item_id in EXCLUDED_ITEMS:
            continue
        if item_costs.get(item_id, 0) < MIN_ITEM_GOLD:
            continue

        current_frame = get_frame_at(frame_timestamps, frames, timestamp_ms)
        if not current_frame:
            continue

        participant_frames = current_frame.get("participantFrames", {})
        game_time_min = timestamp_ms / 60000

        gold_team100 = sum(
            pf["totalGold"] for pid, pf in participant_frames.items()
            if participant_team.get(int(pid), 100) == 100
        )
        gold_team200 = sum(
            pf["totalGold"] for pid, pf in participant_frames.items()
            if participant_team.get(int(pid), 100) == 200
        )

        buying_team    = participant_team.get(participant_id, 100)
        team_gold_diff = (gold_team100 - gold_team200) if buying_team == 100 else (gold_team200 - gold_team100)

        kda = get_kda_at(kda_timeline, timestamp_ms)

        players = []
        for pid_str, pf in participant_frames.items():
            pid = int(pid_str)
            snap = inv_snapshots.get(pid)
            if snap:
                ts_arr, inv_arr = snap
                idx = bisect.bisect_right(ts_arr, timestamp_ms) - 1
                inventory_pid = inv_arr[idx] if idx >= 0 else []
            else:
                inventory_pid = []
            pk    = kda.get(pid, {"kills": 0, "deaths": 0, "assists": 0})
            runes = participant_runes.get(pid, {})
            players.append({
                "participant_id": pid,
                "team":           participant_team.get(pid, 100),
                "champion_id":    participant_champion.get(pid, 0),
                "role":           participant_role.get(pid, ""),
                "keystone":       runes.get("keystone", 0),
                "primary_tree":   runes.get("primary_tree", 0),
                "secondary_tree": runes.get("secondary_tree", 0),
                "stat_offense":   runes.get("stat_offense", 0),
                "stat_flex":      runes.get("stat_flex", 0),
                "stat_defense":   runes.get("stat_defense", 0),
                "level":          pf.get("level", 1),
                "kills":          pk["kills"],
                "deaths":         pk["deaths"],
                "assists":        pk["assists"],
                "cs":             pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0),
                "gold_spent":     gold_spent_from_inventory(inventory_pid, item_costs),
                "gold_current":   pf.get("currentGold", 0) if pid == participant_id else None,
                "items":          inventory_pid,
                "is_buyer":       pid == participant_id,
            })

        X = {
            "game_time_min":  round(game_time_min, 2),
            "team_gold_diff": team_gold_diff,
            "players":        players,
        }

        if winner_team is not None:
            r_win = 1.0 if buying_team == winner_team else -1.0
        else:
            r_win = 0.0

        future_ts    = timestamp_ms + DELTA_GOLD_MINUTES * 60 * 1000
        future_frame = get_frame_at(frame_timestamps, frames, future_ts)

        if future_frame and str(participant_id) in future_frame.get("participantFrames", {}):
            gold_now         = participant_frames.get(str(participant_id), {}).get("totalGold", 0)
            gold_future      = future_frame["participantFrames"][str(participant_id)].get("totalGold", 0)
            delta_gold_raw   = gold_future - gold_now
            avg_gold_rate    = gold_team100 / max(game_time_min, 1) / 5
            delta_gold_norm  = delta_gold_raw / max(avg_gold_rate * DELTA_GOLD_MINUTES, 1)
        else:
            delta_gold_norm = 0.0

        R = r_win + ALPHA * delta_gold_norm

        samples.append({
            "match_id":          match_id,
            "participant_id":    participant_id,
            "timestamp_ms":      timestamp_ms,
            "X":                 X,
            "y":                 item_id,
            "R":                 round(R, 4),
            "r_win":             r_win,
            "r_delta_gold_norm": round(delta_gold_norm, 4),
        })

    return samples


# ── Data Dragon: item costs ────────────────────────────────────────────────────

def load_item_costs() -> dict[int, int]:
    cache = Path("data/item_costs.json")
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    log.info("Downloading item costs from Data Dragon...")
    import urllib.request
    with urllib.request.urlopen("https://ddragon.leagueoflegends.com/api/versions.json") as r:
        latest = json.loads(r.read())[0]
    with urllib.request.urlopen(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json") as r:
        items_data = json.loads(r.read())
    costs = {
        int(item_id): item["gold"]["total"]
        for item_id, item in items_data["data"].items()
        if "gold" in item
    }
    cache.write_text(json.dumps(costs))
    log.info(f"Loaded {len(costs)} item costs.")
    return costs


# ── Worker (top-level for pickling) ───────────────────────────────────────────

_match_results:     dict = {}
_match_participants: dict = {}
_item_costs:        dict = {}


_worker_out = None  # per-worker output file handle

def _process_file(tf_path: Path) -> int:
    """Writes samples directly to a per-worker temp file. Returns sample count."""
    try:
        timeline = json.loads(tf_path.read_text())
        samples  = parse_timeline(tf_path.stem, timeline, _match_results, _match_participants, _item_costs)
        for s in samples:
            _worker_out.write(json.dumps(s) + "\n")
        return len(samples)
    except Exception as e:
        log.error(f"Error on {tf_path.stem}: {e}")
        return 0

def _init_worker(match_results, match_participants, item_costs, tmp_dir):
    global _match_results, _match_participants, _item_costs, _worker_out
    import os
    _match_results      = match_results
    _match_participants = match_participants
    _item_costs         = item_costs
    worker_file = Path(tmp_dir) / f"worker_{os.getpid()}.jsonl"
    _worker_out = worker_file.open("w")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 = sequential)")
    args = parser.parse_args()

    match_results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    if not match_results:
        log.warning("match_results.json not found — r_win will be 0 for all samples")

    match_participants = {}
    if not PARTICIPANTS_FILE.exists():
        log.warning("match_participants.json not found. Run: python downloader.py --download")
    else:
        match_participants = json.loads(PARTICIPANTS_FILE.read_text())
        log.info(f"Match participants loaded: {len(match_participants)}")

    item_costs = load_item_costs()

    timeline_files = list(TIMELINES_DIR.glob("*.json"))
    log.info(f"Timelines to process: {len(timeline_files)} | Workers: {args.workers}")

    total_samples = 0
    with OUTPUT_FILE.open("w") as out:
        if args.workers == 1:
            for tf in tqdm(timeline_files, desc="parsing", unit="match"):
                try:
                    timeline = json.loads(tf.read_text())
                    samples  = parse_timeline(tf.stem, timeline, match_results, match_participants, item_costs)
                    for s in samples:
                        out.write(json.dumps(s) + "\n")
                    total_samples += len(samples)
                except Exception as e:
                    log.error(f"Error on {tf.stem}: {e}")
        else:
            import tempfile, shutil
            tmp_dir = Path(tempfile.mkdtemp())
            try:
                with mp.Pool(
                    processes=args.workers,
                    initializer=_init_worker,
                    initargs=(match_results, match_participants, item_costs, tmp_dir),
                ) as pool:
                    for count in tqdm(
                        pool.imap_unordered(_process_file, timeline_files, chunksize=32),
                        total=len(timeline_files), desc="parsing", unit="match",
                    ):
                        total_samples += count

                # Merge all worker files into OUTPUT_FILE
                log.info("Merging worker outputs...")
                for tmp_file in tmp_dir.glob("worker_*.jsonl"):
                    with tmp_file.open() as f:
                        shutil.copyfileobj(f, out)
            finally:
                shutil.rmtree(tmp_dir)

    log.info(f"Done. Total samples: {total_samples} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
