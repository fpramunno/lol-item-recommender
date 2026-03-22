"""
Extracts samples (X, y, R) from the downloaded JSON timelines.
Output: data/dataset/samples.jsonl  — one sample per line.
"""

import json
import logging
from pathlib import Path

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

def get_frame_at(frames: list, timestamp_ms: int) -> dict | None:
    """Returns the last frame before the given timestamp."""
    best = None
    for frame in frames:
        if frame["timestamp"] <= timestamp_ms:
            best = frame
        else:
            break
    return best


def build_inventory(events: list, participant_id: int, up_to_ms: int) -> list[int]:
    """Reconstructs a player's inventory up to a given timestamp."""
    inventory = []
    for ev in events:
        if ev["timestamp"] > up_to_ms:
            break
        if ev.get("participantId") != participant_id:
            continue
        etype = ev["type"]
        item_id = ev.get("itemId")
        if item_id is None:
            continue
        if etype == "ITEM_PURCHASED":
            inventory.append(item_id)
        elif etype in ("ITEM_SOLD", "ITEM_UNDO"):
            if item_id in inventory:
                inventory.remove(item_id)
    return inventory


def gold_spent_from_inventory(inventory: list[int], item_costs: dict) -> int:
    return sum(item_costs.get(i, 0) for i in inventory)


def compute_kda(events: list, up_to_ms: int) -> dict[int, dict]:
    """Computes kills/deaths/assists for each participant up to a timestamp."""
    kda = {}
    for ev in events:
        if ev["timestamp"] > up_to_ms:
            break
        if ev["type"] != "CHAMPION_KILL":
            continue
        killer = ev.get("killerId", 0)
        victim = ev.get("victimId", 0)
        assists = ev.get("assistingParticipantIds", [])

        if killer not in kda:
            kda[killer] = {"kills": 0, "deaths": 0, "assists": 0}
        if victim not in kda:
            kda[victim] = {"kills": 0, "deaths": 0, "assists": 0}

        kda[killer]["kills"] += 1
        kda[victim]["deaths"] += 1
        for a in assists:
            if a not in kda:
                kda[a] = {"kills": 0, "deaths": 0, "assists": 0}
            kda[a]["assists"] += 1
    return kda


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_timeline(match_id: str, timeline: dict, match_results: dict, match_participants: dict) -> list[dict]:
    """
    Returns a list of samples (X, y, R) extracted from a match.
    """
    samples = []
    info = timeline.get("info", {})
    frames = info.get("frames", [])
    if not frames:
        return []

    # Match duration in minutes
    last_ts_min = frames[-1]["timestamp"] / 60000
    if last_ts_min < MIN_GAME_DURATION:
        return []

    # Win/loss: team 100 or 200
    result = match_results.get(match_id)
    winner_team = result["winner"] if result else None

    # All events in order
    events = [ev for frame in frames for ev in frame.get("events", [])]

    # Map participantId → teamId (100 or 200)
    # The timeline API does not include teamId → use Riot convention: 1-5 = team100, 6-10 = team200
    participant_info = info.get("participants", [])
    participant_team = {
        p["participantId"]: (100 if p["participantId"] <= 5 else 200)
        for p in participant_info
    }

    # Map participantId → {champion_id, team_position} from match details
    pdata = match_participants.get(match_id, {})
    participant_champion = {int(k): v["champion_id"] for k, v in pdata.items()}
    participant_role = {int(k): v["team_position"] for k, v in pdata.items()}

    # Frames by timestamp (to look up gold after a purchase)
    frame_list = sorted(frames, key=lambda f: f["timestamp"])

    # Placeholder item costs — in production load from Data Dragon
    # { item_id: gold_cost }
    item_costs: dict[int, int] = load_item_costs()

    for ev in events:
        if ev["type"] != "ITEM_PURCHASED":
            continue

        participant_id = ev.get("participantId")
        item_id = ev.get("itemId")
        timestamp_ms = ev["timestamp"]

        # Basic filters
        if item_id in EXCLUDED_ITEMS:
            continue
        item_cost = item_costs.get(item_id, 0)
        if item_cost < MIN_ITEM_GOLD:
            continue

        # ── Build X ───────────────────────────────────────────────────────────

        current_frame = get_frame_at(frame_list, timestamp_ms)
        if not current_frame:
            continue

        participant_frames = current_frame.get("participantFrames", {})
        game_time_min = timestamp_ms / 60000

        # Global features (computed from participant frames)
        gold_team100 = sum(
            pf["totalGold"]
            for pid, pf in participant_frames.items()
            if participant_team.get(int(pid), 100) == 100
        )
        gold_team200 = sum(
            pf["totalGold"]
            for pid, pf in participant_frames.items()
            if participant_team.get(int(pid), 100) == 200
        )

        buying_team = participant_team.get(participant_id, 100)
        team_gold_diff = (gold_team100 - gold_team200) if buying_team == 100 else (gold_team200 - gold_team100)

        # KDA computed from events up to this point
        kda = compute_kda(events, timestamp_ms)

        # Features for each player (10 players)
        players = []
        for pid_str, pf in participant_frames.items():
            pid = int(pid_str)
            inventory_pid = build_inventory(events, pid, timestamp_ms)
            pk = kda.get(pid, {"kills": 0, "deaths": 0, "assists": 0})
            players.append({
                "participant_id": pid,
                "team": participant_team.get(pid, 100),
                "champion_id": participant_champion.get(pid, 0),
                "role": participant_role.get(pid, ""),
                "level": pf.get("level", 1),
                "kills": pk["kills"],
                "deaths": pk["deaths"],
                "assists": pk["assists"],
                "cs": pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0),
                "gold_spent": gold_spent_from_inventory(inventory_pid, item_costs),
                "gold_current": pf.get("currentGold", 0) if pid == participant_id else None,
                "items": inventory_pid,
                "is_buyer": pid == participant_id,
            })

        X = {
            "game_time_min": round(game_time_min, 2),
            "team_gold_diff": team_gold_diff,
            "players": players,
        }

        # ── Compute R ─────────────────────────────────────────────────────────

        # Win/loss
        if winner_team is not None:
            r_win = 1.0 if buying_team == winner_team else -1.0
        else:
            r_win = 0.0

        # ΔGold over the next DELTA_GOLD_MINUTES minutes
        future_ts = timestamp_ms + DELTA_GOLD_MINUTES * 60 * 1000
        future_frame = get_frame_at(frame_list, future_ts)

        if future_frame and str(participant_id) in future_frame.get("participantFrames", {}):
            gold_now = participant_frames.get(str(participant_id), {}).get("totalGold", 0)
            gold_future = future_frame["participantFrames"][str(participant_id)].get("totalGold", 0)
            delta_gold_raw = gold_future - gold_now

            # Normalize by the average gold rate of the match
            avg_gold_rate = gold_team100 / max(game_time_min, 1) / 5  # per player
            delta_gold_norm = delta_gold_raw / max(avg_gold_rate * DELTA_GOLD_MINUTES, 1)
        else:
            delta_gold_norm = 0.0

        R = r_win + ALPHA * delta_gold_norm

        samples.append({
            "match_id": match_id,
            "participant_id": participant_id,
            "timestamp_ms": timestamp_ms,
            "X": X,
            "y": item_id,
            "R": round(R, 4),
            "r_win": r_win,
            "r_delta_gold_norm": round(delta_gold_norm, 4),
        })

    return samples


# ── Data Dragon: item costs ────────────────────────────────────────────────────

def load_item_costs() -> dict[int, int]:
    """
    Loads item costs from Data Dragon (local cache).
    If not present, downloads automatically.
    """
    cache = Path("data/item_costs.json")
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    log.info("Downloading item costs from Data Dragon...")
    import urllib.request

    # Latest version
    versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    with urllib.request.urlopen(versions_url) as r:
        latest = json.loads(r.read())[0]

    items_url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json"
    with urllib.request.urlopen(items_url) as r:
        items_data = json.loads(r.read())

    costs = {
        int(item_id): item["gold"]["total"]
        for item_id, item in items_data["data"].items()
        if "gold" in item
    }
    cache.write_text(json.dumps(costs))
    log.info(f"Loaded {len(costs)} item costs.")
    return costs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_FILE.exists():
        log.warning("match_results.json not found — r_win will be 0 for all samples")
        match_results = {}
    else:
        match_results = json.loads(RESULTS_FILE.read_text())

    if not PARTICIPANTS_FILE.exists():
        log.warning("match_participants.json not found — champion_id and role will be 0/''. Run: python downloader.py --participants")
        match_participants = {}
    else:
        match_participants = json.loads(PARTICIPANTS_FILE.read_text())
        log.info(f"Match participants loaded: {len(match_participants)}")

    timeline_files = list(TIMELINES_DIR.glob("*.json"))
    log.info(f"Timelines to process: {len(timeline_files)}")

    total_samples = 0
    with OUTPUT_FILE.open("w") as out:
        for i, tf in enumerate(timeline_files):
            match_id = tf.stem
            try:
                timeline = json.loads(tf.read_text())
                samples = parse_timeline(match_id, timeline, match_results, match_participants)
                for s in samples:
                    out.write(json.dumps(s) + "\n")
                total_samples += len(samples)
            except Exception as e:
                log.error(f"Error on {match_id}: {e}")

            if (i + 1) % 500 == 0:
                log.info(f"Processed {i + 1}/{len(timeline_files)} — {total_samples} samples so far")

    log.info(f"Done. Total samples: {total_samples} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
