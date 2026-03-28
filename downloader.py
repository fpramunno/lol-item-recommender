"""
Downloads match timelines from the Riot API for high-elo players.
Saves raw JSON files in data/timelines/.

App-level rate limit: 20 req/1s, 100 req/2min
Binding limit: 100/120s → 1 req every 1.3s (with margin)
"""

import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
REGION = os.getenv("REGION", "euw1")
CLUSTER = os.getenv("REGIONAL_CLUSTER", "europe")
QUEUE = os.getenv("QUEUE", "RANKED_SOLO_5x5")

TIMELINES_DIR = Path("data/timelines")
MATCH_IDS_FILE = Path("data/match_ids.txt")
PARTICIPANTS_FILE = Path("data/match_participants.json")
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUEST_DELAY = 0.9  # 100 req/120s = 0.83/s → 0.9s has small safety margin


def get(url: str, params: dict = None) -> dict:
    params = params or {}
    params["api_key"] = API_KEY
    for attempt in range(3):
        r = requests.get(url, params=params)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 10))
            log.warning(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
        elif r.status_code == 404:
            return None
        else:
            log.error(f"Error {r.status_code} on {url}")
            time.sleep(5)
    return None


# ── 1. Collect PUUIDs ─────────────────────────────────────────────────────────

def get_high_elo_puuids() -> list[str]:
    tiers = ["challengerleagues", "grandmasterleagues", "masterleagues"]
    puuids = []

    for tier in tiers:
        url = f"https://{REGION}.api.riotgames.com/lol/league/v4/{tier}/by-queue/{QUEUE}"
        data = get(url)
        if not data:
            continue
        entries = data.get("entries", [])
        log.info(f"{tier}: {len(entries)} players found")

        for entry in tqdm(entries, desc=tier):
            if "puuid" in entry:
                puuids.append(entry["puuid"])
            elif "summonerId" in entry:
                time.sleep(REQUEST_DELAY)
                summoner_url = f"https://{REGION}.api.riotgames.com/lol/summoner/v4/summoners/{entry['summonerId']}"
                summoner = get(summoner_url)
                if summoner and "puuid" in summoner:
                    puuids.append(summoner["puuid"])

    puuids = list(set(puuids))
    log.info(f"Total PUUIDs collected: {len(puuids)}")
    return puuids


# ── 2. Collect Match IDs ──────────────────────────────────────────────────────

def get_match_ids(puuids: list[str], matches_per_player: int = 20) -> list[str]:
    all_ids = set()

    if MATCH_IDS_FILE.exists():
        all_ids.update(MATCH_IDS_FILE.read_text().splitlines())
        log.info(f"Match IDs already present: {len(all_ids)}")

    for puuid in tqdm(puuids, desc="match IDs"):
        url = f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        data = get(url, params={"queue": 420, "type": "ranked", "count": matches_per_player})
        if data:
            all_ids.update(data)
        time.sleep(REQUEST_DELAY)

    MATCH_IDS_FILE.write_text("\n".join(all_ids))
    log.info(f"Match IDs saved: {len(all_ids)}")
    return list(all_ids)


# ── 3. Download everything per match (timeline + results + participants) ────────

def download_all_match_data(match_ids: list[str]):
    """
    Single loop per match: downloads timeline + results + participants.
    2 API calls per match (timeline endpoint + match endpoint), no redundant passes.
    Saves: data/timelines/{id}.json, data/match_results.json, data/match_participants.json
    """
    results_file  = Path("data/match_results.json")
    results       = json.loads(results_file.read_text()) if results_file.exists() else {}
    participants  = json.loads(PARTICIPANTS_FILE.read_text()) if PARTICIPANTS_FILE.exists() else {}
    already_tl    = {f.stem for f in TIMELINES_DIR.glob("*.json")}

    def has_runes(entry: dict) -> bool:
        values = list(entry.values())
        return len(values) > 0 and "keystone" in values[0]

    to_fetch = [
        mid for mid in match_ids
        if mid not in already_tl
        or mid not in results
        or mid not in participants
        or not has_runes(participants.get(mid, {}))
    ]
    log.info(f"Matches to process: {len(to_fetch)} / {len(match_ids)}")

    for i, match_id in enumerate(tqdm(to_fetch, desc="downloading")):
        # Timeline (if missing)
        if match_id not in already_tl:
            tl = get(f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline")
            if tl:
                (TIMELINES_DIR / f"{match_id}.json").write_text(json.dumps(tl))
                already_tl.add(match_id)
            time.sleep(REQUEST_DELAY)

        # Match data: results + participants (if missing or incomplete)
        needs_match = (
            match_id not in results
            or match_id not in participants
            or not has_runes(participants.get(match_id, {}))
        )
        if needs_match:
            data = get(f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}")
            if data:
                for team in data["info"]["teams"]:
                    results[match_id] = {"winner": team["teamId"] if team["win"] else (300 - team["teamId"])}
                    break
                pinfo = {}
                for p in data["info"]["participants"]:
                    perks      = p.get("perks", {})
                    styles     = perks.get("styles", [])
                    stat_perks = perks.get("statPerks", {})
                    primary    = next((s for s in styles if s["description"] == "primaryStyle"), {})
                    secondary  = next((s for s in styles if s["description"] == "subStyle"), {})
                    keystone   = primary.get("selections", [{}])[0].get("perk", 0)
                    pinfo[str(p["participantId"])] = {
                        "champion_id":    p["championId"],
                        "team_position":  p.get("teamPosition", ""),
                        "keystone":       keystone,
                        "primary_tree":   primary.get("style", 0),
                        "secondary_tree": secondary.get("style", 0),
                        "stat_offense":   stat_perks.get("offense", 0),
                        "stat_flex":      stat_perks.get("flex", 0),
                        "stat_defense":   stat_perks.get("defense", 0),
                    }
                participants[match_id] = pinfo
            time.sleep(REQUEST_DELAY)

        if (i + 1) % 500 == 0:
            results_file.write_text(json.dumps(results))
            PARTICIPANTS_FILE.write_text(json.dumps(participants))
            log.info(f"Checkpoint: {i + 1}/{len(to_fetch)}")

    results_file.write_text(json.dumps(results))
    PARTICIPANTS_FILE.write_text(json.dumps(participants))
    log.info(f"Done — timelines: {len(already_tl)}, results: {len(results)}, participants: {len(participants)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--puuids", action="store_true")
    parser.add_argument("--matches", action="store_true")
    parser.add_argument("--timelines",    action="store_true", help="alias for --download")
    parser.add_argument("--results",      action="store_true", help="alias for --download")
    parser.add_argument("--participants", action="store_true", help="alias for --download")
    parser.add_argument("--download",     action="store_true", help="download timeline + results + participants in one pass")
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--matches-per-player", type=int, default=20)
    args = parser.parse_args()

    puuids = []

    if args.all or args.puuids:
        puuids = get_high_elo_puuids()
        Path("data/puuids.txt").write_text("\n".join(puuids))

    if args.all or args.matches:
        if not puuids and Path("data/puuids.txt").exists():
            puuids = Path("data/puuids.txt").read_text().splitlines()
        match_ids = get_match_ids(puuids, args.matches_per_player)
    else:
        match_ids = MATCH_IDS_FILE.read_text().splitlines() if MATCH_IDS_FILE.exists() else []

    if args.all or args.download or args.timelines or args.results or args.participants:
        download_all_match_data(match_ids)
