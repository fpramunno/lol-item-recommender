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

REQUEST_DELAY = 1.2  # 100 req/120s = 0.83/s → 1.3s has safety margin


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


# ── 3. Download Timelines ─────────────────────────────────────────────────────

def download_timelines(match_ids: list[str]):
    already = {f.stem for f in TIMELINES_DIR.glob("*.json")}
    to_download = [mid for mid in match_ids if mid not in already]
    log.info(f"Timelines to download: {len(to_download)} (already present: {len(already)})")

    for i, match_id in enumerate(tqdm(to_download, desc="timelines")):
        url = f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        data = get(url)
        if data:
            (TIMELINES_DIR / f"{match_id}.json").write_text(json.dumps(data))

        # Save progress every 500 matches for safety
        if (i + 1) % 500 == 0:
            log.info(f"Checkpoint: {i + 1} timelines downloaded")

        time.sleep(REQUEST_DELAY)

    log.info("Timeline download complete.")


# ── 4. Download Match Results ──────────────────────────────────────────────────

def download_match_results(match_ids: list[str]):
    results_file = Path("data/match_results.json")
    results = {}

    if results_file.exists():
        results = json.loads(results_file.read_text())

    to_fetch = [mid for mid in match_ids if mid not in results]
    log.info(f"Match results to download: {len(to_fetch)}")

    for i, match_id in enumerate(tqdm(to_fetch, desc="match results")):
        url = f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        data = get(url)
        if data:
            for team in data["info"]["teams"]:
                results[match_id] = {"winner": team["teamId"] if team["win"] else (300 - team["teamId"])}
                break

        if (i + 1) % 500 == 0:
            results_file.write_text(json.dumps(results))

        time.sleep(REQUEST_DELAY)

    results_file.write_text(json.dumps(results))
    log.info("Match results saved.")


# ── 5. Download Match Participants (championId + teamPosition) ─────────────────

def download_match_participants(match_ids: list[str]):
    """Saves championId, teamPosition, and runes for each participant of each match."""
    participants = {}
    if PARTICIPANTS_FILE.exists():
        participants = json.loads(PARTICIPANTS_FILE.read_text())

    # Only download matches for which we already have a timeline
    timeline_ids = {f.stem for f in TIMELINES_DIR.glob("*.json")}
    relevant = [mid for mid in match_ids if mid in timeline_ids]

    # Re-fetch matches already downloaded but missing rune data
    def has_runes(entry: dict) -> bool:
        values = list(entry.values())
        return len(values) > 0 and "keystone" in values[0]

    to_fetch = [mid for mid in relevant if mid not in participants or not has_runes(participants[mid])]
    log.info(f"Timelines present: {len(timeline_ids)} | Participants to download: {len(to_fetch)} (already present with runes: {len(relevant) - len(to_fetch)})")

    for i, match_id in enumerate(tqdm(to_fetch, desc="participants")):
        url = f"https://{CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        data = get(url)
        if data:
            pinfo = {}
            for p in data["info"]["participants"]:
                perks       = p.get("perks", {})
                styles      = perks.get("styles", [])
                stat_perks  = perks.get("statPerks", {})
                primary     = next((s for s in styles if s["description"] == "primaryStyle"), {})
                secondary   = next((s for s in styles if s["description"] == "subStyle"), {})
                keystone    = primary.get("selections", [{}])[0].get("perk", 0)

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

        if (i + 1) % 500 == 0:
            PARTICIPANTS_FILE.write_text(json.dumps(participants))
            log.info(f"Checkpoint: {i + 1} match participants saved")

        time.sleep(REQUEST_DELAY)

    PARTICIPANTS_FILE.write_text(json.dumps(participants))
    log.info(f"Match participants saved: {len(participants)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--puuids", action="store_true")
    parser.add_argument("--matches", action="store_true")
    parser.add_argument("--timelines", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--participants", action="store_true")
    parser.add_argument("--all", action="store_true")
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

    if args.all or args.timelines:
        download_timelines(match_ids)

    if args.all or args.results:
        download_match_results(match_ids)

    if args.all or args.participants:
        download_match_participants(match_ids)
