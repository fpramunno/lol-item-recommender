"""
Live inference — connects to the Riot Live Client Data API (port 2999)
and shows the top-5 recommended items for your champion in real time.

Launch League of Legends, enter a game, then run:
    python live_inference.py

Updates automatically every 30 seconds (or press ENTER to refresh manually).
"""

import json
import time
import urllib.request
import ssl
import threading
from pathlib import Path
from inference import encode_state, recommend, champion_names, item_to_idx, champion_to_idx, ROLES

LIVE_API = "https://127.0.0.1:2999/liveclientdata/allgamedata"
REFRESH_SECONDS  = 30   # periodic refresh
POLL_INTERVAL    = 3    # check for purchases every 3 seconds

# Data Dragon: map champion name → numeric ID (inverse of champion_names)
name_to_champion_id = {v: k for k, v in champion_names.items()}

# Item costs for computing gold_spent
item_costs_path = Path("data/item_costs.json")
item_costs = {int(k): v for k, v in json.loads(item_costs_path.read_text()).items()} if item_costs_path.exists() else {}

# Map API position → our model's role
POSITION_MAP = {
    "TOP":     "TOP",
    "JUNGLE":  "JUNGLE",
    "MIDDLE":  "MIDDLE",
    "BOTTOM":  "BOTTOM",
    "SUPPORT": "UTILITY",
    "UTILITY": "UTILITY",
    "":        "",
}


def fetch_game_state() -> dict | None:
    """Calls the Live Client API and returns the raw JSON."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(LIVE_API, context=ctx, timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[Error] Unable to connect to the Live API: {e}")
        print("Make sure you are in a League of Legends game.")
        return None


def parse_live_state(data: dict) -> dict | None:
    """
    Converts the Live API JSON into the format expected by our model.
    """
    game_time_min = data["gameData"]["gameTime"] / 60.0
    active        = data["activePlayer"]
    all_players   = data["allPlayers"]

    # Identify your summoner name
    my_name = active["summonerName"]

    # Find your player in the allPlayers array
    me = next((p for p in all_players if p["summonerName"] == my_name), None)
    if me is None:
        print("[Error] Local player not found in allPlayers.")
        return None

    my_team = me["team"]  # "ORDER" or "CHAOS"

    # Stat runes (offense/flex/defense) only available for active player
    full_runes = active.get("fullRunes", {})
    stat_runes = full_runes.get("statRunes", [])
    my_stat_runes = {
        "stat_offense": stat_runes[0].get("id", 0) if len(stat_runes) > 0 else 0,
        "stat_flex":    stat_runes[1].get("id", 0) if len(stat_runes) > 1 else 0,
        "stat_defense": stat_runes[2].get("id", 0) if len(stat_runes) > 2 else 0,
    }

    # Assign participant_id: ORDER = team 100 (1-5), CHAOS = team 200 (6-10)
    order_players  = [p for p in all_players if p["team"] == "ORDER"]
    chaos_players  = [p for p in all_players if p["team"] == "CHAOS"]

    players = []
    for pid_offset, team_players, team_id in [(0, order_players, 100), (5, chaos_players, 200)]:
        for i, p in enumerate(team_players):
            pid        = pid_offset + i + 1
            is_buyer   = (p["summonerName"] == my_name)
            champ_name = p["championName"]
            champ_id   = name_to_champion_id.get(champ_name, 0)
            role       = POSITION_MAP.get(p.get("position", "").upper(), "")

            scores = p.get("scores", {})
            kills   = int(scores.get("kills",   0))
            deaths  = int(scores.get("deaths",  0))
            assists = int(scores.get("assists", 0))
            cs      = int(scores.get("creepScore", 0))
            level   = int(p.get("level", 1))

            # Item IDs from inventory
            raw_items  = p.get("items", [])
            item_ids   = [it["itemID"] for it in raw_items if it.get("itemID", 0) > 0]
            gold_spent = sum(item_costs.get(iid, 0) for iid in item_ids)

            gold_current = int(active.get("currentGold", 0)) if is_buyer else None

            p_runes = p.get("runes", {})
            runes = {
                "keystone":       p_runes.get("keystone",         {}).get("id", 0),
                "primary_tree":   p_runes.get("primaryRuneTree",  {}).get("id", 0),
                "secondary_tree": p_runes.get("secondaryRuneTree",{}).get("id", 0),
                **(my_stat_runes if is_buyer else {"stat_offense": 0, "stat_flex": 0, "stat_defense": 0}),
            }
            players.append({
                "participant_id": pid,
                "team":           team_id,
                "champion_id":    champ_id,
                "role":           role,
                "level":          level,
                "kills":          kills,
                "deaths":         deaths,
                "assists":        assists,
                "cs":             cs,
                "gold_spent":     gold_spent,
                "gold_current":   gold_current,
                "items":          item_ids,
                "is_buyer":       is_buyer,
                **runes,
            })

    # Gold diff: estimated from gold_spent (totalGold not available in live API)
    my_team_id    = 100 if my_team == "ORDER" else 200
    enemy_team_id = 200 if my_team_id == 100 else 100
    my_gold    = sum(p["gold_spent"] for p in players if p["team"] == my_team_id)
    enemy_gold = sum(p["gold_spent"] for p in players if p["team"] == enemy_team_id)
    team_gold_diff = my_gold - enemy_gold

    return {
        "game_time_min":  round(game_time_min, 2),
        "team_gold_diff": team_gold_diff,
        "players":        players,
    }


def get_buyer_items(raw: dict) -> frozenset:
    """Returns the buyer's current items as a frozenset for comparison."""
    try:
        active_name = raw["activePlayer"]["summonerName"]
        me = next(p for p in raw["allPlayers"] if p["summonerName"] == active_name)
        return frozenset(it["itemID"] for it in me.get("items", []) if it.get("itemID", 0) > 0)
    except Exception:
        return frozenset()


BANNER_LINES = [
    "  ████████╗███████╗ █████╗  ██████╗██╗  ██╗███████╗██████╗ ",
    "  ╚══██╔══╝██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗",
    "     ██║   █████╗  ███████║██║     ███████║█████╗  ██████╔╝ ",
    "     ██║   ██╔══╝  ██╔══██║██║     ██╔══██║██╔══╝  ██╔══██╗ ",
    "     ██║   ███████╗██║  ██║╚██████╗██║  ██║███████╗██║  ██║ ",
    "     ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝",
    "          LoL Item Recommender — Real-Time AI Advisor        ",
]

# ANSI 256-color rainbow cycle (bright 80s palette)
RAINBOW = [196, 202, 208, 214, 220, 226, 154, 46, 51, 39, 27, 57, 129, 165, 201]

def print_banner():
    print()
    color_idx = 0
    for line in BANNER_LINES:
        out = ""
        for ch in line:
            if ch.strip():
                color = RAINBOW[color_idx % len(RAINBOW)]
                out += f"\x1b[38;5;{color}m{ch}"
                color_idx += 1
            else:
                out += ch
        print(out + "\x1b[0m")
    print()

def run():
    print_banner()
    print("  Updates automatically when you buy an item.")
    print("  Press ENTER to refresh manually, Ctrl+C to exit.")
    print("  " + "─" * 56)

    last_items: frozenset = frozenset()
    last_refresh: float = 0.0

    def update(reason: str = ""):
        raw = fetch_game_state()
        if raw is None:
            return None
        state = parse_live_state(raw)
        if state is None:
            return None
        print(f"\n[{time.strftime('%H:%M:%S')}] {reason}")
        recommend(state, top_k=5)
        return raw

    def poll_loop():
        nonlocal last_items, last_refresh
        while True:
            time.sleep(POLL_INTERVAL)
            raw = fetch_game_state()
            if raw is None:
                continue

            current_items = get_buyer_items(raw)
            now = time.time()

            # Purchase detected
            if current_items != last_items and last_items:
                new = current_items - last_items
                names = [item_names.get(iid, f"Item {iid}") for iid in new]
                state = parse_live_state(raw)
                if state:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Purchase detected: {', '.join(names)}")
                    recommend(state, top_k=5)
                last_refresh = now

            # Periodic refresh
            elif now - last_refresh > REFRESH_SECONDS:
                state = parse_live_state(raw)
                if state:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Periodic refresh")
                    recommend(state, top_k=5)
                last_refresh = now

            last_items = current_items

    # Load item_names to display what was purchased
    from inference import item_names

    # Polling thread
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    # First recommendation immediately
    raw = update("Starting up")
    if raw:
        last_items = get_buyer_items(raw)
    last_refresh = time.time()

    # Manual refresh on ENTER
    try:
        while True:
            input()
            raw = update("Manual refresh")
            if raw:
                last_items = get_buyer_items(raw)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    import argparse
    import inference as _inf
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mlp",         action="store_true", help="Load best MLP model")
    group.add_argument("--transformer", action="store_true", help="Load best Transformer model")
    group.add_argument("--checkpoint",  type=str, default=None,
                       help="Path to a specific checkpoint, e.g. checkpoints/transformer/2026-03-28_21-00-00/best_model.pt")
    cli = parser.parse_args()
    if cli.mlp:
        _inf.model = _inf.load_model(arch="mlp")
    elif cli.transformer:
        _inf.model = _inf.load_model(arch="transformer")
    elif cli.checkpoint:
        _inf.model = _inf.load_model(checkpoint=cli.checkpoint)
    run()
