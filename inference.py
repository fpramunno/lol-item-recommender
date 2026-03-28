"""
Tests the model on specific game situations.
Shows the top-5 suggested items with human-readable names.
"""

import json
import torch
import numpy as np
from pathlib import Path
from model import build_model

PROCESSED_DIR  = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

# ── Load resources ─────────────────────────────────────────────────────────────

def load_model(checkpoint: str = None, arch: str = None):
    if checkpoint:
        model_path = Path(checkpoint)
    elif arch:
        # match by config name (e.g. "transformer", "transformer_large", "mlp")
        # supports both checkpoints/mlp/best_model.pt and checkpoints/mlp/<datetime>/best_model.pt
        candidates = (
            list(CHECKPOINT_DIR.glob(f"*{arch}*/*/best_model.pt")) +
            list(CHECKPOINT_DIR.glob(f"*{arch}*/best_model.pt"))
        )
        if not candidates:
            raise FileNotFoundError(f"No best_model.pt found matching '{arch}' in checkpoints/")
        # Pick the one with lowest val_loss stored in the checkpoint
        def _val_loss(p):
            try:
                return torch.load(p, map_location="cpu").get("val_loss", float("inf"))
            except Exception:
                return float("inf")
        model_path = min(candidates, key=_val_loss)
    else:
        legacy = CHECKPOINT_DIR / "best_model.pt"
        if legacy.exists():
            model_path = legacy
        else:
            candidates = sorted(
                list(CHECKPOINT_DIR.glob("*/*/best_model.pt")) +
                list(CHECKPOINT_DIR.glob("*/best_model.pt"))
            )
            if not candidates:
                raise FileNotFoundError("No best_model.pt found in checkpoints/")
            model_path = candidates[-1]
    ckpt        = torch.load(model_path, map_location="cpu")
    arch_config = ckpt.get("arch_config", {"arch": "mlp", "hidden_dims": [1024, 512, 256], "dropout": 0.0})
    model       = build_model(arch_config, ckpt["input_dim"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    arch = arch_config["arch"]
    step = ckpt.get("step", "?")
    if arch == "transformer":
        desc = (f"Transformer  d_model={arch_config.get('d_model')}  "
                f"n_heads={arch_config.get('n_heads')}  "
                f"n_layers={arch_config.get('n_layers')}  "
                f"ffn_dim={arch_config.get('ffn_dim')}")
    else:
        desc = f"MLP  hidden={arch_config.get('hidden_dims')}"
    print(f"  Model loaded: {desc}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Checkpoint  : {model_path}  (step {step})")
    return model

def load_item_names() -> dict[int, str]:
    """Downloads item names from Data Dragon."""
    cache = Path("data/item_names.json")
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    import urllib.request
    versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    with urllib.request.urlopen(versions_url) as r:
        latest = json.loads(r.read())[0]
    items_url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json"
    with urllib.request.urlopen(items_url) as r:
        data = json.loads(r.read())
    names = {int(k): v["name"] for k, v in data["data"].items()}
    cache.write_text(json.dumps(names))
    return names

item_to_idx      = {int(k): v for k, v in json.loads((PROCESSED_DIR / "item_to_idx.json").read_text()).items()}
idx_to_item      = {v: k for k, v in item_to_idx.items()}
champion_to_idx  = {int(k): v for k, v in json.loads((PROCESSED_DIR / "champion_to_idx.json").read_text()).items()}
stats            = json.loads((PROCESSED_DIR / "feature_stats.json").read_text())
item_names       = load_item_names()
model            = load_model()

def load_champion_names() -> dict[int, str]:
    cache = Path("data/champion_names.json")
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    import urllib.request
    with urllib.request.urlopen("https://ddragon.leagueoflegends.com/api/versions.json") as r:
        latest = json.loads(r.read())[0]
    with urllib.request.urlopen(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json") as r:
        data = json.loads(r.read())
    names = {int(v["key"]): v["name"] for v in data["data"].values()}
    cache.write_text(json.dumps(names))
    return names

champion_names = load_champion_names()

# Load champion tags and normalized stats from cache
_champ_cache = json.loads(Path("data/champion_features.json").read_text())
champ_tags_inf       = {int(k): np.array(v, dtype=np.float32) for k, v in _champ_cache["champ_tags"].items()}
champ_stats_norm_inf = {int(k): np.array(v, dtype=np.float32) for k, v in _champ_cache["champ_stats_norm"].items()}

CHAMPION_TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
ROLES         = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
CHAMP_STAT_KEYS_LEN = 13

TREE_IDS         = [8000, 8100, 8200, 8300, 8400]
KEYSTONE_IDS     = [8005, 8008, 8010, 8021, 8112, 8124, 8128, 9923,
                    8214, 8229, 8230, 8351, 8360, 8369, 8437, 8439, 8465]
STAT_OFFENSE_IDS = [5005, 5007, 5008]
STAT_FLEX_IDS    = [5001, 5007, 5008]
STAT_DEFENSE_IDS = [5001, 5011, 5013]

TREE_IDX         = {v: i for i, v in enumerate(TREE_IDS)}
KEYSTONE_IDX     = {v: i for i, v in enumerate(KEYSTONE_IDS)}
STAT_OFFENSE_IDX = {v: i for i, v in enumerate(STAT_OFFENSE_IDS)}
STAT_FLEX_IDX    = {v: i for i, v in enumerate(STAT_FLEX_IDS)}
STAT_DEFENSE_IDX = {v: i for i, v in enumerate(STAT_DEFENSE_IDS)}

# ── Encoding ──────────────────────────────────────────────────────────────────

def norm(value, feat):
    s = stats[feat]
    std = s["max"] - s["min"]
    if std == 0:
        return 0.0
    normalized = (value - s["min"]) / std
    return (normalized - 0.5) / 0.5

def encode_state(game_state: dict) -> torch.Tensor:
    """Converts a game state into the X vector."""
    n_items  = len(item_to_idx)
    n_champs = len(champion_to_idx)
    X = game_state

    global_feats = np.array([
        norm(X["game_time_min"],  "game_time_min"),
        norm(X["team_gold_diff"], "team_gold_diff"),
    ], dtype=np.float32)

    players = sorted(X["players"], key=lambda p: p["participant_id"])
    player_feats = []
    for p in players:
        numeric = np.array([
            norm(p["level"],                "level"),
            norm(p["kills"],                "kills"),
            norm(p["deaths"],               "deaths"),
            norm(p["assists"],              "assists"),
            norm(p["cs"],                   "cs"),
            norm(p["gold_spent"],           "gold_spent"),
            norm(p["gold_current"] or 0,    "gold_current"),
            float(p["is_buyer"]),
        ], dtype=np.float32)

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
        tags_vec = champ_tags_inf.get(cid, np.zeros(len(CHAMPION_TAGS), dtype=np.float32))

        # Role one-hot (5)
        role_vec = np.zeros(len(ROLES), dtype=np.float32)
        role = p.get("role", "")
        if role in ROLES:
            role_vec[ROLES.index(role)] = 1.0

        # Normalized champion base stats (13)
        stats_vec = champ_stats_norm_inf.get(cid, np.zeros(CHAMP_STAT_KEYS_LEN, dtype=np.float32))

        # Rune encoding
        keystone_vec  = np.zeros(len(KEYSTONE_IDS),     dtype=np.float32)
        primary_vec   = np.zeros(len(TREE_IDS),         dtype=np.float32)
        secondary_vec = np.zeros(len(TREE_IDS),         dtype=np.float32)
        stat_off_vec  = np.zeros(len(STAT_OFFENSE_IDS), dtype=np.float32)
        stat_flx_vec  = np.zeros(len(STAT_FLEX_IDS),    dtype=np.float32)
        stat_def_vec  = np.zeros(len(STAT_DEFENSE_IDS), dtype=np.float32)
        kidx = KEYSTONE_IDX.get(p.get("keystone", 0))
        if kidx is not None: keystone_vec[kidx] = 1.0
        pidx = TREE_IDX.get(p.get("primary_tree", 0))
        if pidx is not None: primary_vec[pidx] = 1.0
        sidx = TREE_IDX.get(p.get("secondary_tree", 0))
        if sidx is not None: secondary_vec[sidx] = 1.0
        oidx = STAT_OFFENSE_IDX.get(p.get("stat_offense", 0))
        if oidx is not None: stat_off_vec[oidx] = 1.0
        fidx = STAT_FLEX_IDX.get(p.get("stat_flex", 0))
        if fidx is not None: stat_flx_vec[fidx] = 1.0
        didx = STAT_DEFENSE_IDX.get(p.get("stat_defense", 0))
        if didx is not None: stat_def_vec[didx] = 1.0

        player_feats.append(np.concatenate([
            numeric, item_vec, champ_vec, tags_vec, role_vec, stats_vec,
            keystone_vec, primary_vec, secondary_vec, stat_off_vec, stat_flx_vec, stat_def_vec,
        ]))

    x = np.concatenate([global_feats] + player_feats)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


# ── Inference ─────────────────────────────────────────────────────────────────

def recommend(game_state: dict, top_k: int = 5):
    x = encode_state(game_state)

    # Find items already owned by the buyer
    buyer = next(p for p in game_state["players"] if p["is_buyer"])
    owned_indices = {item_to_idx[iid] for iid in buyer["items"] if iid in item_to_idx}

    with torch.no_grad():
        logits = model(x)
        # Zero out logits for already-owned items
        for idx in owned_indices:
            logits[0, idx] = float("-inf")
        probs  = torch.softmax(logits, dim=1)[0]
        topk   = probs.topk(top_k)

    buyer = next(p for p in game_state["players"] if p["is_buyer"])
    champ_name = champion_names.get(buyer.get("champion_id", 0), f"ID {buyer.get('champion_id', '?')}")
    role       = buyer.get("role", "?")
    kda        = f"{buyer['kills']}/{buyer['deaths']}/{buyer['assists']}"
    print(f"\nBuyer: {champ_name} ({role}) | KDA: {kda} | CS: {buyer['cs']}")
    print(f"Game time: {game_state['game_time_min']} min | Gold diff: {game_state['team_gold_diff']:+}")
    print(f"Top-{top_k} recommended items:")
    for rank, (prob, idx) in enumerate(zip(topk.values, topk.indices), 1):
        item_id   = idx_to_item[idx.item()]
        item_name = item_names.get(item_id, f"Item {item_id}")
        print(f"  {rank}. {item_name:<35} ({prob.item()*100:.1f}%)")


# ── Test scenarios ─────────────────────────────────────────────────────────────

def make_player(pid, team, champion_id, role, level, kills, deaths, assists, cs, gold_spent, items, is_buyer, gold_current=0,
                keystone=0, primary_tree=0, secondary_tree=0, stat_offense=0, stat_flex=0, stat_defense=0):
    return {
        "participant_id": pid,
        "team": team,
        "champion_id": champion_id,
        "role": role,
        "level": level,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "cs": cs,
        "gold_spent": gold_spent,
        "items": items,
        "is_buyer": is_buyer,
        "gold_current": gold_current if is_buyer else None,
        "keystone": keystone,
        "primary_tree": primary_tree,
        "secondary_tree": secondary_tree,
        "stat_offense": stat_offense,
        "stat_flex": stat_flex,
        "stat_defense": stat_defense,
    }


# ── Random scenario generator ─────────────────────────────────────────────────

import random

# Realistic items by role (mid/late game, >900g)
ITEMS_BY_ROLE = {
    "TOP":     [3068, 3082, 3742, 3143, 3748, 3071, 3036, 3053, 3111, 3065],
    "JUNGLE":  [3748, 3111, 3147, 3142, 3071, 3053, 6673, 6631, 6632, 3041],
    "MIDDLE":  [3157, 3089, 3165, 3135, 3040, 3102, 3285, 4645, 3100, 3916],
    "BOTTOM":  [3031, 3006, 3094, 3036, 3046, 6672, 6675, 3095, 3087, 3139],
    "UTILITY": [3190, 3107, 3222, 3109, 3050, 3117, 2065, 3867, 3011, 3853],
}

def rand_items(role: str, n: int) -> list[int]:
    pool = ITEMS_BY_ROLE.get(role, ITEMS_BY_ROLE["MIDDLE"])
    return random.sample(pool, min(n, len(pool)))

def rand_player(pid: int, team: int, role: str, game_time: float,
                is_buyer: bool = False, gold_current: int = 0) -> dict:
    """Generates a player with plausible stats for the given game_time."""
    base_level  = max(1, int(game_time * 0.8) + random.randint(-1, 2))
    level       = min(18, base_level)
    cs_rate     = {"TOP": 7, "JUNGLE": 4, "MIDDLE": 7, "BOTTOM": 8, "UTILITY": 1}.get(role, 6)
    cs          = int(game_time * cs_rate + random.gauss(0, 15))
    kills       = max(0, int(random.expovariate(1/2.5)))
    deaths      = max(0, int(random.expovariate(1/2)))
    assists     = max(0, int(random.expovariate(1/3)))
    gold_per_cs = 22
    gold_spent  = int(cs * gold_per_cs + kills * 300 + game_time * 80 + random.gauss(0, 300))
    gold_spent  = max(0, gold_spent)
    n_items     = min(6, max(0, int(gold_spent / 2800)))
    items       = rand_items(role, n_items)
    return make_player(pid, team, random.choice(list(champion_to_idx.keys())), role,
                       level, kills, deaths, assists, max(0, cs),
                       gold_spent, items, is_buyer, gold_current)

def random_scenario(label: str = None):
    """Generates and recommends a realistic random scenario."""
    game_time   = round(random.uniform(10, 35), 1)
    gold_diff   = random.randint(-6000, 6000)
    buyer_role  = random.choice(ROLES)
    buyer_gold  = random.randint(900, 3500)

    roles_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    players = []
    for i, role in enumerate(roles_order):
        is_buyer = (role == buyer_role)
        players.append(rand_player(i+1, 100, role, game_time,
                                   is_buyer=is_buyer,
                                   gold_current=buyer_gold if is_buyer else 0))
    for i, role in enumerate(roles_order):
        players.append(rand_player(i+6, 200, role, game_time))

    title = label or f"Random — {buyer_role} | {game_time} min | gold diff {gold_diff:+}"
    print("\n" + "="*60)
    print(title)
    recommend({"game_time_min": game_time, "team_gold_diff": gold_diff, "players": players})


def rand_buyer(role: str, game_time: float, gold_current: int,
               kills: int = None, deaths: int = None, cs: int = None) -> dict:
    """Buyer with optional stat overrides, everything else plausible."""
    cs_rate    = {"TOP": 7, "JUNGLE": 4, "MIDDLE": 7, "BOTTOM": 8, "UTILITY": 1}.get(role, 6)
    cs         = cs if cs is not None else max(0, int(game_time * cs_rate + random.gauss(0, 15)))
    kills      = kills if kills is not None else max(0, int(random.expovariate(1/2.5)))
    deaths     = deaths if deaths is not None else max(0, int(random.expovariate(1/2)))
    assists    = max(0, int(random.expovariate(1/3)))
    gold_spent = max(0, int(cs * 22 + kills * 300 + game_time * 80 + random.gauss(0, 200)))
    n_items    = min(6, max(0, int(gold_spent / 2800)))
    items      = rand_items(role, n_items)
    level      = min(18, max(1, int(game_time * 0.8) + random.randint(-1, 2)))
    return make_player(1, 100, BUYER_CHAMPION, role,
                       level, kills, deaths, assists, cs,
                       gold_spent, items, True, gold_current)


# External contexts: who is fed / game situation
CONTEXTS = {
    "You are fed (10+ kills)": dict(
        buyer_kills=random.randint(10, 18), buyer_deaths=random.randint(0, 2),
        gold_diff=random.randint(2000, 6000),
    ),
    "You are struggling (0/5+)": dict(
        buyer_kills=0, buyer_deaths=random.randint(5, 10),
        gold_diff=random.randint(-5000, -1000),
    ),
    "Even game": dict(
        buyer_kills=None, buyer_deaths=None,
        gold_diff=random.randint(-500, 500),
    ),
    "Ally jungle fed": dict(
        buyer_kills=None, buyer_deaths=None,
        gold_diff=random.randint(1000, 4000),
        ally_feeder_role="JUNGLE", ally_feeder_kills=12,
    ),
    "Enemy ADC fed": dict(
        buyer_kills=None, buyer_deaths=None,
        gold_diff=random.randint(-4000, -1000),
        enemy_feeder_role="BOTTOM", enemy_feeder_kills=15,
    ),
}


def fixed_buyer_scenarios(champion_id: int, role: str, n: int = 5):
    """
    Runs N scenarios with a fixed buyer (champion + role), but randomized external contexts.
    Useful for understanding how recommendations change as the game situation varies.
    """
    global BUYER_CHAMPION
    BUYER_CHAMPION = champion_id
    champ_name = champion_names.get(champion_id, f"ID {champion_id}")

    context_names = list(CONTEXTS.keys())
    random.shuffle(context_names)

    roles_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

    for i, ctx_name in enumerate(context_names[:n]):
        ctx         = CONTEXTS[ctx_name]
        game_time   = round(random.uniform(12, 30), 1)
        gold_diff   = ctx.get("gold_diff", random.randint(-3000, 3000))
        buyer_gold  = random.randint(900, 3500)

        buyer = rand_buyer(
            role, game_time, buyer_gold,
            kills=ctx.get("buyer_kills"),
            deaths=ctx.get("buyer_deaths"),
        )

        players = [buyer]
        for j, r in enumerate(roles_order):
            if r == role:
                continue
            kills_override = ctx.get("ally_feeder_kills") if ctx.get("ally_feeder_role") == r else None
            p = rand_player(j+2, 100, r, game_time)
            if kills_override:
                p["kills"] = kills_override
            players.append(p)

        for j, r in enumerate(roles_order):
            kills_override = ctx.get("enemy_feeder_kills") if ctx.get("enemy_feeder_role") == r else None
            p = rand_player(j+6, 200, r, game_time)
            if kills_override:
                p["kills"] = kills_override
            players.append(p)

        # Reassign participant_ids in order
        for idx, p in enumerate(sorted(players, key=lambda x: x["participant_id"])):
            p["participant_id"] = idx + 1

        print("\n" + "="*60)
        print(f"{champ_name} ({role}) — {ctx_name}")
        recommend({"game_time_min": game_time, "team_gold_diff": gold_diff, "players": players})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", type=int, default=None,
                        help="Riot champion ID (e.g. 54=Malphite, 61=Orianna, 222=Jinx)")
    parser.add_argument("--role", type=str, default=None,
                        help="Role: TOP JUNGLE MIDDLE BOTTOM UTILITY")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of scenarios (default 3)")
    args = parser.parse_args()

    if args.champion and args.role:
        fixed_buyer_scenarios(args.champion, args.role.upper(), args.n)
    else:
        for i in range(1, args.n + 1):
            random_scenario(f"SCENARIO {i}")
