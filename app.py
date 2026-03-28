"""
Streamlit UI for the LoL Item Recommender.

Run:
    streamlit run app.py
    streamlit run app.py -- --transformer
    streamlit run app.py -- --checkpoint checkpoints/transformer/2026-03-28_21-00-00/best_model.pt
"""

import sys
import time
import argparse

import torch
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoL Item Recommender",
    page_icon="⚔️",
    layout="centered",
)

# ── Parse CLI args (everything after --) ─────────────────────────────────────
@st.cache_resource
def _parse_cli():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mlp",         action="store_true")
    group.add_argument("--transformer", action="store_true")
    group.add_argument("--checkpoint",  type=str, default=None)
    try:
        idx  = sys.argv.index("--")
        args = parser.parse_args(sys.argv[idx + 1:])
    except ValueError:
        args = parser.parse_args([])
    return args


# ── Load inference module once ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_inf():
    import inference as _inf
    cli = _parse_cli()
    if cli.mlp:
        _inf.model = _inf.load_model(arch="mlp")
    elif cli.transformer:
        _inf.model = _inf.load_model(arch="transformer")
    elif cli.checkpoint:
        _inf.model = _inf.load_model(checkpoint=cli.checkpoint)
    return _inf

inf = _load_inf()

# ── Data Dragon version (for item icons) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _dd_version():
    try:
        import urllib.request, json
        with urllib.request.urlopen(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=3
        ) as r:
            return json.loads(r.read())[0]
    except Exception:
        return "14.10.1"

DD_VER        = _dd_version()
ITEM_ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{ver}/img/item/{id}.png"


# ── Helper: recommend → list of dicts ────────────────────────────────────────
def _recommend(game_state: dict, top_k: int = 5) -> list[dict]:
    x     = inf.encode_state(game_state)
    buyer = next(p for p in game_state["players"] if p["is_buyer"])
    owned = {inf.item_to_idx[iid] for iid in buyer["items"] if iid in inf.item_to_idx}

    with torch.no_grad():
        logits = inf.model(x)
        for idx in owned:
            logits[0, idx] = float("-inf")
        probs = torch.softmax(logits, dim=1)[0]
        topk  = probs.topk(top_k)

    return [
        {
            "item_id": inf.idx_to_item[idx.item()],
            "name":    inf.item_names.get(inf.idx_to_item[idx.item()], f"Item {idx.item()}"),
            "prob":    prob.item(),
        }
        for prob, idx in zip(topk.values, topk.indices)
    ]


def _show_recs(recs: list[dict], context: dict):
    st.markdown(
        f"**{context['champ']}** ({context['role']}) · "
        f"KDA {context['kda']} · CS {context['cs']} · "
        f"{context['game_time']:.1f} min · Gold diff {context['gold_diff']:+}"
    )
    st.divider()
    for i, rec in enumerate(recs, 1):
        pct      = rec["prob"] * 100
        icon_url = ITEM_ICON_URL.format(ver=DD_VER, id=rec["item_id"])
        col_icon, col_text = st.columns([1, 8])
        with col_icon:
            st.image(icon_url, width=40)
        with col_text:
            st.markdown(f"**{i}. {rec['name']}**")
            st.progress(min(rec["prob"], 1.0), text=f"{pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — model info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚔️ LoL Item\nRecommender")
    st.divider()
    if hasattr(inf.model, "blocks"):
        d = inf.model.projection.out_features
        l = len(inf.model.blocks)
        st.info(f"🤖 Transformer  d={d}  layers={l}")
    else:
        st.info("🤖 MLP")
    st.divider()
    st.caption("To switch model, restart with:\n`streamlit run app.py -- --transformer`\nor `-- --mlp`")


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚔️ LoL Item Recommender")
st.caption("AI-powered item suggestions — runs entirely on your machine.")
st.divider()

st.info(
    "League of Legends must be **open and in a game** for this to work. "
    "The app reads from the local Riot Live Client API (port 2999)."
)

fetch = st.button("🔄 Refresh Now", type="primary", use_container_width=True)

if fetch:
    with st.spinner("Connecting to League of Legends…"):
        try:
            from live_inference import fetch_game_state, parse_live_state
            raw = fetch_game_state()
            if raw is None:
                st.error("Could not connect to the Live API. Are you in a game?")
            else:
                state = parse_live_state(raw)
                if state is None:
                    st.error("Could not parse game state.")
                else:
                    buyer      = next(p for p in state["players"] if p["is_buyer"])
                    champ_name = inf.champion_names.get(buyer.get("champion_id", 0), "Unknown")
                    recs       = _recommend(state, top_k=5)
                    st.session_state["live_recs"] = recs
                    st.session_state["live_ctx"]  = {
                        "champ":     champ_name,
                        "role":      buyer.get("role", "?"),
                        "kda":       f"{buyer['kills']}/{buyer['deaths']}/{buyer['assists']}",
                        "cs":        buyer["cs"],
                        "game_time": state["game_time_min"],
                        "gold_diff": state["team_gold_diff"],
                    }
                    st.session_state["last_update"] = time.strftime("%H:%M:%S")
        except Exception as e:
            st.error(f"Error: {e}")

if "live_recs" in st.session_state:
    _show_recs(st.session_state["live_recs"], st.session_state["live_ctx"])
    st.caption(f"Last updated: {st.session_state.get('last_update', '—')}  ·  auto-refreshes every 5 s")
else:
    st.markdown("_Waiting for game data…_")

# Auto-refresh every 5 s
time.sleep(5)
try:
    from live_inference import fetch_game_state, parse_live_state
    raw = fetch_game_state()
    if raw:
        state = parse_live_state(raw)
        if state:
            buyer      = next(p for p in state["players"] if p["is_buyer"])
            champ_name = inf.champion_names.get(buyer.get("champion_id", 0), "Unknown")
            recs       = _recommend(state, top_k=5)
            st.session_state["live_recs"] = recs
            st.session_state["live_ctx"]  = {
                "champ":     champ_name,
                "role":      buyer.get("role", "?"),
                "kda":       f"{buyer['kills']}/{buyer['deaths']}/{buyer['assists']}",
                "cs":        buyer["cs"],
                "game_time": state["game_time_min"],
                "gold_diff": state["team_gold_diff"],
            }
            st.session_state["last_update"] = time.strftime("%H:%M:%S")
except Exception:
    pass
st.rerun()
