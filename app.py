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
    page_title="TEACHER",
    page_icon="🕹️",
    layout="centered",
)

# ── Global retro 80s CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    background-color: #05050f !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #c8d8e8 !important;
}

/* Scanlines overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.07) 0px,
        rgba(0,0,0,0.07) 1px,
        transparent 1px,
        transparent 3px
    );
    z-index: 9999;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #08081a !important;
    border-right: 1px solid #ff00ff44 !important;
}
[data-testid="stSidebar"] * {
    color: #00ffff !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 2px solid #ff00ff !important;
    color: #ff00ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 12px #ff00ff88, inset 0 0 12px rgba(255,0,255,0.05) !important;
    transition: all 0.2s ease !important;
    border-radius: 2px !important;
}
.stButton > button:hover {
    background: rgba(255,0,255,0.15) !important;
    box-shadow: 0 0 25px #ff00ffbb, inset 0 0 25px rgba(255,0,255,0.15) !important;
    color: #ffffff !important;
}

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #ff00ff, #00ffff) !important;
    box-shadow: 0 0 8px #00ffff88 !important;
}
[data-testid="stProgressBar"] {
    background-color: #1a1a2e !important;
    border: 1px solid #ffffff22 !important;
    border-radius: 2px !important;
}

/* ── Info / error boxes ── */
[data-testid="stAlert"] {
    background-color: #0a0a1e !important;
    border-left: 3px solid #00ffff !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ffff !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #ff00ff !important;
}

/* ── Divider ── */
hr {
    border-color: #ff00ff44 !important;
}

/* ── Captions / small text ── */
.stCaption, small {
    color: #ffffff55 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Item recommendation card ── */
.rec-card {
    background: #0a0a1e;
    border: 1px solid #00ffff44;
    border-left: 3px solid #ff00ff;
    border-radius: 2px;
    padding: 10px 14px;
    margin-bottom: 10px;
    box-shadow: 0 0 10px rgba(0,255,255,0.08);
    display: flex;
    align-items: center;
    gap: 14px;
}
.rec-card:hover {
    border-left-color: #00ffff;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
}
.rec-rank {
    font-size: 1.4rem;
    color: #ff00ff;
    text-shadow: 0 0 10px #ff00ff;
    min-width: 28px;
    text-align: center;
}
.rec-icon {
    width: 44px;
    height: 44px;
    border: 1px solid #00ffff66;
    border-radius: 2px;
    image-rendering: pixelated;
}
.rec-info {
    flex: 1;
}
.rec-name {
    color: #00ffff;
    text-shadow: 0 0 6px #00ffff88;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.rec-bar-wrap {
    background: #1a1a2e;
    border: 1px solid #ffffff22;
    border-radius: 1px;
    height: 6px;
    width: 100%;
}
.rec-bar {
    height: 6px;
    border-radius: 1px;
    background: linear-gradient(90deg, #ff00ff, #00ffff);
    box-shadow: 0 0 6px #00ffff88;
}
.rec-pct {
    color: #ffff00;
    text-shadow: 0 0 6px #ffff0088;
    font-size: 0.75rem;
    margin-top: 3px;
}

/* ── Context box ── */
.ctx-box {
    background: #08081a;
    border: 1px solid #ff00ff44;
    border-radius: 2px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: #aabbcc;
    letter-spacing: 0.05em;
}
.ctx-box span { color: #00ffff; }

/* ── Waiting text ── */
.waiting {
    text-align: center;
    color: #ff00ff66;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    padding: 2rem 0;
    text-shadow: 0 0 8px #ff00ff44;
}

/* ── Banner ── */
.retro-banner {
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    font-size: 0.72rem;
    line-height: 1.2;
    text-align: center;
    white-space: pre;
    background: linear-gradient(180deg, #ff00ff, #ff6600, #ffff00, #00ff99, #00ffff, #cc00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 8px #ff00ff);
    margin-bottom: 0.2rem;
}
.retro-sub {
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    text-align: center;
    color: #00ffff;
    letter-spacing: 0.25em;
    font-size: 0.85rem;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
    margin-bottom: 1rem;
}
.retro-divider {
    border: none;
    border-top: 2px solid;
    border-image: linear-gradient(90deg, transparent, #ff00ff, #00ffff, #ff00ff, transparent) 1;
    margin: 0.8rem 0 1.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Parse CLI args ────────────────────────────────────────────────────────────
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
@st.cache_resource(show_spinner="[ LOADING MODEL... ]")
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

# ── Data Dragon version ───────────────────────────────────────────────────────
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


# ── Recommend ─────────────────────────────────────────────────────────────────
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


# ── Render recommendations ────────────────────────────────────────────────────
def _show_recs(recs: list[dict], context: dict):
    gd = context['gold_diff']
    gd_color = "#00ff99" if gd >= 0 else "#ff4444"
    gd_str   = f"+{gd:,}" if gd >= 0 else f"{gd:,}"
    st.markdown(f"""
    <div class="ctx-box">
        <span>{context['champ']}</span> &nbsp;·&nbsp; {context['role']}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        KDA <span>{context['kda']}</span>
        &nbsp;·&nbsp; CS <span>{context['cs']}</span>
        &nbsp;·&nbsp; <span>{context['game_time']:.1f}</span> min
        &nbsp;·&nbsp; Gold <span style="color:{gd_color};text-shadow:0 0 6px {gd_color}88">{gd_str}</span>
    </div>
    """, unsafe_allow_html=True)

    cards_html = ""
    for i, rec in enumerate(recs, 1):
        pct      = rec["prob"] * 100
        bar_pct  = min(pct, 100)
        icon_url = ITEM_ICON_URL.format(ver=DD_VER, id=rec["item_id"])
        cards_html += f"""
        <div class="rec-card">
            <div class="rec-rank">#{i}</div>
            <img class="rec-icon" src="{icon_url}" onerror="this.style.display='none'">
            <div class="rec-info">
                <div class="rec-name">{rec['name']}</div>
                <div class="rec-bar-wrap">
                    <div class="rec-bar" style="width:{bar_pct}%"></div>
                </div>
                <div class="rec-pct">{pct:.1f}%</div>
            </div>
        </div>
        """
    st.markdown(cards_html, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-size:1.5rem;color:#ff00ff;text-shadow:0 0 15px #ff00ff">🕹️</div>
        <div style="color:#ff00ff;letter-spacing:0.3em;font-size:1rem;
                    text-shadow:0 0 10px #ff00ff;margin-top:4px">TEACHER</div>
        <div style="color:#ffffff44;font-size:0.65rem;letter-spacing:0.15em;margin-top:4px">
            AI ITEM ADVISOR
        </div>
    </div>
    <hr style="border-color:#ff00ff33;margin:0.8rem 0">
    """, unsafe_allow_html=True)

    if hasattr(inf.model, "blocks"):
        d = inf.model.projection.out_features
        l = len(inf.model.blocks)
        st.markdown(f"""
        <div style="color:#00ffff;font-size:0.75rem;letter-spacing:0.08em;padding:4px 0">
            ▸ TRANSFORMER<br>
            &nbsp;&nbsp;d_model &nbsp;= {d}<br>
            &nbsp;&nbsp;layers &nbsp;&nbsp;= {l}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#00ffff;font-size:0.75rem;letter-spacing:0.08em;padding:4px 0">
            ▸ MLP
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:#ff00ff33;margin:0.8rem 0">
    <div style="color:#ffffff33;font-size:0.65rem;line-height:1.8;letter-spacing:0.05em">
        TO SWITCH MODEL:<br>
        <span style="color:#ffff0066">-- --transformer</span><br>
        <span style="color:#ffff0066">-- --mlp</span>
    </div>
    """, unsafe_allow_html=True)


# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="retro-banner">  ████████╗███████╗ █████╗  ██████╗██╗  ██╗███████╗██████╗
  ╚══██╔══╝██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗
     ██║   █████╗  ███████║██║     ███████║█████╗  ██████╔╝
     ██║   ██╔══╝  ██╔══██║██║     ██╔══██║██╔══╝  ██╔══██╗
     ██║   ███████╗██║  ██║╚██████╗██║  ██║███████╗██║  ██║
     ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝</div>
<div class="retro-sub">★  R E A L - T I M E   A I   I T E M   A D V I S O R  ★</div>
<hr class="retro-divider">
""", unsafe_allow_html=True)

# ── Status / button ───────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#ffffff44;font-size:0.72rem;
            letter-spacing:0.12em;margin-bottom:1rem">
    ⚠ &nbsp; LEAGUE OF LEGENDS MUST BE OPEN AND IN-GAME
</div>
""", unsafe_allow_html=True)

fetch = st.button("[ REFRESH ]", type="primary", use_container_width=True)

if fetch:
    with st.spinner("CONNECTING TO LIVE API..."):
        try:
            from live_inference import fetch_game_state, parse_live_state
            raw = fetch_game_state()
            if raw is None:
                st.error("NO SIGNAL — are you in a game?")
            else:
                state = parse_live_state(raw)
                if state is None:
                    st.error("PARSE ERROR — could not read game state.")
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
            st.error(f"ERROR — {e}")

# ── Recommendations ───────────────────────────────────────────────────────────
if "live_recs" in st.session_state:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _show_recs(st.session_state["live_recs"], st.session_state["live_ctx"])
    last = st.session_state.get("last_update", "—")
    st.markdown(f"""
    <div style="text-align:center;color:#ffffff33;font-size:0.65rem;
                letter-spacing:0.1em;margin-top:1rem">
        LAST UPDATE: {last} &nbsp;·&nbsp; AUTO-REFRESH EVERY 5s
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="waiting">
        . . . &nbsp; AWAITING GAME DATA &nbsp; . . .
    </div>
    """, unsafe_allow_html=True)

# ── Auto-refresh every 5 s ────────────────────────────────────────────────────
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
