```
  ████████╗███████╗ █████╗  ██████╗██╗  ██╗███████╗██████╗
  ╚══██╔══╝██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗
     ██║   █████╗  ███████║██║     ███████║█████╗  ██████╔╝
     ██║   ██╔══╝  ██╔══██║██║     ██╔══██║██╔══╝  ██╔══██╗
     ██║   ███████╗██║  ██║╚██████╗██║  ██║███████╗██║  ██║
     ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
          LoL Item Recommender — Real-Time AI Advisor
```

# Teacher — LoL Item Recommender

An AI that recommends the next best item to buy in League of Legends, based on the full live game state: your champion, role, KDA, CS, gold, items owned, runes, and the entire team composition.

Works in real time via the Riot Live Client Data API — no manual input required.

---

## How it works

The model is trained on millions of item purchase events from high-elo EUW matches (Challenger / Grandmaster / Master). At each purchase, it receives a snapshot of the full game state and learns to predict which item was bought.

**Input — 4482 features per sample**
| Feature group | Details |
|---|---|
| Global (×2) | Game time, team gold difference |
| Per player (×10 × 448) | Level, KDA, CS, gold spent, current gold, items owned (binary), champion one-hot, champion tags, role, base stats, runes |

**Training**
- Loss: reward-weighted cross-entropy
- Reward: win/loss (±1) + 0.3 × normalized gold earned in the next 4 minutes
- Fog of war: enemy player features randomly zeroed (p=0.5) at training and validation time

**Output:** probability distribution over ~200 items → top-K recommendations

**Architectures** (selectable via config file)
- `MLP` — flat feed-forward network, fast and strong baseline (~5M params)
- `Transformer` — per-player token attention + FiLM conditioning on game context (~2.5M params, faster convergence)

---

## Setup

```bash
git clone https://github.com/your-username/lol-item-recommender
cd lol-item-recommender
```

**Install Python 3.12**

Option A — Microsoft Store (Windows, easiest):
```bash
winget install Python.Python.3.12
```
Option B — [Official installer](https://www.python.org/downloads/release/python-3120/) — check **"Add Python to PATH"** during installation.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your [Riot API key](https://developer.riotgames.com):
```bash
cp .env.example .env
```

Download a model checkpoint from Google Drive and place it under `checkpoints/`:
> **[Download checkpoint](https://drive.google.com/drive/folders/141aMKdmWNnapXo9F_5yyllljTKNysuTe?usp=sharing)**

---

## Live usage (in-game)

Start a League of Legends game, then choose your preferred interface:

### Option A — Browser UI (recommended)

```bash
streamlit run app.py
```

This opens a browser window automatically at `http://localhost:8501`. The app polls the live game every 5 seconds and shows the top-5 recommended items with icons and confidence bars. No input needed — everything is read from the running game.

> **First time?** Make sure Streamlit is installed:
> ```bash
> pip install streamlit
> ```

### Option B — Terminal

```bash
python live_inference.py
```

Shows top-5 recommendations in the terminal. Updates automatically when you buy an item. Press `ENTER` to refresh manually.

---

### Selecting a model

Both interfaces accept the same flags:

```bash
# Load the best Transformer checkpoint
streamlit run app.py -- --transformer
python live_inference.py --transformer

# Load the best MLP checkpoint
streamlit run app.py -- --mlp
python live_inference.py --mlp

# Load a specific checkpoint
streamlit run app.py -- --checkpoint checkpoints/transformer/best_model_TRANSFORMER.pt
python live_inference.py --checkpoint checkpoints/transformer/best_model_TRANSFORMER.pt
```

If no flag is given, the best available checkpoint is loaded automatically.

---

## Run offline scenarios

Test the model without being in a game:

```bash
# 3 random scenarios
python inference.py

# Fixed champion + role, 5 different game contexts
python inference.py --champion 54 --role TOP --n 5
```

Evaluate a checkpoint on the validation set:

```bash
python evaluate.py
python evaluate.py --checkpoint checkpoints/transformer/2026-03-28_21-00-00/best_model.pt
```

---

## Train

Training is configured via YAML files in `configs/`. Two presets are included:

```bash
# Transformer (recommended)
python train.py --config configs/transformer.yaml

# MLP baseline
python train.py --config configs/mlp.yaml

# Resume a run
python train.py --config configs/transformer.yaml --resume checkpoints/transformer/2026-03-28_21-00-00/ckpt_step50000.pt
```

Checkpoints are saved to `checkpoints/{config_name}/{datetime}/`. Training is logged to [Weights & Biases](https://wandb.ai).

---

## Retrain from scratch

```bash
# 1. Collect data (requires Riot API key in .env)
python downloader.py --all

# 2. Parse match timelines into purchase events
python parser.py

# 3. Compute normalization statistics
python compute_stats.py

# 4. Preprocess into sharded numpy arrays
python preprocess.py

# 5. Compute random / most-frequent baselines
python baseline.py

# 6. Train
python train.py --config configs/transformer.yaml
```

---

## Pipeline overview

```
Riot API
  ↓ downloader.py
data/timelines/                     ← raw match JSON
data/match_participants.json        ← champion, role, runes per player
  ↓ parser.py
data/dataset/samples.jsonl          ← (X, y, R) per purchase event
  ↓ compute_stats.py + preprocess.py
data/processed/shards/X_NNN.npy … ← sharded numpy arrays (mmap-friendly)
  ↓ train.py --config configs/*.yaml
checkpoints/{config}/{datetime}/best_model.pt
  ↓ app.py / live_inference.py / inference.py
top-K item recommendations
```

---

## Requirements

- Python 3.11+
- PyTorch
- Streamlit (browser UI)
- Weights & Biases (training logs)
- Riot API key (data collection only)
- League of Legends client (live mode)
