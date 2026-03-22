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

A machine learning model that recommends the next best item to buy in League of Legends, based on the current game state (champion, role, KDA, CS, gold, items owned, team composition, runes).

Works in real time via the Riot Live Client Data API — no manual input required.

---

## How it works

The model is an MLP trained on ~450k item purchase events from high-elo EUW matches (Challenger/Grandmaster/Master). For each purchase, the model receives a snapshot of the full game state and predicts which item was bought.

**Input:** 4112 features
- Global: game time, team gold difference
- Per player (×10): level, KDA, CS, gold spent, current gold, items owned (binary), champion one-hot, champion tags, role, champion base stats

**Training:** reward-weighted cross-entropy
- Reward = win/loss (±1) + 0.3 × normalized gold earned in the next 4 minutes
- Fog of war masking: enemy features are randomly zeroed (p=0.5) during training

**Output:** probability distribution over 207 items → top-K recommendations

See [`model_summary.md`](model_summary.md) for full details.

---

## Setup

```bash
git clone https://github.com/your-username/lol-item-recommender
cd lol-item-recommender
```

**Install Python 3.12**

Option A — Microsoft Store (Windows, easiest):
```
winget install Python.Python.3.12
```

Option B — Official installer: https://www.python.org/downloads/release/python-3120/
> Check **"Add Python to PATH"** during installation.

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

Download the model checkpoint from Google Drive and place it in `checkpoints/best_model.pt`:
> **[Download checkpoint](https://drive.google.com/file/d/191ur4DDl3m7wdojLIOvzIHJvwtyUbFDK/view?usp=drive_link)**

---

## Live usage (in-game)

Start a League of Legends game, then run:

```bash
python live_inference.py
```

The tool automatically detects item purchases and shows the top-5 recommendations. Press `ENTER` to refresh manually.

---

## Run test scenarios

```bash
# 3 random scenarios
python inference.py

# Fixed champion + role, 5 different game contexts
python inference.py --champion 54 --role TOP --n 5
```

---

## Retrain from scratch

```bash
# 1. Collect data (requires Riot API key)
python downloader.py --all

# 2. Parse timelines into samples
python parser.py

# 3. Compute normalization stats
python compute_stats.py

# 4. Preprocess into numpy arrays
python preprocess.py

# 5. Compute baselines
python baseline.py

# 6. Train
python train.py
```

---

## Pipeline overview

```
Riot API
  ↓ downloader.py
data/timelines/        ← raw match JSON
data/match_participants.json  ← champion, role, runes per player
  ↓ parser.py
data/dataset/samples.jsonl    ← (X, y, R) per purchase event
  ↓ compute_stats.py + preprocess.py
data/processed/X.npy, y.npy, R.npy
  ↓ train.py
checkpoints/best_model.pt
  ↓ inference.py / live_inference.py
top-K item recommendations
```

---

## Requirements

- Python 3.11+
- PyTorch
- Riot API key (personal key recommended for data collection)
- League of Legends client (for live mode)
