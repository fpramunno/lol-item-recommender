# Model Summary

## Input — vettore X (4112 feature)

### Globali (2)
| Feature | Descrizione |
|---------|-------------|
| game_time_min | Minuto di gioco, normalizzato [-1, 1] |
| team_gold_diff | Differenza gold totale tra le due squadre, normalizzato [-1, 1] |

### Per ognuno dei 10 player (× 10 = 4110)
| Feature | Dims | Descrizione |
|---------|------|-------------|
| Numeric stats | 8 | level, kills, deaths, assists, cs, gold_spent, gold_current, is_buyer — normalizzati [-1, 1] |
| Items owned | 207 | Binary vector: 1 se possiede quell'item, 0 altrimenti |
| Champion ID | ~172 | One-hot vector |
| Champion tags | 6 | Binary: Fighter, Tank, Mage, Assassin, Marksman, Support |
| Role | 5 | One-hot: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY |
| Champion base stats | 13 | HP, armor, MR, range, ecc. da Data Dragon — normalizzati [-1, 1] |

> `is_buyer = 1` per il player per cui si vuole la raccomandazione, 0 per tutti gli altri.
> `gold_current` è valorizzato solo per il buyer, 0 per tutti gli altri.

---

## Modello — MLP

```
Input (4112)
    → Linear(4112, 1024) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(1024, 512)  → BatchNorm → ReLU → Dropout(0.3)
    → Linear(512, 256)   → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256, 207)   → logits
```

**Training:** reward-weighted cross-entropy
```
loss = mean( weight * CE(logits, y) )
weight = clip( (R - R_min) / (R_max - R_min), 0, 1 )
R = r_win + 0.3 * delta_gold_norm
```
- `r_win` = +1 se la partita è vinta, -1 se persa
- `delta_gold_norm` = gold guadagnato nei 4 minuti successivi all'acquisto, normalizzato per il gold rate medio

**Fog of war masking:** durante il training, le feature dei 5 player nemici vengono azzerate con probabilità 0.5 per ogni sample, rendendo il modello robusto a informazioni parziali.

---

## Output — vettore di logits (207)

Un punteggio per ognuno dei 207 item nel dataset. Dopo softmax diventa una distribuzione di probabilità.

A inference time si prendono i top-K item con probabilità più alta.
