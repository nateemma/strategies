# Pre-registration: NNNC framework study, stage 2 (2026-10-13)

**Committed before any stage-2 arm backtest.** Machine-readable copy: [preregistration.json](preregistration.json);
`judge.py` reads only that file and refuses if it is dirty or any arm record predates its commit.

**Question:** which entry, exit and stop-loss settings improve NNNC's *model* trades — measured on
the models themselves, not carried over from SimpleTrigger (the switch experiment showed carry-over
is unreliable).

## How the candidates were chosen (stage 1, no backtests)

- `diagnose.py` replays every control trade of the four models (arm D/ARO, GBR, GFM, MLX) under one
  exit/stop lever at a time with 027's faithful replay, now fed the strategies' `.json` operating point
  (null replay matches 99.9% of recorded exits on ARO/GBR/GFM; MLX 87.8% — its 45 model-exit-signal
  trades are the only misses — so MLX was not screened for exits).
- `signals.py` + `entries.py` rebuild each model's per-bar signals on the test windows and simulate
  entries (≥ 99.8% of recorded entries reproduced, both ways, on all four models) to price entry levers.
- Both screens report the **early half (f00–f07) only**. `select.py`, committed (5984344) before any
  lever result was read, chose per lever the value with the largest summed early gain among those with
  ARO ≥ +0.5 pp and ≥ 2 replications > 0; at most six levers.

## Candidates (one lever each, on all four models)

| key | lever | value | early screen sum (pp) |
|---|---|---|---|
| GR | grace window (`stoploss_grace_hours`) | 6.5 h → 24 h | +56.37 |
| TH | prediction threshold | fold threshold − 0.05 | +21.16 |
| TP | fixed take-profit (`cexit_take_profit`) | 0.022 → 0.04 | +16.99 |
| TK | ATR take-profit (`take_profit_atr_mult`) | off → 4.5 | +11.70 |
| SS | post-grace ATR stop (K, floor, cap) | × 2.0 | +8.13 |
| U1 | `unclog_1d_hours` | 24 → 12 | +0.91 |

Arms `fs_<m>_<key>`, classes `NNNC_FS_<M>_<KEY>`, each on the base model's own per-fold weights
(model key; verified identical to the base's key for all 24) against that model's 030-baseline control.

## Rule (fixed here; same as the switch experiment)

Per model PASS = early gain > 0 AND late gain > 0 AND the 15-fold paired verdict and both risk
verdicts (Sharpe, maxDD) not WORSENED; INCONCLUSIVE if the control has < 200 trades, the arm changed
no fold, or a verdict is ABSORBED/UNRESOLVABLE. **ADOPT** = PASS on ARO and on ≥ 2 of GBR, GFM, MLX.
Early gain is not independent (it chose the value); the **late half is the out-of-sample test**.

If ≥ 2 candidates are ADOPTED, a combination of all adopted levers is run and judged the same way
(TP and TK conflict; TP would be kept).

**Notes.** GR 24 h, TP 0.04, TK 4.5 and SS 2.0 sit at the edges of their grids, so this tests the
chosen value, not the optimum. TH adds trades. TP and TK act on the same exit.
