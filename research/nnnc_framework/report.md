# NNNC framework study — report (2026-10-13)

Pre-registration: `f879338` ([preregistration.md](preregistration.md)). Selection rule: `5984344`.
Judge: `judge.py` → [verdicts.json](verdicts.json). **PROVISIONAL** (2023–26 halves only).

## Result: one lever adopted — the 24 h grace window

Cells = verdict, early / late gain in pp vs the model's 030-baseline control (sum of per-fold return).

| key | lever | ARO (arm D) | GBR | GFM | MLX | decision |
|---|---|---|---|---|---|---|
| GR | exit:grace_hours = 24.0 | PASS +5.56 / +1.69 | PASS +8.89 / +0.92 | FAIL +15.40 / -1.25 | PASS +4.11 / +5.75 | **ADOPTED** |
| TH | entry:thr_shift = -0.05 | FAIL +8.13 / -18.80 | FAIL +1.39 / -25.21 | FAIL +12.22 / -22.44 | FAIL -0.18 / +0.22 | **NOT ADOPTED** |
| TP | exit:take_profit = 0.04 | FAIL +11.41 / -4.00 | FAIL +3.24 / -5.36 | FAIL +4.05 / -2.59 | PASS +2.39 / +0.38 | **NOT ADOPTED** |
| TK | exit:tp_atr_k = 4.5 | FAIL +8.06 / -1.09 | FAIL -1.97 / -3.79 | FAIL +2.35 / -3.24 | PASS +2.85 / +0.08 | **NOT ADOPTED** |
| SS | exit:stop_scale = 2.0 | PASS +3.13 / +2.29 | FAIL +2.33 / -0.73 | FAIL +1.11 / -2.25 | PASS +0.35 / +0.49 | **NOT ADOPTED** |
| U1 | exit:unclog_1d_hours = 12.0 | FAIL +0.37 / -0.20 | PASS +0.27 / +0.72 | PASS +0.13 / +0.07 | PASS +0.09 / +0.35 | **NOT ADOPTED** |

Only GR is ADOPTED (ARO + 2 of 3), so no combination is run (the rule needed ≥ 2 adopted).

**GR (grace 6.5 h → 24 h, the −27% grace stop held for a day):**
- 15-fold totals: ARO 42.71 → 49.97, GBR 43.65 → 53.46, GFM 25.61 → 39.76, MLX 25.32 → 35.19.
- Every paired verdict UNCHANGED (p 0.41 / 0.32 / 0.05 / 0.08), risk (Sharpe, maxDD) UNCHANGED
  on all four, both halves UNCHANGED. Positive on all four models in total; GFM fails only on its late
  half (−1.25).
- Mechanism (ARO exits, arm vs control): post-grace stops 469 → 199; unclog_12h 84 → 212 and
  unclog_1d 1 → 27 — trades that used to be stopped in hours 6.5–24 recover to a small profit;
  take-profit 824 → 694; trades 1,420 → 1,176 (longer occupancy). This is 027's finding (45–63% of
  stop exits later reach the target) confirmed on model entries, and 027's exploratory C4 arm
  (12 h, reversion SimpleTrigger) pointing the same way.
- MLX was not screened for exits (replay fidelity 87.8%) yet passes both halves.
- 24 h is the edge of the screened grid: this tests the value, not the optimum.

**Not adopted:**
- **TH (threshold −0.05)** reverses hard out of sample on the three guards-open models (late −19 to −25 pp):
  the extra low-confidence signals helped in 2023–24 and lose in 2025–26. Model probabilities are
  tightly bunched (arm D median 0.735, max 0.81), so small threshold moves change trade count a lot.
- **TP (0.04) and TK (ATR k 4.5)**: wider take-profits help early, lose late on ARO/GBR/GFM; pass on MLX only.
- **SS (post-grace stop ×2)**: passes ARO and MLX, fails GBR/GFM late (small).
- **U1 (unclog_1d 12 h)**: ±1 pp everywhere; fails ARO late by −0.20.

## Screen vs arm (reported only)

Late-half screen deltas (computed after the pre-registration commit), for comparison with the arms'
late gains above: {"exit:grace_hours": {"ARO": {"early": 19.54, "late": 9.17}, "GBR": {"early": 16.594, "late": 3.794}, "GFM": {"early": 20.233, "late": -1.772}}, "entry:thr_shift": {"ARO": {"early": 8.116, "late": -17.644}, "GBR": {"early": 1.165, "late": -25.483}, "GFM": {"early": 12.058, "late": -22.763}, "MLX": {"early": -0.181, "late": 0.218}}, "exit:take_profit": {"ARO": {"early": 9.243, "late": -4.387}, "GBR": {"early": 3.544, "late": -3.644}, "GFM": {"early": 4.206, "late": -2.761}}, "exit:tp_atr_k": {"ARO": {"early": 6.846, "late": -1.442}, "GBR": {"early": 1.698, "late": -2.317}, "GFM": {"early": 3.156, "late": -2.551}}, "exit:stop_scale": {"ARO": {"early": 4.201, "late": 1.313}, "GBR": {"early": 1.783, "late": 2.191}, "GFM": {"early": 2.15, "late": -1.143}}, "exit:unclog_1d_hours": {"ARO": {"early": 0.504, "late": -0.173}, "GBR": {"early": 0.276, "late": 0.792}, "GFM": {"early": 0.133, "late": 0.074}}}

The screens (replay / simulator, which ignore re-sequencing) predicted the sign of 18 of the 19
late-half arm results they cover (MLX exits unscreened); the one miss is SS on GBR (screen +2.19, arm
−0.73). Magnitudes match closely for entry and take-profit levers; for GR the arm keeps only ~0.2–0.7×
of the screen's late gain (longer occupancy re-sequences entries, which the replay cannot see). The
screens are a usable filter for model-entry framework changes.

## Stage 1 findings worth keeping (early half, descriptive)

- Entry guards (atr, close_norm, adx, bb_width, rvol at MLX's values) all cost money on the guards-open
  models; the soft gate (raise the threshold in volreg off-state) does not help (ARO −0.7 to −1.0).
- max_hold (2–10 d) and metric_overbought are inert: every arm D trade ends within 24 h.
- ROI changes, tighter grace levels (−10% … −20%) and shorter grace (3 h) all lose.

## Tools added

`diagnose.py` (exit levers, 027 replay + the .json operating point), `signals.py` (per-bar test-window
signal tables, strict model-key mode), `entries.py` (entry simulator, ≥ 99.8% of entries reproduced),
`select.py`, `build_arms.py`, `judge.py`, `late_screen.py`. Fix: 027's loader now handles a force-exit
stamped past the end of the data.
