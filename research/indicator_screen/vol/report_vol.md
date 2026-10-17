# 037 re-run on tradeable bars (volume > 0)

Pre-registration `preregistration_vol.json` (commit 8d45973b): the only change is that every row
needs volume > 0 (32% of the original data_ok bars were zero-volume). Detail: `report_tb.md`,
`report_tbatr.md`, `mofn.json`, `single_labels.json`, `pairs_*.json` in this folder. SCREEN only.

| | original (all data_ok bars) | tradeable bars only |
|---|---|---|
| label:buy, candidate bars: top single columns | atr_pct_roll 0.594, kc_upper/close 0.593 | kc_upper/close 0.597, atr_pct 0.594 (same set) |
| instrument check | PASS | PASS |
| tbatr: oscillators (WILLR, %K, CMO, RSI), AUC early / late / E0 | ~0.445 / 0.44 / 0.47 | ~0.473 / 0.48 / 0.485 (about half the strength) |
| tbatr: width book lift E0 / early / late | 0.96 / 1.16 / 1.15 | 0.96 / 1.09 / 1.11 |
| tbatr: confirmed pairs (all / candidate bars) | 14 / 6 (mostly on zero-volume bars) | 4 / 3, lifts 1.2–1.5 |
| tb (fixed %): width book lift E0 / early / late | 1.41 / 1.69 / 1.67 | 1.36 / 1.62 / 1.62 |
| tb: confirmed pairs (all / candidate) | 19 / 13 | 19 / 13 (KST∧ZScore 1.94 / 1.88 / 1.44) |
| ARO p_buy AUC on tbatr, all bars (early / late) | 0.530 / 0.514 | 0.518 / 0.497 |
| M-of-N on tbatr, all bars | 3-of-8: no | 4-of-8: **yes** (1.18 / 1.13 / 1.19 vs width 1.09 / 1.11 / 0.96; fires 0.4%) |

- The gbb_path label screen and the fixed-% triple-barrier screen barely change: their results
  were mostly on traded bars.
- The volatility-neutral (tbatr) results shrink: the oscillator edge halves, and 14 confirmed
  pairs become 4. The zero-volume bars were inflating them.
- ARO has essentially no directional skill on traded bars (AUC ~0.50–0.52 on all bars).
