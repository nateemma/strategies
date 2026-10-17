# M-of-N trigger (037 4-of-8 vote): report

**Sources:** `preregistration.json` (commit af8708bd, before any backtest); `verdicts.json`;
`instrument.json`. Run 2026-10-17.
**Label: PROVISIONAL**; the vote was chosen and confirmed with the late half in view. Nothing is adopted.

## Decision: not a candidate (both verdicts FAIL)

| arm | trades (early / late) | alone pp (gbb 11.04) | vs gbb early / late | paired | risk | beside width book: total / early / late (w_aro 51.44 / 35.85 / 15.59) | paired (whole) |
|---|---|---|---|---|---|---|---|
| `st_mofn48` | 941 (553 / 388) | −15.28 | −32.43 / +6.10 → FAIL | INCONCLUSIVE_ARM_LOSES (p 0.15; early p 0.011) | Sharpe WORSENED (p 0.043), maxDD UNCHANGED | 36.14 / 33.90 / 2.24 → FAIL | WORSENED (−15.3 pp, p 0.013) |

- Instrument PASS: causality PASS (all triggers re-run, control FAILs, no other verdict changed);
  every member equals the 037 harvest's raw_buy column on all 11 pairs (0 mismatches).
- Exits: 30% of the arm's trades reach the take-profit (gbb 42%) and 25% leave on the 12 h unclog
  (gbb 16%). The vote fires on calm bars: the screen favoured it only on the volatility-neutral
  label (tbatr), and on the fixed-% label the ARM_D_FAMILY exits pay (tb) it was "no" (lift 1.16 /
  1.17 vs width 1.62 / 1.62). The pre-registration flagged this; the result matches it.
- Beside the width book, 89 trades held a pair the width book also held.
