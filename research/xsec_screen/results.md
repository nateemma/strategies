# Cross-sectional rank-IC screen -- results (2026-10-21)

Pre-registration: `preregistration.json` (d4a20283). Raw numbers: `results.json`. 10 wf pairs, binanceus 15m,
early 2023-01-01..2024-12-21, late 2024-12-21..2026-09-07. No training, CPU only.

## Verdict: 0 / 30 PASS

No signal has its declared sign in both halves with pooled |t| >= 3.

- **Momentum (declared +) is REVERSAL on these pairs.** `mom_1d@1d` IC -0.048 / -0.055 (t -3.41 / -3.58,
  pooled -4.94); `mom_3d@1d` pooled t -3.92; `resmom_7d@1d` early t -3.86 (late -1.11). Opposite to the declared
  sign, so FAIL by rule -- and the most persistent effect in the panel.
- **Illiquidity (Amihud, +):** nothing (|t| <= 1.85, wrong sign).
- **Realized variance / upside semivariance (declared -):** weakly POSITIVE (pooled t +2.1 / +2.3 at 1d), wrong
  sign, below the bar. **Signed jump:** flips sign between halves.
- **Baseline `mom_90d`** (the live basket's ranking): IC ~ 0 at every horizon on these 10 pairs (the basket
  trades a broad universe; this says nothing about the basket itself).

## Exploratory (post hoc, NOT pre-registered): is the reversal tradeable?

Long the 3 worst 1-day performers, equal weight, rebalanced daily, 30 bp round trip, vs equal weight all 10:

| book | early gross | early net | late gross | late net |
|---|---|---|---|---|
| reversal mom_1d@1d | +35.7 %/yr | -42.2 %/yr | +46.4 %/yr | -32.1 %/yr |
| reversal mom_3d@3d | -1.6 | -28.9 | +16.9 | -9.3 |

The 1-day reversal is real gross in both halves but daily rebalancing costs ~78 %/yr; it does not survive as
a standalone book. It is the same mean-reversion NNNC's gbb trigger already harvests more selectively.

## Open lead (needs its own pre-registration)

Use the 1-day cross-sectional return rank as a **selector/sizer among gbb candidates** (no extra turnover:
it only re-orders trades NNNC already takes). Cheapest test: the 412 early width-book trades
(research.nnmt.screen / research.nnpredict.screen metrics), rank key = -mom_1d at the signal bar.
