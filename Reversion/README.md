# Oversold-reversion study (2026-08-24) — SIGNAL REAL, PORTFOLIO WEAK

The "opposite end of momentum" question: is there a rebound/reversion analogue to
`Basket/MomentumRegimeBasket15m`? A cross-sectional version was already built and
REJECTED (`study1_xsec` / `study1b_validate`: ZEC was ~92% of the edge, liquid-only lost
−300% at 10bp). This is the **different** signal that Study 2 measured but never built:
**absolute** oversold, which decays slowly instead of in hours.

Tools: `tools/oversold_reversion_gate.py` (decay curve + selection), `tools/reversion_correlation_gate.py`.

## 1. Decay curve — the signal is slow, and NOT an illiquidity artifact

Mean forward return after RSI(14) crosses below 30, 1h bars, 2021→2026:

| horizon | all 75 pairs | liquid-15 |
|---|---|---|
| 4h  | +64.1 bp | +67.6 bp |
| 24h | +98.6    | +91.0    |
| 48h | +105.2   | **+109.6** |
| 72h | +126.1   | **+138.7** |
| 96h | +109.0   | +115.9   |

Rises through 72h (confirms Study 2 on 5y of data) and **liquid-only is as good or better**
— the opposite of the cross-sectional version's failure mode.

## 2. Selection — one filter dominates

Quintiles of *distance below SMA50* at the signal (median fwd-48h, bp):
`< −20%: +136` | `−20..−12%: +46` | `−12..−6%: +24` | `−6..+2%: +20` | `> +2%: 0`

RSI depth separates far less (+44 vs +27). **Dislocation from trend beats depth of oversold.**

**Filter: RSI<30 cross AND >20% below SMA50** — 20% of signals, ~327/yr over 15 pairs.
Ex-ONE/USDT (70% of the *sum* but only +15bp median — its share is pure tail):

| | n | mean | median | win |
|---|---|---|---|---|
| all | 1,553 | +147 bp | **+154 bp** | 59% |
| P1 2021-05..2022-12 | 840 | | +81 bp | |
| P2 2023-01..2024-08 | 167 | | +256 bp | |
| P3 2024-09..2026-08 | 546 | | +211 bp | |

Every window clears a 40bp round trip. **Dropping the dominant pair IMPROVES the median** —
the inverse of the cross-sectional study, where dropping ZEC collapsed it.

## 3. Correlation gate vs the momentum book — PASSES, but

```
corr(daily) +0.042   corr(weekly) +0.047   corr on days BOTH active (n=75) +0.354
momentum  Sharpe(ann) 1.45     reversion 0.26 (PRE-COST)     50/50 blend 1.21
```

Orthogonal, so not momentum in disguise. **But the standalone Sharpe is too weak to earn a
co-equal allocation — a 50/50 blend REDUCES combined Sharpe from 1.45 to 1.21.** And the
near-zero unconditional correlation is largely NON-OVERLAP (momentum in cash during
risk-off, reversion firing in drawdowns); on days both are live it is +0.354, so the
diversification thins exactly when both books are exposed.

## Verdict

**Signal real and persistent; this portfolio implementation is not worth building as-is.**
The gap between "+154bp median per signal across three regimes" and "Sharpe 0.26" is
portfolio construction, not signal quality: invested only 22% of hours, equal-weight across
~3.8 concurrent positions, holding high-vol beaten-down coins (2.73%/day).

If revisited, the work is in harvesting, not in finding: concentrate on the best signals
rather than taking all ~327/yr, size by conviction, and re-measure with costs and
liquidity-capped fills on the Basket infrastructure.

## Caveats
- Conditional forward returns, NOT a strategy: no sizing, no capital constraint, no
  overlapping-signal handling; assumes entry at signal close, exit exactly 48h later.
- Correlation overlap is only 422 days (2025-01 on, essentially P3) — longest momentum
  equity curve available.
- No costs in the correlation sim, so 0.26 is an overstatement (~0.19 at 40bp round trip).
- "Liquid-15" median is ~$4.4k quote volume per HOUR on Binance US — a thin venue.
- Do NOT reuse the momentum whitelist: liquid-only is a design constraint here, not a
  robustness check. `max_open_trades=3` is also likely wrong — reversion wants many small bets.
