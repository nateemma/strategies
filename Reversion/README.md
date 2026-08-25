# Oversold-reversion study (2026-08-24) — VALIDATES STANDALONE, WORTH BUILDING

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

## 4. STANDALONE evaluation (net of 40bp round trip) — this is the operative one

Sections 1-3 judged the signal per-trade and as a blend. Treated as an INDEPENDENT
strategy the tuning changes and the verdict improves. Sweep over hold x threshold x
maxpos, ex-ONE, 14 liquid pairs, 2021-2026:

| hold | thr | maxpos | trades | invested | CAGR | vol | Sharpe | maxDD |
|---|---|---|---|---|---|---|---|---|
| 48 | −20% | 5 | 1553 | 22% | **−5.9%** | 38% | 0.03 | −62% |
| 72 | −30% | 5 | 539 | 13% | +10.2% | 32% | 0.46 | −52% |
| 96 | −40% | 5 | 181 | 5% | +10.1% | 25% | 0.51 | −24% |
| **72** | **−30%** | **12** | 539 | 13% | **+11.0%** | **18%** | **0.66** | **−29%** |

**CORRECTION to section 2: the −20% threshold LOSES MONEY net of costs.** Median
forward return per signal was the wrong objective — it ignores costs and the fact that
you take every signal, not one. **−30% is the viable threshold.**

`maxpos` matters more than expected and monotonically (Sharpe 0.41/0.46/0.54/0.66 for
3/5/8/12; vol 36%→18%, maxDD −57%→−29%). Reversion wants MANY SMALL BETS — the opposite
of the momentum book's three concentrated ones, and a slope rather than a spike.

**Per-window validation of hold=72h / thr=−30% / maxpos=12:**

| window | CAGR | vol | Sharpe | maxDD |
|---|---|---|---|---|
| P1 2021-05..2022-12 | +19.7% | 32% | 0.72 | −29% |
| P2 2023-01..2024-08 | +2.3% | 2% | 1.07 | −1% |
| P3 2024-09..2026-08 | +13.6% | 10% | 1.30 | −5% |

Positive in all three; per-window Sharpes EXCEED the pooled 0.66 (pooling mixes very
different vol regimes). Config was chosen from a 36-cell full-sample sweep, so the
per-window check is the thing that makes it credible — plus the monotonic maxpos slope.

## Verdict

**VALIDATES STANDALONE — worth building properly.** (An earlier verdict here said "not
worth building"; that rested on a 50/50 blend argument which does not apply when the two
are run as independent strategies.) Modest but real: ~+11% CAGR at 18% vol, −29% maxDD,
539 trades over 5.6y, capital deployed only 13% of the time.

Weaker than the momentum book on return, much better on drawdown.

## Caveats
- Conditional forward returns, NOT a strategy: no sizing, no capital constraint, no
  overlapping-signal handling; assumes entry at signal close, exit exactly 48h later.
- Correlation overlap is only 422 days (2025-01 on, essentially P3) — longest momentum
  equity curve available.
- No costs in the correlation sim, so 0.26 is an overstatement (~0.19 at 40bp round trip).
- "Liquid-15" median is ~$4.4k quote volume per HOUR on Binance US — a thin venue.
- Do NOT reuse the momentum whitelist: liquid-only is a design constraint here, not a
  robustness check. `max_open_trades=3` is also likely wrong — reversion wants many small bets.
