# BTC-Behaviour Study — Final Report

**Question:** Does BTC price action carry predictive information for altcoin returns, and can it improve trading signals?
**Data:** Binance.US OHLCV, 1h base (BTC + 12 alts, ~2.2yr, 2024-04 → 2026-07), 4h cross-check. Log returns throughout. All features/regimes strictly causal (data ≤ t); models judged out-of-sample.

---

## TL;DR

1. **BTC does not *lead* alts at hourly resolution — it moves *with* them.** Correlation peaks at lag 0 (0.24–0.79) for every alt and collapses to ~0 by lag 1h. Any true lead is sub-hourly. **The naive "BTC leads → trade the lag" premise is falsified.**
2. **BTC features add a small, consistent predictive lift** (OOS IC 0.040 → 0.052, +0.012, 9/12 alts), but via BTC's **volatility/trend regime**, not its returns. Directional accuracy barely moves (≈50%). Real but marginal.
3. **The beta-spread mean-reverts robustly** (monotonic conditional returns, 12/12 positive gross), but net of costs the edge survives only on **low-beta idiosyncratic coins** (BCH, ZEC, AAVE, NEAR).
4. **The single strongest pattern:** the **least** BTC-correlated coins are the **most** forecastable. High-beta mirrors (ETH/SOL/AVAX) efficiently track BTC and carry ~zero idiosyncratic edge; the idiosyncratic names carry tradeable structure.
5. **Regime matters and it's counter-intuitive:** alts pay best in **Weak Up** (+2.2 bp/bar, 53% win), *not* Strong Up; **Sideways is a trap** (−1.5 bp, worst drawdown). The market-neutral spread edge is regime-robust and strongest exactly where directional longs fail (down/sideways) — the two dovetail.

---

## Experiment 1 — Lead-lag correlation
- Peak correlation at **lag 0 for all pairs, both timeframes, 100% of rolling 90-day windows** (peak-lag std = 0 → perfectly stable).
- Decay is immediate: e.g. SOL lag0 **0.747** → lag1 **−0.016**; ETH 0.791 → 0.004.
- Contemporaneous corr (co-movement) ranking: ETH 0.79 > SOL 0.75 > LINK 0.68 > AVAX 0.66 > XRP 0.62 > SUI 0.58 > LTC 0.56 > DOT 0.54 > NEAR 0.52 > AAVE 0.46 > BCH 0.39 > ZEC 0.24.
- **Verdict:** relationship is strong and stable (success criteria met) but **coincident, not predictive** — no exploitable lead-lag.

## Experiment 2 — BTC predictive features (Model A alt-only vs Model B +BTC)
- OOS walk-forward (3 folds, LightGBM), 4h-forward return target.
- Mean IC **0.040 → 0.052** (ΔIC **+0.012**, **9/12 improved**); directional acc 49.9% → 50.3%; **R² < 0 for all** (no magnitude skill; edge is rank-only and thin).
- Feature importance: top signals are **volatility & trend strength** (`alt_vol24`, `alt_atrpct`, `btc_vol24` #3, `btc_adx` #4, `alt_adx`) — the `*_ret_*h` features rank low. **BTC = 44.7% of total importance**, entirely via regime state, not returns.
- **Verdict:** BTC features *materially improve* only in the weak sense — consistent but small. Confirms Exp 1: the edge lives in regime state, not return lead.

## Experiment 3 — Relative-strength spread reversion
- Beta = rolling 30d OLS (causal); spread = z-scored trailing-24h idiosyncratic residual `e = alt_ret − β·btc_ret`.
- `corr(spread, forward idiosyncratic return)` **negative for all 12 alts at all horizons** (mean −0.058). Pooled conditional table is **monotonic**: Q0 (most underperformed) **+11.9 bp** fwd-8h → Q4 (most outperformed) **−8.7 bp**.
- Cost stress (per-unit turnover, ×2 legs for the BTC hedge): gross 12/12 positive → **2bp: 11/12 → 5bp: 7/12 → 10bp: 4/12**.
- **Survivors at 10bp:** BCH (+2.2 bp, Sharpe@5bp **6.3**), ZEC (+2.7), NEAR (+0.8), AAVE (+0.6).
- **Verdict:** mean reversion produces positive expectancy (criterion met gross/low-cost), but net-tradeable only on the high-idiosyncratic-vol names. *Caveat:* those are the least liquid, so the 5bp assumption is most optimistic exactly where the edge concentrates. Sharpe is autocorrelation-inflated and full-sample.

## Experiment 4 — Regime detection
- BTC regime = ADX strength (<20 sideways, 20–30 weak, >30 strong) × EMA-slope direction.
- Passive alt basket by regime: **Weak Up +2.21 bp/bar, 53.1% win, shallowest DD** (best); **Sideways −1.52 bp** (worst, 30% of time); **Strong Up only +0.12 bp** (late-cycle disappointment).
- Spread-reversion net@5bp is **positive in every regime** (0.46–1.29 bp), strongest in Weak/Strong Down & Sideways — i.e. where directional longs bleed.
- **Verdict:** certain regimes clearly outperform (criterion met). Directional and market-neutral books are complementary across regimes.

---

## Master table — per-altcoin synthesis (ranked by OOS predictive power = Exp-2 IC)

| # | alt | lag0 corr | OOS IC_B | ΔIC (BTC lift) | spread net@5bp | spread net@10bp | Sharpe@5bp | best regime (bp) |
|--:|---|--:|--:|--:|--:|--:|--:|---|
| 1 | **BCH** | 0.39 | **0.158** | +0.045 | +3.76 | **+2.19** | **6.30** | Weak Up (1.1) |
| 2 | **AAVE** | 0.46 | **0.114** | +0.033 | +1.94 | **+0.62** | 3.09 | Weak Up (1.9) |
| 3 | **NEAR** | 0.52 | **0.111** | +0.019 | +2.21 | **+0.76** | 3.58 | Weak Up (0.6) |
| 4 | **ZEC** | 0.24 | **0.101** | +0.002 | +3.81 | **+2.66** | 3.23 | Weak Up (4.0) |
| 5 | DOT | 0.54 | 0.090 | +0.013 | +1.03 | −0.35 | 1.95 | Weak Up (1.3) |
| 6 | LTC | 0.56 | 0.039 | +0.008 | +0.44 | −0.89 | 0.95 | Weak Up (2.1) |
| 7 | ETH | 0.79 | 0.018 | +0.022 | −0.61 | −1.71 | −2.27 | Weak Up (2.6) |
| 8 | SUI | 0.58 | 0.017 | −0.007 | +0.23 | −1.02 | 0.41 | Weak Up (3.3) |
| 9 | SOL | 0.75 | 0.013 | +0.004 | −0.40 | −1.57 | −1.15 | Weak Up (2.0) |
| 10 | XRP | 0.62 | −0.009 | −0.001 | −0.91 | −1.99 | −1.88 | Weak Up (2.7) |
| 11 | LINK | 0.68 | −0.013 | +0.005 | −0.10 | −1.37 | −0.24 | Weak Up (3.0) |
| 12 | AVAX | 0.66 | −0.016 | −0.004 | −0.04 | −1.26 | −0.09 | Weak Up (3.0) |

**Tradeable shortlist** (OOS IC > 0.05 **and** spread survives 10bp): **BCH, AAVE, NEAR, ZEC.**

Note the inverse relationship down the table: high `lag0_corr` (BTC-mirrors, rows 7–12) ↔ near-zero/negative IC and cost-losing spreads; low `lag0_corr` (rows 1–4) ↔ the strongest edge.

---

## Recommended strategy synthesis
1. **Market-neutral spread reversion on {BCH, ZEC, AAVE, NEAR}** — beta-hedge vs BTC, contrarian to the 24h idiosyncratic z-score. The only book that survives realistic costs; regime-robust; strongest in down/sideways BTC.
2. **Directional alt exposure gated to Weak-Up BTC regime** — flat in Sideways (the trap) and skeptical of Strong Up (late-cycle). Works for the mirrors (ETH/SOL/AVAX), whose *only* edge is regime timing.
3. **Use BTC as a regime/hedge reference, not a return-lead predictor** — its value is `btc_vol24`/`btc_adx` state and its role as the hedge leg, not its lagged return.

## Caveats / limitations
- Exp 3–4 stats are full-sample (in-sample); only Exp 2 is walk-forward OOS. The shortlist agrees across both, which is reassuring, but the spread Sharpes are upper bounds (autocorrelation-inflated, single sample, optimistic cost on illiquid names).
- Directional accuracy ≈ coin-flip everywhere; all edges are thin rank/tail effects, not sign-calling.
- Regime thresholds (ADX 20/30) and horizons (4h target) are unoptimized defaults.
- No slippage/borrow/funding modeled beyond the flat per-turn cost; the BTC short leg assumes it's executable.

## Suggested next steps
- Walk-forward the spread strategy (Exp 3) and regime gating (Exp 4) to confirm OOS, with per-name realistic cost/liquidity.
- Test a cross-sectional version: rank all 12 alts by spread z each bar, long the bottom / short the top (dollar-neutral) — likely cheaper and more robust than per-name.
- Combine: regime-gated directional book + market-neutral spread book, sized by their (low) mutual correlation.
