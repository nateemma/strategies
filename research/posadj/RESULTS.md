# Repeat-signal position adjustment on NNNC_MLX — **REJECTED**

**Date**: 2026-09-15 · **Origin**: `docs/WALKFORWARD_TODO.md` item 3
**Code**: `NNNC/_position_adjust.py`, `research/posadj/arms.py` · **Data**: `research/posadj/out/arms.json`

## Why this was run

Fold f08 measured that freqtrade's one-trade-per-pair rule silently discards **~43% of entry
signals** (≥54 of ~125). `--enable-position-stacking` showed the size of the loss but is
backtest-only and not reproducible in dry/live. `position_adjustment_enable` +
`adjust_trade_position` is the **deployable** form: on a repeat `enter_long`, add one fixed
stake to the open position, capped at 2 adds, minimum 4 bars apart.

## Verdict: REJECT

| | |
|---|---|
| **Return** | **NOT RESOLVED** — −0.455 pp/yr, 7/13 live folds improved, sign test p = 1.0. \|Δ\| is far below the 7.633 pp noise floor, so this is "no effect detected", not "no effect". |
| **Drawdown** | **RELIABLY WORSE** — balance drawdown up in **12 of 13** live folds, sign test **p = 0.0017**, Wilcoxon **p = 0.00024**, mean +1.58 pp, worst +8.77 pp (f08). |

A deployment modification is supposed to buy drawdown at flat return. This does the opposite:
flat return, reliably worse drawdown. There is no operating point in this design worth having.

Liveness (Principle IV): **83 adds across 15 folds**, 13 of 15 folds live; f00 and f14 saw zero
adds and are excluded as inert rather than counted as ties.

## The mechanism, measured rather than asserted

Pooled across the arms, trades grouped by how many adds they received:

| adds | n | win rate | mean `profit_ratio` | worst |
|---|---|---|---|---|
| 0 | 302 | **0.791** | +0.01571 | −0.1365 |
| 1 | 55 | 0.709 | +0.00593 | −0.1113 |
| 2 | 14 | **0.286** | **−0.03898** | **−0.2713** |

**Monotonic in the wrong direction.** Win rate falls 79% → 71% → 29%; mean profit ratio goes
from clearly positive to clearly negative; the worst loss doubles.

Adds also concentrate on trades that stop out: **38%** of added-to trades exit by
`trailing_stop_loss` (26/69) versus **21%** of the rest (63/302).

So the repeat signal is **not a random resample of entry quality — it is conditionally worse.**
A second `enter_long` while a position is open arrives disproportionately when the first entry
is going against you, and adding there does two harmful things at once: it grows the loser, and
because the ATR-derived stop is computed off the position's average entry price, averaging down
**moves the stop down with it**, giving the trade more room to keep losing.

That is the difference between this and `--enable-position-stacking`, and it is why the `--eps`
upper bound never transferred: stacking creates N independent trades each with **its own** entry
and **its own** stop; position adjustment merges them into one trade whose stop drifts away from
the original thesis.

## What this closes and what it does not

**Closed.** Unconditional repeat-signal adds on NNNC_MLX. Do not re-propose.

**Not closed — and now better motivated.** The masked-signal finding stands: ~43% of entry
signals are discarded, and this experiment recovered only 83 adds across 15 folds, a small
fraction of them. What it establishes is that the masked signals are **not free money waiting to
be collected** — at least the subset arriving during an open position is adverse-selected.

Two variants remain untested, each a separate parameter space (Principle VIII):
1. **Add only when the position is in profit** (pyramiding). The table above predicts this is
   where any value would be, since the harm concentrates in adds made on trades heading for a
   stop.
2. **Independent concurrent trades per pair** with separate stops — the thing `--eps` actually
   simulated. Not expressible in freqtrade's live engine, so it would need a different execution
   model, not a strategy change.

**Meta-lesson for this repository.** An upper bound measured with a *different mechanism* is not
an upper bound on the mechanism you can deploy. `--eps` showed +1.27pp on f08; the deployable
form delivered −6.31pp on the same fold. The two differ in exactly one structural way — shared
versus independent stops — and that one difference reversed the sign.

## Process note

The first liveness probe used `nr_of_successful_entries`, which is **absent from exported
backtest trade dicts**, so `.get(key, 1)` reported **zero adds on a working arm**. It was caught
only because an "inert" arm had moved P&L by 6 pp. Add counts are now derived from
`stake_amount` in base-stake units, which is present and correct.
