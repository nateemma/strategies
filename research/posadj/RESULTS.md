# Repeat-signal position adjustment on NNNC_MLX

| variant | return | drawdown | verdict |
|---|---|---|---|
| **A** unconditional adds | −0.395 pp/yr, not resolved | **worse in 12 of 13 folds**, p = 0.0017 | **REJECT** |
| **B** profit-gated (pyramiding) | −0.304 pp/yr, not resolved | +0.336 pp, worse in 4 of 10, p = 0.83 | **NOT RESOLVED — do not adopt** |

Variant B removes A's harm and delivers no benefit, largely by doing four times less: 19 trades
received adds versus 81. A change that costs complexity and moves nothing measurable is not
worth carrying.

**Date**: 2026-09-15 · **Origin**: `docs/WALKFORWARD_TODO.md` item 3
**Code**: `NNNC/_position_adjust.py`, `research/posadj/arms.py` · **Data**: `research/posadj/out/arms.json`

## Why this was run

Fold f08 measured that freqtrade's one-trade-per-pair rule silently discards **~43% of entry
signals** (≥54 of ~125). `--enable-position-stacking` showed the size of the loss but is
backtest-only and not reproducible in dry/live. `position_adjustment_enable` +
`adjust_trade_position` is the **deployable** form: on a repeat `enter_long`, add one fixed
stake to the open position, capped at 2 adds, minimum 4 bars apart.

## Variant A — unconditional adds: REJECT

| | |
|---|---|
| **Return** | **NOT RESOLVED** — −0.455 pp/yr, 7/13 live folds improved, sign test p = 1.0. \|Δ\| is far below the 7.633 pp noise floor, so this is "no effect detected", not "no effect". |
| **Drawdown** | **RELIABLY WORSE** — balance drawdown up in **12 of 13** live folds, sign test **p = 0.0017**, Wilcoxon **p = 0.00024**, mean +1.58 pp, worst +8.77 pp (f08). |

A deployment modification is supposed to buy drawdown at flat return. This does the opposite:
flat return, reliably worse drawdown. There is no operating point in this design worth having.

Liveness (Principle IV): **81 trades received adds across 15 folds**, 13 of 15 folds live; f00
and f14 saw zero and are excluded as inert rather than counted as ties.

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

## Variant B — profit-gated pyramiding: NOT RESOLVED

Identical to A but for one condition: `current_profit > 0` before adding. Variant A's own
diagnostic pointed here — the harm concentrated in adds made on trades heading for a stop, so
adding only to winners is where value could survive.

| | measured |
|---|---|
| Return | **−0.304 pp/yr** on the 10 live folds, CI [−2.186, +1.626], improved 5/10. Far inside the 7.633 pp noise floor: **not resolved**. |
| Drawdown | **+0.336 pp**, worse in 4 of 10, sign test p = 0.83, Wilcoxon p = 0.11. **A's harm is gone.** |
| Liveness | 19 trades received adds across 10 live folds, versus A's 81 across 13. |

**The gate works as a safety mechanism and fails as an edge.** It removes the adverse-selected
adds that made A dangerous — exactly as A's win-rate table predicted — but what remains is too
small and too neutral to justify shipping. This is the inert-parameter pattern in a mild form:
the fix succeeds by doing almost nothing.

Do not adopt. Do not re-propose either variant.

## What this closes and what it does not

**Closed.** Unconditional repeat-signal adds on NNNC_MLX. Do not re-propose.

**Closed.** Profit-gated pyramiding on NNNC_MLX — safe, but no measurable benefit.

**Not closed — and now better motivated.** The masked-signal finding stands: ~43% of entry
signals are discarded, and this experiment recovered only 83 adds across 15 folds, a small
fraction of them. What it establishes is that the masked signals are **not free money waiting to
be collected** — at least the subset arriving during an open position is adverse-selected.

One variant remains untested: **independent concurrent trades per pair** with separate stops —
the thing `--eps` actually simulated. Not expressible in freqtrade's live engine, so it would
need a different execution model, not a strategy change. Both position-adjustment forms are now
closed, and between them they say the masked signals are not recoverable through a shared-stop
mechanism.

**Meta-lesson for this repository.** An upper bound measured with a *different mechanism* is not
an upper bound on the mechanism you can deploy. `--eps` showed +1.27pp on f08; the deployable
form delivered −6.31pp on the same fold. The two differ in exactly one structural way — shared
versus independent stops — and that one difference reversed the sign.

## Process note — the liveness probe was wrong twice

**First**, it used `nr_of_successful_entries`, which is **absent from exported backtest trade
dicts**, so `.get(key, 1)` reported **zero adds on a working arm**. Caught only because an
"inert" arm had moved P&L by 6 pp.

**Then**, the replacement counted `round(stake / base) - 1`. Adds are capped by available wallet
(`max_stake`), so most land on **non-integer multiples** — 1.48, 1.69, 1.80, 1.95 were all
observed — and rounding those to 1 reports zero adds for a trade that received one. Caught only
because variant B showed a fold with `adds = 0` whose P&L differed from its control, which is
impossible.

That second error mattered: it put variant B at 9 live folds when it had **10**, the difference
between inadmissible and admissible against the minimum. Counts now use *any* excess over one
base stake, plus the deployed capital in base-stake units.

The general lesson is cheap to state and was expensive twice: **a liveness probe needs its own
sanity check.** Here the check is free — a fold with zero adds must be byte-identical to its
control, and any divergence means the probe is lying.
