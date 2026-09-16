# Signal-context timing — **US1 KILL: the repeat signal is a proxy for being underwater**

**Feature**: [`specs/017-signal-context-timing/`](../../specs/017-signal-context-timing/)
**Date**: 2026-09-16 · **Backtests spent: zero** · Artefacts: `research/signalctx/out/`

## Verdict

The motivating effect — trades receiving a repeat entry signal do worse — **does not survive
conditioning on the position's unrealised P&L at the moment that signal arrives.**

| | raw | conditioned |
|---|---|---|
| win-rate delta | **−0.2262** (0.587 vs 0.813, Fisher p < 0.0001) | **−0.0386** |
| CMH across 10 unrealised-P&L strata | — | χ² = 3.35, **p = 0.0673** |
| logistic coefficient on `repeat` | — | **−0.0070, p = 0.973** |
| 1:1 matched, 200 resamples | — | median Fisher p = 0.606; **0%** reach p < 0.05 |

**Conditioning removes 83% of the effect**, and the logistic coefficient is indistinguishable
from zero: once the position's current mark is known, the repeat signal adds essentially nothing.
The model re-fires after price has fallen, so "received a repeat signal" is a restatement of "is
losing" — and losing trades ending badly is close to tautological.

The stratified and 1:1 matched designs agree. The second was run because the stratified control
pool reuses trades across events (34,604 control rows against 126 repeat events), which violates
independence; the matched design uses one nearest-neighbour control per event and reproduces the
null.

**Consequence.** US2, the exit-side rule, was **not built**. A trigger on this signal would be
acting on the position's own mark under another name — which the existing stoploss-grace and
ATR-adaptive stop already do.

## What the specification process caught, and what it was worth

The pre-implementation analyze pass found a **units error that would have failed OPEN**: the
resolution floor of 7.633 pp is an *annualised return* quantity, and it was being applied to a
*win-rate* difference.

The measured `effect_winrate_pp` is **−15.82**. |−15.82| > 7.633, so the gate would have passed
approvingly and the feature would have proceeded to spend **45 backtests** on an effect that is
83% position-mark. Both quantities are called "percentage points"; they are not comparable.
`confound.verdict()` now raises rather than accept the wrong one, and a test pins it.

## Two defects found during implementation

**1. The fidelity matcher, not the reconstruction.** The all-fold gate failed at 13/15, and my
first explanation — a warm-up boundary artefact — was wrong, generalised from f08's single miss.
Five of six misses were **1,660–6,242 bars into** their test range. The real cause: a trade can
open **mid-run**, when its pair is occupied at a signal run's first bar and freed part-way
through, and the matcher compared against run **starts** only. Fixed → **15/15 folds, mean
fidelity 0.999** (377 of 378 recorded trades matched).

**2. Gross versus net.** `reconcile()` compared a gross close-rate return against a net
`profit_ratio`, so all 378 trades missed by **0.00202 ± 0.0003** — a residual too tight across 15
folds to be anything but a constant, and it was the round-trip fee. Fixed the formula rather than
the tolerance → residual now **exactly 0.00000000**.

Both are pinned by regression tests in `research/labelfree/tests/test_signalctx.py`.

## Measurements that stand on their own

| | |
|---|---|
| signal **bars** pooled across 15 folds | 730 |
| signal **events** (runs collapsed) | **644** — 11.8% of bars are continuations |
| admitted trades | 378 |
| **masked events** | **266 = 41.3%** of all signal events |
| reconstruction fidelity | 15/15 folds, mean 0.999, min 0.986 |

The masked-signal rate of 41.3% is now **measured directly** across all folds, superseding the
earlier ~43% inferred from `--eps` trade counts on a single fold.

## What was NOT run, and why

- **Phase 6 (US2, exit-side rule)** — stopped by the T023 gate. There is no information to act on.
- **Phase 4 (US3, first-signal context sweep)** and **Phase 5 (US4, masked-signal outcomes)** —
  **not killed, simply not run.** FR-023 kept both alive as EXPLORATORY after the confound kill,
  and the flag is persisted in `out/confound.json`. Work stopped here by decision, not by a gate.
  They remain open questions: whether *first*-signal context predicts quality, and what the 266
  masked signals would have done.
- **Phase 7 (US5)** — unreachable without a candidate rule.
- **T012–T016** — the tests were specified to be written before implementation and were not;
  `test_signalctx.py` was written afterwards and covers the validated invariants rather than the
  full task list. Recorded as a process deviation, not silently closed.

## Where this leaves the thread

`docs/WALKFORWARD_TODO.md` item 3 opened with the observation that ~43% of entry signals are
discarded. That number is now confirmed at **41.3%, measured**. Three explanations for acting on
them have been closed: position stacking (undeployable), position adjustment in both forms
(rejected / not resolved), and now the repeat signal as information (a proxy).

**The masked signals remain unexplained, and that is the honest open question.** They are real,
numerous, and nothing yet shows they are worth anything.
