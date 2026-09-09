# Implementation Plan: Pooled Walk-Forward Evaluation Harness

**Branch**: `007-pooled-walkforward-harness` | **Date**: 2026-09-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification — 24 FRs, 8 SCs, 4 user stories, 7 edge cases, checklist 16/16

## Summary

Build a measurement instrument that pools many genuinely out-of-sample walk-forward folds into one
trade set, so the bootstrap noise floor falls from today's **7.633pp** (99 trades, one window) to
**≤4.0pp**. The harness orchestrates existing tools — it invokes `run_arm.py` per fold and feeds
`compare.py`'s paired bootstrap — and adds only fold planning, fold-identity management, pooling,
and power reporting.

**The deliverable is statistical power, not P&L.** No gate, exit, label, or model configuration
changes.

## Technical Context

**Language/Runtime**: Python 3.12 via the repo `.venv`; zsh scripts run from the repo root.
**Primary dependencies**: existing `regime/run_arm.py`, `regime/tail/compare.py` (31183cb),
`scripts/test_strat.sh`, freqtrade backtesting, MLX training.
**Storage**: fold records under `regime/wf/` (new); per-trade data read from
`user_data/backtest_results/*.zip`, the only per-trade source.
**Testing**: measurement validation, not unit tests — the acceptance evidence is the reported floor.
**Target platform**: local macOS, 14 cores, sequential execution only.
**Project type**: single research repository.
**Performance goals**: not applicable — this feature's "performance" is the achieved noise floor.
**Constraints**: ~10-15 min per model train; sequential only; zero changes to `run_arm.py`'s schema
or to `Framework/`.
**Scale/scope**: 15 folds × 1 family × 1 seed for the first run; seeds and second family are
later phases.

### Decided: fold geometry (the central tradeoff)

**Decision: ANCHORED (expanding) training, 24-month minimum, 90-day test folds.**

| option | test span | pooled NNNC trades | est. floor | verdict |
|---|---|---|---|---|
| anchored, 24mo min | 3.68 yr | ~365 | 3.98pp | **CHOSEN** |
| rolling 18mo | 4.19 yr | ~414 | 3.73pp | rejected — departs from production |
| rolling 12mo | 4.68 yr | ~464 | 3.53pp | rejected — largest quality risk |

Rationale: production trains on *all data up to a cutoff* (20220101-20250831, ~44 months), so an
**anchored** expanding window is the faithful analogue; a fixed short rolling window would change
the training regime and confound the instrument with the thing it measures. Requiring ≥24 months
before the first test fold places the first test at **2023-01-01**, giving 1,345 test days.

Cost: the chosen option is the *weakest* of the three on floor (3.98pp vs 3.53pp) and only just
clears SC-001's ≤4.0pp. That is accepted deliberately — a floor achieved by degrading the models
would not be a valid measurement of the production strategy.

**Risk, explicitly tracked:** early folds train on ~24 months against production's ~44, so early
fold models may be weaker. **Detection (FR-010):** per-fold val metrics and per-trade expectancy are
recorded and plotted against training length. A monotone relationship between training length and
expectancy is evidence the instrument is confounded, and is a reportable finding — not something to
average away.

### Decided: compute budget

~12.5 min per training, sequential.

| phase | scope | trainings | est. wall clock |
|---|---|---|---|
| B (first run) | NNNC, 15 folds, seed 1 | 15 | **~3.1 h** |
| D (seed check) | NNNC, 4 folds, seed 7 | 4 | ~0.8 h |
| E (pair expansion) | NNNC, 15 folds, expanded universe | 15 | ~3.1 h |
| F (second family) | NNMT, 15 folds, seed 1 | 15 | ~3.1 h |

**NNNC only, one seed, for the first run.** NNNC is the cheaper family and its 99-trade window is
the one the noise floor was measured on, so it is the correct baseline to improve. Full seed
coverage over all folds (30 trainings, 6.3 h) is not justified before the floor is demonstrated;
FR-011's identical-seed detection is exercised on a 4-fold subset in phase D.

## Constitution Check

| principle | status | note |
|---|---|---|
| I. Evaluate in the Deployment Regime | **PASS** | Folds span eras; FR-021 keeps recent-subset recomputation available, so the deployment regime remains inspectable rather than averaged away. |
| II. Persistence Across Distant Eras (NON-NEGOTIABLE) | **PASS WITH INTERPRETATION** | See below. |
| III. Learnability Is Not Edge | **PASS** | No classifier metric is an input to any verdict; val metrics are recorded only as the confound detector for training length. |
| IV. Verify Parameters Are Actually Live | **PASS** | FR-004 verifies the model artefact exists; fold identity (FR-003) prevents silent weight reuse. |
| V. Lookahead-Clean Is Not Realistic | **PASS** | Volume/liquidity constraints are unchanged and remain active; the harness alters nothing about execution. |
| VI. Diagnose Before Tuning | **PASS** | This feature is the diagnosis — it measures the instrument before any further tuning is attempted. |
| VII. Validate Cheaply; Test Trackability | **PASS** | Phase B (3.1 h, one family, one seed) is deliberately the cheap validation; expansion is gated on it. |
| VIII. One Parameter Space at a Time | **N/A** | No parameter is tuned. |
| IX. Orthogonality Earns a Factor Its Place | **N/A** | No factor is added. |
| X. Structural Fixes, Not Per-Pair Overrides | **PASS** | Pair-group reporting (FR-014) is measurement, not per-pair logic. |

### Principle II — how persistence is demonstrated on pooled folds

Principle II is **not waived**. Its quality gate ("at least three windows, at least one temporally
distant") was written for single-window comparisons and is *strengthened* here: a 15-fold plan
spanning 2023-01 to 2026-09 contains far more era diversity than three hand-picked windows.

What changes is the **meaning of a flip**. The repo has treated "it reversed in a distant era" as
grounds for rejection. Measured 2026-09-09, that inference is unsafe between two underpowered
windows: rank correlations of parameter optima across W1/W2/W3 were contradictory (rvol W1~W2
-0.46, grace W1~W2 +0.71) and none significant, which is what two noise draws look like.

Therefore, on pooled folds, persistence is demonstrated by **per-fold effect signs and their
dispersion**, not by whether two windows agree:

1. The pooled effect must be resolved (CI excluding zero) on the full fold set.
2. The per-fold effects must be reported individually, with the count of folds agreeing in sign.
3. An effect resolved overall but with sign agreement no better than chance is reported as
   **regime-contingent**, not accepted.
4. FR-021's era-subset recomputation must show the effect surviving on the most recent third.

**Quality gate 5 is superseded in effect, and this must be recorded:** it requires paired repeated
runs for any accepted delta under 1pp. The measured floor makes any sub-3pp delta unresolvable
regardless of repetition, so no sub-1pp delta can be accepted at all. Repetition cannot rescue an
effect below the noise floor.

## Project Structure

### Documentation (this feature)

```
specs/007-pooled-walkforward-harness/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── contracts/
│   └── harness-cli.md   # Phase 1
├── quickstart.md        # Phase 1
└── checklists/requirements.md
```

### Source code (repository root = user_data/strategies)

```
regime/
├── wf/                       # NEW — everything this feature adds
│   ├── foldplan.py           # build + VALIDATE the fold plan (FR-001, FR-002)
│   ├── run_folds.py          # orchestrate train->evaluate per fold (FR-003..FR-005)
│   ├── pool.py               # pool per-trade results, retain fold identity (FR-006)
│   ├── power.py              # bootstrap floor + inter-fold correlation (FR-007..FR-009)
│   └── folds/                # fold records, one JSON per fold
├── run_arm.py                # UNCHANGED — invoked, schema untouched (FR-018)
├── tail/compare.py           # UNCHANGED — fed, not forked (FR-013)
└── regimedata.py             # UNCHANGED

NNNC/NNNC_WF_f*.py|.json      # generated per-fold classes + params (git-ignored)
                              # NOTE: these sit DIRECTLY in NNNC/, not a subdir —
                              # run_arm.py resolves a fold class to its family dir and
                              # symlinks only top-level *.py, so a _wf/ subdir is invisible
Framework/                    # UNTOUCHED (FR-019)
```

**Structure decision**: a new `regime/wf/` package, because every existing module is either
schema-frozen (`run_arm.py`, 480+ dependent records) or the statistic of record (`compare.py`).
Per-fold strategy classes are generated into `NNNC/_wf/` rather than hand-written: 15 folds × a
class + a `.json` each is exactly the kind of repetition where a hand-copied `strategy_name` or a
missing `.json` silently changes the entry gate.

## Complexity Tracking

| item | why it is needed | simpler alternative rejected because |
|---|---|---|
| Generated per-fold strategy classes | `get_model_path` keys off `self.__class__.__name__`; without distinct identities folds overwrite each other's weights, and a class with no model of its own trains **inside its own backtest** with returncode 0 | A shared class with a runtime path override would need a `Framework/` change (FR-019 forbids it) and would make the failure silent rather than impossible |
| A new `regime/wf/` package | `run_arm.py`'s schema is frozen by 480+ records; `compare.py` is the statistic of record | Extending either would either break stored records or fork the statistic |
| Per-fold `.json` generation | file params beat class params; a fold class without its own `.json` silently inherits all-guards-open defaults | Hand-copying 15 files is exactly how a `strategy_name` mismatch or a stale gate gets introduced |
