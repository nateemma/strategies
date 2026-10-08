# Implementation Plan: Label-Exit Coordination

**Branch**: `021-label-exit-coordination` | **Date**: 2026-10-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/021-label-exit-coordination/spec.md`

## Summary

Extract the exit policy into one shared primitive, precompute a per-bar outcome table under
that one policy, and express labels as `trigger AND positive outcome` so the only thing that
varies between labels is the trigger. Then score every trigger on two independent axes —
economic value with no model, and learnability via one fixed probe — to settle whether a
trigger earns its keep by selecting collectable moments or by handing the model an easy
target.

## Technical Context

**Language/Version**: Python 3.12 (the virtualenv at `.venv`; never conda)

**Primary Dependencies**: pandas, numpy, scikit-learn (the probe), freqtrade (data loading and
the live strategy path), MLX only where existing machinery already requires it

**Storage**: Feather for the outcome table, keyed by `(dataset, exit-configuration hash)`;
JSON for the trigger leaderboard. Mirrors `research/clfstudy`'s existing cache conventions.

**Testing**: pytest, in the existing per-package test directories

**Target Platform**: macOS / Apple silicon, local

**Project Type**: Research instrument plus one shared runtime primitive

**Performance Goals**: Evaluating one trigger — both columns — takes seconds. Building the
outcome table is a one-off per `(dataset, exit config)` and may take minutes.

**Constraints**: All code under `user_data/`. A live dry-run (`MomentumRegimeBasket15mFast`,
26 days running) must see no behaviour change. `saved_data/main_scaler.pkl` and
`gan_scaler_a.pkl` must not be touched. The exit policy is held FIXED. Any P&L claim comes
from `evaluation/`, which resolves ~3–4pp/yr and not 1–2pp.

**Scale/Scope**: 11 production pairs, 15m timeframe, ~200k bars per pair, 2021 to present.
One shared exit primitive, one table builder, a trigger registry, one probe, one report.

## Constitution Check

| Principle | Status | How this feature satisfies it |
|---|---|---|
| I. Evaluate in the Deployment Regime | PASS | Promotion claims come from the pooled walk-forward harness over the deployment era; the outcome table spans the same span. |
| II. Persistence Across Distant Eras | PASS | The harness's 15 folds span 2023–2026; no single-window claim is made. |
| III. Learnability Is Not Edge | **CENTRAL** | The feature exists because learnability tracked return across four arms while one of them lost money. FR-015a makes the learnability column explicitly diagnostic and FR-018 bars it from acceptance. |
| IV. Verify Parameters Are Actually Live | PASS | SC-001 proves the extracted exit reproduces live decisions bar-by-bar; SC-002 proves no second copy exists. |
| V. Lookahead-Clean Is Not Realistic | PASS | FR-009a/b record fillability and report every value both unfiltered and fillable-only, so a trigger cannot earn value on bars that could not have been traded. |
| VI. Diagnose Before Tuning | PASS | The whole feature is diagnosis: it measures what triggers do before anything is tuned. |
| VII. Validate Cheaply Before Building | PASS | The table makes trigger evaluation seconds instead of 1.5 hours — this principle is the feature's design rationale. |
| VIII. One Parameter Space at a Time | PASS | The exit is held fixed (FR-019); only the trigger varies. |
| IX. Orthogonality Earns a Factor Its Place | N/A | No multi-factor combination here. |
| X. Structural Fixes, Not Per-Pair Overrides | PASS | The table and triggers are computed per pair but no trigger may be pair-specific. |
| XI. No Data Leakage in Model Evaluation | PASS | FR-009 structurally bars outcome columns from reaching a model; the probe reuses `research/clfstudy`'s splits, which already carry purge/embargo ≥ label horizon. |

**Repository constraints**: all new code under `user_data/strategies/`; `.venv` interpreter;
nothing written into the upstream `freqtrade/` package.

No gate violations. One judgement is recorded in Complexity Tracking below.

## Project Structure

### Documentation (this feature)

```
specs/021-label-exit-coordination/
├── spec.md
├── plan.md               <- this file
├── research.md           <- Phase 0 decisions
├── data-model.md         <- Phase 1 entities
├── contracts/
│   ├── exit-policy.md    <- the shared primitive's contract
│   ├── outcome-table.md  <- schema, cache key, leakage guard
│   └── trigger-cli.md    <- registry and command surface
├── quickstart.md         <- runnable validation
└── checklists/
    └── requirements.md
```

### Source Code

```
user_data/strategies/
├── utils/
│   └── ExitPolicy.py              <- NEW. The single exit primitive. Pure functions:
│                                     no strategy, no dataprovider, no freqtrade imports.
├── Framework/
│   ├── BaseStrategy.py            <- MODIFIED. custom_stoploss and _resolve_normal_stop
│                                     become thin callers of utils.ExitPolicy. The
│                                     freqtrade return convention (1.0 = no change,
│                                     rescaling relative to current_rate) stays here —
│                                     it is glue, not arithmetic.
│   └── tests/
│       └── test_exit_policy_single_source.py   <- NEW. The duplication test (SC-002)
│                                                   and the characterisation test (SC-001).
├── NNPredict/
│   └── NNPredictStrategy.py       <- MODIFIED. _stop_aware_labels calls utils.ExitPolicy;
│                                     its duplicated arithmetic is deleted.
├── research/outcomes/             <- NEW package. Offline; nothing here runs in a backtest.
│   ├── __init__.py                <- version, cache paths, interpreter guard
│   ├── table.py                   <- builds/loads the per-bar outcome table
│   ├── triggers.py                <- the trigger registry
│   ├── probe.py                   <- the fixed learnability probe
│   ├── leaderboard.py             <- the two-column report + verdict rule
│   └── tests/
└── evaluation/                    <- UNCHANGED. Still the only source of a P&L claim.
```

**Structure Decision**: the shared exit primitive goes in **`utils/`**, not `Framework/` and
not `evaluation/`. `Framework/` is strategy-agnostic *runtime* code that strategies inherit
from, and the offline table must not import a strategy to get a stop level. Nothing in
`evaluation/` runs inside a backtest, so the live path cannot depend on it. `utils/` is
already the home of shared primitives imported by both the strategy path
(`utils/DataframePopulator.py`, `utils/Environment.py`) and research code
(`research/clfstudy` imports `utils.DataframePopulator`), and it imports cleanly with no
strategy present — verified. That makes it the only existing location both sides can reach.

The offline work goes in a new `research/outcomes/` package rather than inside
`research/clfstudy/`, because clfstudy is a closed study with a frozen results file and 340
tests pinned to its config-id scheme; adding a second concern to it would put those at risk.

## Complexity Tracking

| Decision | Why it is justified |
|---|---|
| A new `research/outcomes/` package rather than extending `research/clfstudy` | clfstudy's `config_id` scheme is frozen by a test asserting historical ids are unchanged, and its `results.jsonl` is an append-only record of ~2,000 runs. Extending it would couple a new concern to that invariant for no benefit. The probe reuses clfstudy's splits and metrics by import. |
| Modifying `Framework/BaseStrategy.py` while a dry-run is live | Unavoidable: the duplication cannot be removed without touching the place it lives. Mitigated by SC-001, a bar-by-bar characterisation test over real data, and by leaving the freqtrade return convention untouched so only the arithmetic moves. |
| An allowlist-based duplication test rather than AST analysis | See research.md R3. An AST matcher for "is this expression a stop clamp" is itself a heuristic that would drift; an explicit allowlist of files permitted to name the exit parameters is unambiguous, cheap, and fails loudly when someone adds a copy. |
