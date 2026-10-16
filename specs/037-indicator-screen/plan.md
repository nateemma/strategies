# Implementation Plan: Indicator Screen Against Labels and Regimes

**Branch**: `037-indicator-screen` (spec directory; work is committed to master) | **Date**: 2026-10-16 |
**Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/037-indicator-screen/spec.md`

## Summary

The feature harvests every SimpleStrategies indicator (67 strategies) once per pair. Each strategy's
own code is called, and its raw signals are taken before any filter or guard. Columns that fail a
truncation lookahead test are dropped. The surviving columns are then scored against:
- the gbb_path BUY and SELL labels (the ARO control's labels);
- three regime definitions (A1/A2 strategy outcome, B direction, C volatility) at 12 h / 3 d / 7 d,
  each scored on the pair's own indicator values and on BTC's.

Scoring is done:
- on two populations (all bars, gbb candidate bars);
- per walk-forward test window, plus a descriptive 2021–22 era E0.

It runs at three levels:
- single columns, including residual strength beyond bb_width + guard_metric;
- pairs of signals, both firing (selected early, confirmed late);
- per-fold tree models against a bb_width + guard_metric-only model.

Late windows are locked behind a committed pre-registration. Outputs are JSON plus a report. No
backtests, no neural training, and no change to any strategy or bot.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**:
- freqtrade (StrategyResolver, history loader);
- TA-Lib, pandas-ta, finta (already used by the strategies);
- pandas, numpy, scikit-learn 1.9.0 (DecisionTree, HistGradientBoosting), scipy (clustering).

**Storage**:
- feather harvest cache in `user_data/data/indicator_screen/` (outside the nested repo; never
  committed);
- committed JSON and markdown in `research/indicator_screen/`.

**Testing**: pytest (`research/indicator_screen/tests`), added to the full-suite command.

**Target Platform**: macOS (Apple silicon), CPU only.

**Project Type**: research tooling (CLI modules under `research/`) plus one plotting strategy class.

**Performance Goals**:
- harvest measured on one pair first, expected to take minutes per pair;
- each screen finishes in under an hour;
- one fold per process for model fits.

**Constraints**:
- must not disturb the live NNNC bot (pid file `user_data/logs/live_nnnc.pid`) or the momentum bot
  (pid 2661);
- one heavy chain at a time;
- per-pair streaming to keep memory under about 2 GB.

**Scale/Scope**:
- 10 pairs + BTC × ~200k bars;
- ~350 columns;
- ~90 raw buy signals → ~4,000 pairs;
- 15 folds + E0;
- 2 label targets + 6 regime target families × 3 horizons.

## Constitution Check

| Principle | Status | How |
|---|---|---|
| I Deployment regime | PASS | The late half (2025-01 → 2026-09) is reported separately, and no full-history average is the headline. The screen makes no deployment claim. |
| II Persistence (NON-NEGOTIABLE) | PASS | Every candidate needs early AND late. E0 (2021–22) is scored for single columns and pairs as the distant era (R3). Multivariate models have no E0, so their limitation is stated, and a later backtested feature must cover 2022. |
| III Learnability ≠ edge | PASS | Screen only (FR-025). No acceptance. Candidates are nominated for backtests in later features. |
| IV Parameters live | PASS | Every override (Ehlers triggers, BBBreakout forced trend_mode, Ichimoku switches) is proven to change the signal (R1). Constant or never-firing columns are excluded with a reason. |
| V Lookahead / realism | PASS | Truncation test on every column (R12). No P&L figures are produced. |
| VI Diagnose before tuning | PASS | Nothing is tuned. Model settings are fixed in the pre-registration. |
| VII Validate cheaply | PASS | Harvest cost measured on one pair before the full run. The instrument check gates every later screen. |
| VIII One space at a time | N/A | No parameter search. |
| IX Orthogonality | PASS | Redundancy groups and correlations with guard_metric and bb_width (R8), plus the stratified residual (R7), are exactly this check. |
| X No pair-specific code | PASS | Generic discovery. Overrides are per strategy class, not per pair. |
| XI No leakage | PASS | MI bin edges, residual cells, tercile cut points and model fits all come from each fold's training window, with purge ≥ the target horizon (R6, R7, R10, R11). The production scaler is not used (indicator values are raw). The A2 cut-point exception is pre-registered and flagged. |
| Repository boundary | PASS | Code in the nested repo only. The cache is under `user_data/data/` (not committed). |
| Framework strategy-agnostic | PASS | Framework/ untouched. |
| Params merge | PASS | AllIndicators defines no parameter dicts of its own beyond merging from the parent. |

Re-check after Phase 1: unchanged. The design adds no fitted artefact outside a fold's training
window.

## Project Structure

### Documentation (this feature)

```text
specs/037-indicator-screen/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/cli.md
└── tasks.md            # /speckit-tasks
```

### Source Code (nested repository `user_data/strategies`)

```text
SimpleStrategies/
└── AllIndicators.py            # plotting strategy; populate_indicators = harvest (R14)

research/indicator_screen/
├── __init__.py
├── spec_037_draft.md           # (exists)
├── discover.py                 # in-scope classes + exclusions (R2)
├── overrides.py                # raw-signal rules: Ehlers triggers, BBBreakout tm0/tm1, Ichimoku (R1)
├── harvest.py                  # base frame + per-class columns + raw signals; cache I/O (R1, R13)
├── causality.py                # column truncation test via research.outcomes.causality (R12)
├── windows.py                  # fold test/train windows, E0, purge, early/late, prereg lock (R3, R15)
├── targets.py                  # labels, candidate flag, regime targets A1/A2/B/C, baselines (R4, R5, R11)
├── scores.py                   # AUC, binned MI, lift, stratified AUC, aggregation (R6, R7)
├── screen_single.py            # US2 + US4 single-column screens; instrument check; redundancy (R8)
├── screen_pairs.py             # US3/US4 pairwise AND screen (R9)
├── screen_models.py            # US3/US4 per-fold tree models, one fold per process (R10)
├── report.py                   # report.md + the SC-004 answer lists
├── preregistration.json        # committed before any late window is scored (R15)
├── manifest.json, causality.json, single_*.json, pairs_*.json, models_*.json   # outputs
├── report.md
└── tests/
    ├── test_discover.py
    ├── test_overrides.py
    ├── test_harvest.py
    ├── test_causality.py
    ├── test_windows.py
    ├── test_targets.py
    ├── test_scores.py
    ├── test_screens.py
    └── test_report.py
```

**Structure Decision**: one research package (the same pattern as `research/new_inputs`,
`research/label_screen`) plus one plotting strategy in SimpleStrategies. The cache lives outside the
repo.

## Complexity Tracking

No constitution violations.
