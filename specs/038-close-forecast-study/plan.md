# Implementation Plan: Close-Price Forecast Study

**Branch**: `038-close-forecast-study` (spec directory; work is committed to master) | **Date**: 2026-10-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/038-close-forecast-study/spec.md`

## Summary

The study asks whether any forecasting method predicts the N-candle return of close well enough to
trade long after costs, and at which N. It works in five steps:

1. **Fix the forecaster library.** Seven defects (R4), each confirmed with a known-answer test.
2. **Score baselines and a cost line per horizon.** This rules out horizons that are too short before
   any model is read.
3. **Pre-register a cost-aware pass rule.**
4. **Score three tiers of methods offline.** Statistical forecasters at stride 16, LightGBM on lagged
   returns with and without the ARO inputs, and LightGBM with wavelet/FFT inputs. A shuffled-target
   null control and a seed-2 rerun guard the learned tier.
5. **Give one verdict per (method, N).**

The verdict uses long-only trades:
- entered at the next open and held N candles, one position per pair at a time;
- net of 0.10% fees plus 5 bps slippage per side;
- compared with random entry, in both halves.

Skill, IC and hit rate are reported as description only.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)

**Primary Dependencies**:
- numpy 1.26.4, pandas 3.0.5, scipy 1.17.1;
- statsmodels 0.15.0, scikit-learn 1.9.0, lightgbm 4.7.0, PyWavelets 1.8.0;
- freqtrade's history loader, for gap-filled candles.

**Storage**:
- feather files under `user_data/data/forecast_038/` (frames and predictions; git-ignored);
- JSON and markdown under `research/forecast/` (committed).

**Testing**: pytest (`utils/tests/`, `research/forecast/tests/`)

**Target Platform**: macOS on Apple silicon; CPU only for this study.

**Project Type**: an offline research harness (a CLI made of modules) plus library fixes.

**Performance Goals**:
- baselines and the statistical tier finish in under 1 h (SC-006): baselines are vectorised, and the
  statistical tier takes about 20 min with 4 processes (R5);
- the learned tier takes 3–5 h as one sequential job (R7).

**Constraints**:
- one training job at a time;
- never touch the running bots (pids 2661, 41849);
- no pair-specific code;
- no freqtrade backtests.

**Scale/Scope**:
- 10 pairs, about 1.29 M scored test candles, 8 horizons;
- about 15 methods, including the controls;
- 480 LightGBM fits.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | How the plan complies |
|---|---|
| I. Deployment regime | The PASS requires a positive net lower bound in the LATE half (R9 d). Full-sample figures are only context |
| II. Persistence (NON-NEGOTIABLE) | The rule must hold in both halves. There is no distant era (binance.com data is empty), so every PASS is PROVISIONAL, as in 023 onwards (R2). This is a declared limitation, not a silent exception |
| III. Learnability ≠ edge | The verdict rests on the cost check against random entry (R8, R9). R², IC and hit rate are descriptive (FR-010) |
| IV. Parameters live | A liveness assertion checks that forecasts differ across N and that the null differs from L (R12) |
| V. Realistic execution | Long only, entry at the next open, fees plus a slippage grid, zero-volume candles not tradeable (R1, R3, R8). Trade size is not modelled: the cost check is a per-trade return, independent of stake, and slippage stands in for size. This exception is justified under Complexity Tracking |
| VI. Diagnose before tuning | There is no tuning. All hyperparameters are fixed in the pre-registration, and per-pair and per-fold diagnostics are reported |
| VII. Validate cheaply | Story 2 (baselines and cost line) runs first and can rule out horizons before any model runs. Tier 1 runs before tier 2 |
| VIII. One parameter space | There is no search: window 64, one LightGBM configuration |
| IX. Orthogonality | Not decision-relevant here, since the verdict is per forecast. Correlation with the NNNC signal is deferred to the follow-up feature that tests a passing forecast as a filter or sizer |
| X. No per-pair overrides | Pooled models and the same configuration for every pair (FR-015) |
| XI. No leakage | Anchored training ends N candles before the test start. LightGBM needs no scaler. The 037 inputs passed truncation, and the LW inputs pass it before use (FR-004, FR-012) |
| Gate 1 (pinned range) | The fold windows are fixed by `plan_cs_aro.json` |
| Gate 2 (distant era) | Not met, so verdicts are PROVISIONAL (see II) |
| Gate 3 (realistic execution) | Partly met (see V and Complexity Tracking) |
| Gate 4 (liveness) | R12 |
| Gate 5 (seed robustness) | A learned PASS requires a seed-2 PASS (R9 e) |
| Gate 6 (trade count) | At least 200 trades per half, otherwise INCONCLUSIVE. Zero-trade cells are never read as zero return |
| Gate 7 (scalers) | No feature scalers are used, so it does not apply |
| Repo boundary | All edits are in `user_data/strategies`. Commits go to master in the nested repo |
| Framework agnostic | No `Framework/` edits |

**Result**: PASS with two written exceptions: Gate 2 (PROVISIONAL verdicts, the same stance as every
study since 023) and the size part of Principle V / Gate 3 (see Complexity Tracking).

**Re-check after Phase 1**: PASS. The design adds no scaler, no per-pair code and no backtest.

## Project Structure

### Documentation (this feature)

```text
specs/038-close-forecast-study/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/cli.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code

```text
utils/
├── Forecasters.py                      # fixes D1–D5 (R4)
├── test_data.npy                       # restored exploration data (committed)
├── test_forecasters.py                 # plot playground: enable HOLT; catch ForecastUnavailable (T014)
└── tests/test_forecaster_horizons.py   # known-answer tests (Story 1)
TSPredict/TS_Wavelet.py                 # D6: dead methods raise NotImplementedError

research/forecast/
├── __init__.py
├── common.py          # paths, pairs, folds, horizons, costs, pre-registration loader, refusal guard
├── frames.py          # candle frames with targets, trade returns, eligibility
├── baselines.py       # zero / historical mean; horizon table and cost line
├── instrument.py      # SC-001/SC-002 checks
├── stat_tier.py       # tier 1 (stride 16, worker pool)
├── features.py        # sets L, LA, LW
├── truncation.py      # FR-012 truncation test for LW inputs
├── learned_tier.py    # tier 2/3 LightGBM walk-forward, null control, seed
├── score.py           # trade simulation, cell scores, bootstrap, descriptive metrics
├── verdict.py         # R9 rule; refusal guard
├── report.py          # report.md
├── preregistration.json
└── tests/
    ├── test_frames.py
    ├── test_score.py
    └── test_verdict.py
```

**Structure Decision**: one research package, next to the other studies (033–037), with the fixes made
in place in the shared library. Large outputs go under `user_data/data/forecast_038/`.

## Complexity Tracking

| Exception | Why needed | Simpler alternative rejected because |
|-----------|------------|-------------------------------------|
| Principle V / Gate 3: trade size against volume is not modelled | The study's output is a forecast verdict, not a sized book. The per-trade return is independent of stake, and zero-volume candles are already untradeable | Modelling stake against candle volume needs a book (stake, wallet, concurrency) that this study deliberately does not build (FR-014). Capacity for any passing forecast is assessed in the follow-up filter/sizer feature, under the 036 cost model when the live gate opens |
| Gate 2: no distant era | binance.com data is empty, so halves are the only persistence test available | No other data exists; verdicts are labelled PROVISIONAL |
