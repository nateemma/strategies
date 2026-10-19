# Implementation Plan: NNPredict Improvement Study

**Branch**: `039-nnpredict-improvement` (spec directory; work is committed to master) | **Date**: 2026-10-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/039-nnpredict-improvement/spec.md`

## Summary

Five steps, each judged on 15-fold walk-forward P&L under rules committed before their results:

1. **Fix NNPredict's plumbing:** the model-key classification, Ridge artifacts and the README (R1–R2).
2. **Score the existing Ridge configuration** on the common harness, plus a `stop_aware` target arm.
3. **Add a plain-LSTM regressor** (the regression twin of ARO's `LSTM_RAW`) with three input sets: raw,
   raw + indicators, raw + lagged returns.
4. **Port the NNNC entry framework** to the family base with everything off, prove it gives identical
   trades, then test 24 h grace, the in-profit held-pair add, momentum family exits and the gbb
   trigger gate on the best model.
5. **Screen the best model's predicted move** as a ranker of the early width-book trades. If it
   passes, run it as a `cs_aro`-style sizer against the width book.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)

**Primary Dependencies**:
- freqtrade (the repo's checkout) and MLX (the regressors);
- scikit-learn (Ridge), numpy/pandas, scipy (the screen);
- the existing harness: `evaluation/run_folds.py`, `research/nnnc_switches/judge.py`,
  `research/nnnc_framework/signals.py`, `research/new_inputs/trades.py`.

**Storage**:
- trained models under `saved_data/`;
- fold records in `regime/arms/` (local, not committed, as for recent studies);
- verdicts and reports in `research/nnpredict/`.

**Testing**: pytest (`NNPredict/tests/`, `Framework/tests/test_model_key.py`, `utils/tests`,
`research/nnpredict/tests/`)

**Target Platform**: macOS on Apple silicon, MLX GPU for training.

**Project Type**: strategy-family changes plus a research harness.

**Performance Goals**:
- one trainer at a time;
- LSTM fold training measured on f00 before the rest runs (R10).

**Constraints**:
- never touch the bots (pids 2661, 41849);
- parameter dicts merge from their parents;
- no pair-specific code;
- `Framework/` stays family-agnostic;
- the user's uncommitted files are never staged.

**Scale/Scope**:
- 5 model arms (Ridge, Ridge stop_aware, 3 LSTM), 1 identity arm, 4 feature arms, 1 sizer arm;
- 15 folds each.

## Constitution Check

| Principle | Compliance |
|---|---|
| I. Deployment regime | Verdicts need the late half (f08–f14) not worse (judge_one rule maps) |
| II. Persistence | Both halves. No distant era, so PROVISIONAL, as since 023 |
| III. Learnability ≠ edge | Verdicts on P&L only. Fit metrics are reported, never decisive (FR-010) |
| IV. Liveness | The identity arm proves the port is inert. Every feature arm must change trades on at least one fold (judge_one's `folds_changed`; 0 = INERT, a defect) |
| V. Realistic execution | The harness's fillable cap and fees, as for all books. Capacity is deferred to the 036 cost model, as for `cs_aro` |
| VI. Diagnose before tuning | No gate tuning. Prediction spread is reported per fold (edge case: near-constant predictions) |
| VII. Validate cheaply | The sizer backtest is gated behind the S1 screen (FR-008). LSTM training is timed on f00 first |
| VIII. One parameter space | Each feature arm changes one thing against its fw0 |
| IX. Orthogonality | The sizer's agreement is correlated with `cs_aro`'s p_buy agreement and reported |
| X. No per-pair overrides | None |
| XI. No leakage | Models are trained per fold on its training window (run_folds). New inputs use causal rolling normalisation, with no fitted artifact (R4). The shared 6-column scaler is a pre-existing exposure, identical across arms (035 R1), and is recorded |
| Gates 1–7 | Pinned folds. Halves (gate 2: PROVISIONAL). Harness execution. Liveness (IV). Seed: any accepted delta < 1 pp gets a seed-2 rerun. Trade counts reported. No new fitted scaler (gate 7 n/a) |

**Result**: PASS, with the same declared limitations as 033–038 (gate 2; the shared-scaler exposure).

## Project Structure

```text
specs/039-nnpredict-improvement/   spec, plan, research, data-model, quickstart, contracts/, tasks
NNPredict/
├── NNPredictStrategy.py      R1 model-key fix; R6 EntryFrameworkMixin (all off)
├── NNPredictRegressorMLX.py  R4 LSTM_RAW regressor
├── _inputs.py                R4 raw8 / lagged-return input mixins (causal-normalised)
├── NNPredict_LSTM_Raw.py, NNPredict_LSTM_RawInd.py, NNPredict_LSTM_RawRet.py
├── NNPredict_Ridge_SA.py     stop_aware arm
├── README.md                 FR-003
└── tests/                    model key, artifact check, inputs causality, framework identity
evaluation/run_folds.py       R2 (only if the artifact check needs .joblib)
evaluation/folds/plan_nnp_*.json
research/nnnc_framework/signals.py   R8 plan entry + %predict_gain capture
research/nnpredict/
├── preregistration_models.json, preregistration_framework.json, preregistration_sizer.json
├── judge.py                  models / framework / sizer verdicts
├── screen.py                 R8 S1 screen
├── build_arms.py             fold plans + arm classes for the R7-selected model
└── report.md, verdicts_*.json
```

## Complexity Tracking

| Exception | Why | Rejected alternative |
|---|---|---|
| Shared `main_scaler.pkl` reused by every fold (Principle XI exposure) | Identical across all arms and comparators. A per-fold refit would void every existing comparator (035 R1) | Per-fold scalers for all arms: retrains the comparators, out of scope |
| R7 selects with both halves in view | A single model must carry the framework and sizer arms, and the cheap screen keeps the sizer's late half blind | Separate selection data: none exists (no distant era) |
