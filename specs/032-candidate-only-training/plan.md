# Implementation Plan: Candidate-Only Training for NNNC Models

**Branch**: `032-candidate-only-training` (spec directory; committed to master) | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/032-candidate-only-training/spec.md`

## Summary

Train ARO and GBR only on windows whose last bar passes the gbb rule (guard_metric < 0.3 AND
bb_width > 0.035). The model's capacity then goes to ranking the rule's candidates instead of
re-learning the rule. The arms are judged in pre-registered stages against their 24 h controls
and the model-free width books:
1. within-candidate AUC;
2. the early half;
3. the late half, confirmatory.

**Technical approach**:
- A strategy-agnostic row filter, declared as data on the strategy (`TRAIN_ROW_FILTER`).
- One shared mask helper (`Framework/RowFilter.py`), used at training (after windowing), at entry,
  and at threshold picking. So all three select the same bars.
- The filter enters the 030 model key only on the new mixin's classes.
- The threshold is picked at the controls' target rate (0.014735) with ineligible bars counted as
  0 (Clarification Q1: A).
- Train first, then pick, then backtest, one model at a time (research R7).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**:
- freqtrade (backtesting, `StrategyResolver`)
- MLX (`Predictors/MLXClassifierNary`, `NNNClassifierMLX` LSTM_RAW)
- numpy, pandas

**Storage**:
- model weights at 030 model-key locations under `saved_data/`
- arm records `regime/arms/*.json`
- plans and thresholds in `evaluation/folds/`
- research JSON in `research/candidate_training/`

**Testing**: pytest, with the suite as in quickstart §1, plus new tests in `Framework/tests/` and
`research/candidate_training/tests/`.

**Target Platform**: macOS on Apple Silicon (MLX), local.

**Project Type**: a research framework, the nested strategies repository.

**Performance Goals**: about 12 min training per fold, 30 folds ≈ 6 h, plus about 15 threshold
forward passes and 15 backtests per model.

**Constraints**:
- one trainer or backtest chain at a time;
- byte-identical when off;
- `Framework/` strategy-agnostic;
- no `MODEL_CODE_VERSION` bump;
- the live dry-run bot (pid 2661) untouched.

**Scale/Scope**: 2 models × 15 folds. GFM and MLX are out of scope.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / constraint | Status | How |
|---|---|---|
| I. Deployment regime | PASS | Uses the 2023–26 walk-forward, the current regime. The late half is confirmatory. |
| II. Persistence across distant eras | PASS by restriction | No distant era is possible: 730-day training plus data from 2021-01 means the first test is 2023-01. So no outcome is recorded as a FINDING. A pass is "PASS (PROVISIONAL)", a provisional result with a named forward-confirmation route (the post-2026-09-07 folds or dry-run). Per-third pp is reported as context (research R8). |
| III. Learnability ≠ edge | PASS | S1 (AUC) is only a mechanism gate. A pass needs P&L stages S2/S3 against the control and the width book. |
| IV. Parameters live | PASS | RowFilterReport per fold (kept share, SELL = 0), 0 entry violations, and simulator fidelity of at least 95% (R9). |
| V. Realistic execution | PASS | The same backtest config, liquidity cap and fees as the controls. |
| VI. Diagnose before tuning | PASS | Follows the diagnostic chain in `research/model_calibration/findings.md`. One change. |
| VII. Validate cheaply | PASS | Gated on the simulator screens (width_sizing: real within-width information on ARO/GBR). S1 gates before P&L is read. |
| VIII. One parameter space | PASS | Only the training rows change. The threshold follows by rule (same rate), not by tuning. |
| XI. No leakage | PASS | Class weights come from the kept train labels. Thresholds come from the training window only (`TrainWindow`). Scalers are unchanged and the feature list is unchanged, so no regeneration is needed. |
| Framework strategy-agnostic | PASS | Framework gets a generic `(column, op, value)` mask and reads `TRAIN_ROW_FILTER` by getattr. The gbb values live in `NNNC/_candidate_training.py`. |
| Params merge from parent | PASS | No params dict is redefined. The arm `.json` files are copied from the control snapshots. |
| No pair-specific code | PASS | — |
| Staging and commits | PASS | Named files only, committed to master. |
| Quality gates 1–7 | PASS / declared | Pinned timeranges (the fold plan), liquidity on, liveness, trade counts reported. Gate 2: see II. Gate 5: a stage passing by less than 1 pp is INCONCLUSIVE until a paired seed-2 repeat (research R8, task T021a). Gate 7 is not applicable (features unchanged). |

**Post-design re-check**: no violations. Principle II and gate 2 are satisfied by never recording
the outcome as a finding (PASS (PROVISIONAL) at most, with a forward-confirmation route).

## Project Structure

### Documentation (this feature)

```text
specs/032-candidate-only-training/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/row_filter.md
├── checklists/requirements.md
├── verification.md   # SC-001 spot check (T011)
└── tasks.md            # /speckit-tasks
```

### Source Code (nested repository `user_data/strategies`)

```text
Framework/
├── RowFilter.py                      # NEW: row_filter_mask (generic)
├── TrainingEngine.py                 # prepare_training_data applies the filter; report sidecar
└── tests/test_row_filter.py          # NEW: mask, parity-off fixture, filtered windows, model key
NNNC/
├── _candidate_training.py            # NEW: CandidateTrainingMixin (gbb filter, entry gate, add off)
├── NNNC_CO_ARO.py / .json            # generated by build_arms
└── NNNC_CO_GBR.py / .json
evaluation/
├── pick_threshold.py                 # keep-mask; fold_has_model for "trained?"
├── test_pick_threshold.py            # keep=None identical; keep zeros ineligible bars
└── folds/plan_co_aro.json, plan_co_gbr.json, thresholds_co_*.json
research/candidate_training/
├── build_arms.py, train.py, auc.py, judge.py
├── preregistration.json/.md          # committed before training
├── verdicts.json, report.md
└── tests/
research/nnnc_framework/signals.py    # add CO_ARO / CO_GBR plans
Framework/tests/test_exit_policy_single_source.py  # allowlist if a new script names exit params
```

**Structure Decision**: The existing nested repository layout. The generic code goes in
`Framework/`, the NNNC specifics in `NNNC/`, and the study in `research/candidate_training/`. The
threshold tool is changed in place.

## Complexity Tracking

No constitution violations to justify. Principle II and gate 2 are handled by restriction
(research R8): the result can be at most a provisional result, never a finding, until it is
confirmed forward.
