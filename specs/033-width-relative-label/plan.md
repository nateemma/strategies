# Implementation Plan: Width-Relative Label for NNNC Models

**Branch**: `033-width-relative-label` (spec directory; committed to master) | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/033-width-relative-label/spec.md`

## Summary

Retrain ARO and GBR on a label that asks whether a rule candidate did better than bars of similar
band width:
- the gbb_path trade outcome, ranked within the fold's training-window width deciles, with the top
  half labelled BUY;
- training on eligible rows only (032's filter).

The model is then used as a capital-neutral stake sizer on the width book: 1.33× when it agrees,
0.67× when it does not. That arm is judged in pre-registered stages against the width book and the
24 h controls.

**Technical approach:**
- A generic pooled-relabel hook in `TrainingEngine._train_keyed` (`Framework/PooledLabel.py`),
  declared as data (`POOLED_LABEL`).
- A generic close-based `path_outcome` in `TrainingSignals` (R1, R2).
- A median split per bucket with a speed tie-break instead of "above the mean". The diagnostic in
  R3 shows the mean split would keep the label width-correlated.
- New NNNC mixins: `WidthRelativeMixin` (label and filter, in the model key) and `WidthSizerMixin`
  (tag-based stake weights and a 4-decimal threshold), on top of the existing `WidthBookMixin`.
- The thresholds give a 50% agree share on the width-book candidates of the validation tail
  (R6, R7).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**:
- freqtrade (backtesting, `StrategyResolver`, `custom_stake_amount`)
- MLX (LSTM_RAW / raw8 classifiers)
- numpy, pandas, scipy (Spearman)

**Storage**:
- weights at new 030 model-key locations, with `width_relative.json` and `row_filter.json` beside
  them;
- plans and thresholds in `evaluation/folds/`;
- arm records in `regime/arms/`;
- research outputs in `research/width_relative/`.

**Testing**: pytest. New tests in `Framework/tests/test_pooled_label.py`,
`research/width_relative/tests/`, and `evaluation/` if the threshold picker changes.

**Target Platform**: macOS on Apple Silicon, local.

**Project Type**: a research framework, the nested strategies repository.

**Performance Goals**:
- ARO training about 12 min per fold, GBR about 1–2 min per fold, so about 3.5 h in total;
- 30 single-fold picks;
- 2 × 15 backtests of about 13 s each.

**Constraints**:
- one chain at a time, with picks one fold per process;
- byte-identical when off;
- `Framework/` strategy-agnostic;
- the live bot untouched;
- no default promoted.

**Scale/Scope**: ARO and GBR × 15 folds. GFM/MLX, the filter arm and the entry-source arm are out of
scope.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / constraint | Status | How |
|---|---|---|
| I. Deployment regime | PASS | The 2023–26 walk-forward. The late half is confirmatory. |
| II. Distant eras | PASS by restriction | No distant era exists for NNNC models. The outcome is at most PASS (PROVISIONAL), with a forward-confirmation route; it is never recorded as a finding. |
| III. Learnability ≠ edge | PASS | S0/S1 are gates on the label and the mechanism. The verdict needs P&L (S2/S3) against the width book and the control. |
| IV. Parameters live | PASS | The relabel report per fold, the 4-decimal threshold load check, the flat-stake twin hash, tag-to-agree 0 mismatches, and the stake shortfall. |
| V. Realistic execution | PASS | The same config, fees and liquidity cap. The larger stakes go through BaseStrategy's fillable cap, and any shortfall is reported. |
| VI. Diagnose before tuning | PASS | R3's per-decile hit rates set the median-split design before any build. |
| VII. Validate cheaply | PASS | S1 (simulator, identical trades) gates the P&L read. The f00 smoke trains one fold first. |
| VIII. One parameter space | PASS | Only the label changes. The 50% share and the 1.0 : 0.5 weights are fixed in advance, not tuned. |
| IX. Orthogonality | PASS (design goal) | The label is built to be uncorrelated with width (S0 measures it). |
| XI. No leakage | PASS | Buckets come from training-window frames only. Thresholds come from the validation tail. Scalers are unchanged (features unchanged). |
| Framework strategy-agnostic | PASS | `PooledLabel` and `path_outcome` are generic, and the NNNC values are on the mixin. |
| Params merge / no pair code | PASS | No params dict is redefined. The arm `.json` files are copied from the control snapshots. |
| Quality gates | PASS / declared | Pinned fold plan, liveness evidence, trade counts. Gate 5: margins under 1 pp → paired seed-2 repeat. |

**Post-design re-check:** no violations. One spec refinement is needed and is applied with this
plan: FR-003/FR-004 change from "mean plus margin" to "the bucket's median split with a speed
tie-break" (R3).

## Project Structure

### Documentation (this feature)

```text
specs/033-width-relative-label/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/pooled_label.md, contracts/width_sizer.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code (nested repository `user_data/strategies`)

```text
Framework/
├── PooledLabel.py                 # NEW: relabel(frames, labels, spec) -> labels, report (generic)
├── TrainingSignals.py             # + path_outcome(df, target, stop, horizon)
├── TrainingEngine.py              # _train_keyed: POOLED_LABEL hook; width_relative.json sidecar
└── tests/test_pooled_label.py     # NEW
NNNC/
├── _width_relative.py             # NEW: WidthRelativeMixin (label spec, filter + %wr_valid, key)
├── _width_sizer.py                # NEW: WidthSizerMixin (tag, stake weights, 4-dp threshold)
├── NNNC_WR_ARO.py / .json         # generated
└── NNNC_WR_GBR.py / .json
research/width_relative/
├── build_arms.py, train.py, pick.py (validation-tail agree share), instrument.py,
│   separation.py (S1), judge.py (032 pattern)
├── preregistration.json/.md       # committed before training
├── verdicts.json, report.md
└── tests/
research/nnnc_framework/signals.py # + WR plans (runtime registration, as 032)
Framework/tests/test_exit_policy_single_source.py  # allowlist if a script names exit params
```

**Structure Decision**: generic code in `Framework/`, NNNC specifics in `NNNC/`, and the study in
`research/width_relative/`. 032's training driver, run-match and judge patterns are reused rather
than generalised. This keeps edits surgical.

## Complexity Tracking

No violations to justify.
