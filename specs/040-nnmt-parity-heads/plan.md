# Implementation Plan: NNMT Parity and Head Study

**Branch**: `040-nnmt-parity-heads` (spec directory; work is committed to master) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/040-nnmt-parity-heads/spec.md`

## Summary

Six stages, each judged on 15-fold walk-forward P&L, or on a screen, under rules committed before
their results:

1. **Port the NNNC entry framework** to `NNMTStrategy` with everything off (R1) and clean up the model
   key (R2). Prove the trades are identical against a pre-port baseline (`mt` → `mt_fw0`, 15/15).
2. **Score NNMT** at its defaults (`mt`) and at ARO's operating point (`mt_op`), reporting the share
   of entries the model decides (R4).
3. **Add a multi-task plain LSTM** (`LSTM_RAW`) and causal raw8 inputs. Score four variants: raw8 in
   the current network, and the plain LSTM on the indicators, on raw8, and on both. Select one by
   rule (R6, R7). Run its trading-head-only twin.
4. **Test NNNC's adopted features** on the selected model: 24 h grace, in-profit add, gbb gate and
   C2 take-profit (R5). Screen its p_buy as a width-book sizer, with an arm only if the screen
   passes (R9).
5. **Train a heads-trained twin** (every auxiliary head on), screen each head for skill over a
   causal baseline and for trade relevance, and run a pre-declared arm per passing head (R10).
6. **Define a causal regime-transition target** (UP/DOWN at 1, 3 and 7 days; pair and BTC) and
   truncation-test it. Screen it with a cheap per-fold model against base-rate and trailing-trend
   baselines plus a P&L proxy. Run trading arms (gate, scaling, standalone long, risk-off exit) only
   for passing cells, and an NNMT-head version only if the stand-in passes (R11, R12).

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`)

**Primary Dependencies**:
- freqtrade (the repo's checkout) and MLX (NNMT's multi-task network);
- scikit-learn (the HGB stand-in), numpy/pandas, scipy;
- the existing harness:
  - `evaluation/foldplan.py`, `evaluation/run_folds.py`;
  - `research/nnnc_switches/judge.py` (`judge_one`);
  - `research/nnnc_framework/signals.py`;
  - `research/new_inputs/trades.py`;
  - `research/indicator_screen/` (`fit_fold`, `causality`);
  - `research/control_sizer/pick.py`;
  - `NNNC/_width_book.py`, `NNNC/_mix_sizer.py`.

**Storage**:
- trained models under `saved_data/`;
- fold records in `regime/arms/` (local, not committed);
- per-fold signal and prediction tables under `research/nnnc_framework/signals/` and
  `user_data/data/nnmt_*` (local);
- pre-registrations, verdicts and the report in `research/nnmt/`.

**Testing**: pytest:
- `NNMT/tests/`: model key, framework identity, `LSTM_RAW` shapes, head-probability capture;
- `research/nnmt/tests/`: transition label causality and base rates, screen metrics;
- `Framework/tests/test_model_key*.py`, `utils/tests`.

**Target Platform**: macOS on Apple silicon, MLX GPU for training.

**Project Type**: strategy-family changes plus a research harness.

**Performance Goals**:
- NNMT fold training is timed on f00 before any chain;
- about 135 MLX fold trainings plus seed-2 reruns, run sequentially (R14).

**Constraints**:
- never touch the bots (pid 2661 momentum, pid in `user_data/logs/live_nnnc.pid`);
- one trainer at a time (MLX or HGB);
- parameter dicts merge from their parents;
- no pair-specific code;
- `Framework/` stays family-agnostic, and the shared `Predictors/MLXClassifierMultiTask` is unchanged;
- the user's uncommitted files are never staged.

**Scale/Scope**: 15 folds per arm.
- **Model stage:** 1 baseline, 1 identity arm, 1 operating-point arm, 4 variants.
- **Framework stage:** 4 feature arms and 1 twin.
- **Sizer:** 1 arm (conditional).
- **Heads:** 1 heads-trained model, ≤ 5 head arms (conditional).
- **Transition:** 18 screen cells, ≤ 8 arms, 1 head version (all conditional).

## Constitution Check

| Principle | Compliance |
|---|---|
| I. Deployment regime | Verdicts need the late half (f08–f14) not worse (judge_one rule maps). Transition screens need both halves |
| II. Persistence | Both halves. No distant era, so PROVISIONAL, as since 023 (declared limitation) |
| III. Learnability ≠ edge | Model and arm verdicts are on P&L only. AUC is used ONLY to eliminate (screens), never to accept. Every screen pass leads to a P&L arm before anything is called useful (FR-013) |
| IV. Liveness | The identity arm proves the port is inert. Every feature, head and transition arm must change trades on at least one fold, or it is INERT (a defect). The head-probability capture is checked against the argmax columns NNMT already writes |
| V. Realistic execution | The harness's fillable cap and fees, as for all books. Capacity waits for 036's cost model. The transition P&L proxy subtracts a 30 bp round trip |
| V. End-to-end lookahead | The transition label and the stand-in inputs pass a truncation test (SC-005, R11) |
| VI. Diagnose before tuning | No threshold or task-weight tuning. Model-decided share, head base rates and transition base rates are reported before any arm |
| VII. Validate cheaply | The sizer arm is gated behind its screen, head arms behind two screens, and the transition head retrain behind a CPU stand-in screen. Fold training is timed on f00 first |
| VIII. One parameter space | Each feature arm changes one thing against the selected model's fw0. Head and transition arms each change one use |
| IX. Orthogonality | The sizer and head scores' correlation with `cs_aro`'s p_buy is reported. Transition signals are judged against `cs_aro`, which already holds the width/ARO axis |
| X. No per-pair overrides | None. The BTC level is a market input applied to every pair |
| XI. No leakage | Models are trained per fold on its training window (run_folds). The training dataframe ends at `test_start`, so labels at its end are truncated, not filled from test bars (a purge ≥ the 48-bar label horizon; verified on f00, R4). New inputs use causal rolling normalisation, with no fitted artifact. The HGB stand-in purges H bars at each training end. Percentile thresholds come from the training tail only. The shared 6-column `main_scaler.pkl` is a pre-existing exposure, identical across arms (035 R1), and is recorded |
| Gates 1–7 | (1) Pinned folds. (2) Halves: PROVISIONAL. (3) Harness execution. (4) Liveness (IV). (5) Seed 2 for any PASS, the selected model, and any delta < 1 pp. (6) Trade counts and model-decided share reported; zero trades = INCONCLUSIVE. (7) New input sets add no fitted scaler (causal normalisation), so there is nothing to regenerate. The 6-column scaler is unchanged |

**Result**: PASS, with the declared limitations of 033–039 (gate 2; the shared-scaler exposure). Two
justified exceptions are in Complexity Tracking.

**Post-design re-check (after Phase 1)**: unchanged. The design adds no fitted artifact and no
shared-framework code. Every new decision rule is pre-registered (R13).

## Project Structure

### Documentation (this feature)

```text
specs/040-nnmt-parity-heads/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/cli.md
├── checklists/requirements.md
└── tasks.md            (/speckit-tasks)
```

### Source Code

```text
NNMT/
├── NNMTStrategy.py          R1 EntryFrameworkMixin (all off); R2 key classification;
│                            R8 head-probability capture (_head_probs)
├── NNMTClassifierMLX.py     R6 ClassifierTypeMLX.LSTM_RAW (no preamble, linear heads)
├── _nnmt_operating_point.py R4 ARO operating point (merged buy params); _nnmt_inputs.py R6 input sets
├── _transition.py           R12 regime-head retarget with the R11 label (conditional)
├── NNMT_LR.py, NNMT_LR_Raw.py, NNMT_LR_RI.py, NNMT_Raw.py    R6 variants
├── README.md                stale reference fixed; status at the end
└── tests/                   model key, identity wiring, LSTM_RAW, head capture
research/nnmt/
├── preregistration_{models,framework,screens,transition}.json
├── build_arms.py            plans + arm classes (models / framework / sizer / heads / transition)
├── judge.py                 identity / models (+R7) / framework / sizer / heads / transition
├── screen.py                R9 sizer screen and R10 screen 2 (trade relevance)
├── head_screen.py           R10 screen 1 (skill vs causal trailing baselines)
├── transition.py            R11 labels, base rates, truncation, stand-in fits, proxy, tables
├── report.md, verdicts_*.json, screens_*.json
└── tests/
research/nnnc_framework/signals.py   R8 NNMT plan entries and p_<task>_<class> columns
evaluation/folds/plan_mt*.json       fold plans (family NNMT)
```

**Structure Decision**:
- Strategy code stays in `NNMT/`. Input sets are imported from `NNPredict/_inputs.py`, and the width
  book and sizer mixins from `NNNC/`, as 039 imported across families.
- Study scripts live in `research/nnmt/`. They copy 039's patterns rather than editing
  `research/nnpredict/`, so the committed 039 instruments are untouched.

## Complexity Tracking

| Exception | Why | Rejected alternative |
|---|---|---|
| `LSTM_RAW` added to `NNMTClassifierMLX`, not `NNNClassifierMLX` (constitution: new MLX types go in NNNClassifierMLX) | `NNNClassifierMLX` is single-task. The multi-task heads and loop live in `NNMTClassifierMLX`, which already holds NNMT's 14 types. The pattern (enum member + model class + factory) is followed there. 039 precedent: `NNPredictRegressorMLX` | A multi-task path in `NNNClassifierMLX`: a parallel framework inside it |
| Shared `main_scaler.pkl` reused by every fold (Principle XI exposure) | Identical across all arms and comparators. A per-fold refit would void every existing comparator (035 R1) | Per-fold scalers for all arms: retrains the comparators, out of scope |
| R7 selects with both halves in view | A single model must carry the later stages. The sizer screen keeps its late half blind | Separate selection data: none exists (no distant era) |
