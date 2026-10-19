# Research: 039 NNPredict Improvement Study

Decisions taken while planning, from reading the code on 2026-10-19. Each gives its reason and the
alternatives rejected.

## R1 — Model-key classification (FR-001)

**Decision**:
- **Moved out of `NNPredictStrategy.MODEL_KEY_ATTRIBUTES` into `MODEL_KEY_IGNORED`:** `entry_z`,
  `min_magnitude`, `rolling_window`, `magnitude_floor_mode`, `magnitude_pctile`, `mag_std_mult`,
  `entry_quantile`, `exit_quantile`.
- **Kept in the attributes:** `target_mode`, `target_max_gain`, `target_max_loss`,
  `LABEL_SMOOTH_WINDOW` and `atr_floor`.

**Rationale**:
- The eight moved names are read only in `get_predictions`, after `classifier.predict` (lines
  699–764).
- `atr_floor` divides the labels (lines 233, 256, 384, 477), so it shapes the weights.

**Test**: two classes differing only in `entry_z` give equal keys; two differing in `target_mode` give
different keys. Uses the `Framework/tests/test_model_key.py` pattern, and that audit test must still
pass.

## R2 — Ridge artifact check (FR-002)

**Facts**:
- `RidgeRegressor` rewrites `.safetensors` to `.joblib` (`Predictors/RidgeRegressor.py:97`).
- The fold runner checks a trained model in two places:
  - `model_exists(cls)`, which globs `saved_data/<cls>/*.safetensors`;
  - `fold_has_model()`, which uses `ModelKey.locate_for`.

**Decision**: read both checks during implementation. Whichever one only accepts `.safetensors`
gains `.joblib` (a family-agnostic change: "any saved weight file the predictors write"). A Ridge
fold must pass end to end on f00 before the baseline runs.

**Outcome (T005)**: no fix is needed. The training gate is `fold_has_model`, which goes through
`ModelKey.weights_exist`, and `WEIGHT_SUFFIXES` already includes `.joblib`. `model_exists`
(safetensors-only) only feeds `train_fold`'s return value, which the runner ignores. This is locked in
by `evaluation/test_run_folds_joblib.py`.

## R3 — Harness and comparators

**Decision**:
- **Plans:** built with `evaluation/foldplan.py` (or copied from the `plan_cs_aro.json` shape): family
  `NNPredict`, 15 anchored folds, `config_wf`, seed 1.
- **Runs:** `evaluation/run_folds.py`, which writes one record per fold under `regime/arms/`.
- **Verdicts:** `research.nnnc_switches.judge.judge_one(pr, arm, control)`, the same rule maps as
  every study since 029.
- **Comparators** (existing records, no rerun): `w_aro` (width book), `cs_aro` (NNNC reference book),
  `st_g24_mom` (momentum reference).

## R4 — Model variants (FR-005; user request)

**Decision**: a regression twin of NNNC's `LSTM_RAW` (ARO's architecture: features → LSTM → linear
head, no resize or conv), added to `NNPredict/NNPredictRegressorMLX.py` as `RegressorTypeMLX.LSTM_RAW`
and trained by the existing `MLXRegressor` loop. Same target, H, gate and exits as the baseline. The
three input sets:

| Arm | Inputs (`include_list`) |
|---|---|
| `nnp_lstm_raw` | raw8 (`NNNC/_raw8_features.py`: ret1, ret4, log_hc, log_lc, log_oc, range, logvol_z, vol_ratio) + NNPredict's own `current_gain` and `recent_gain` |
| `nnp_lstm_ind` | NNPredict's current inputs (the 24-indicator include list + `current_gain`, `recent_gain`, `atr_norm`, `ema_fast_norm`) + raw8, i.e. "raw + indicators" |
| `nnp_lstm_ret` | raw8 + lagged returns (038 set L: log returns over 8/16/32/64/96 candles, realised vol over 16/96) + `current_gain`, `recent_gain` |

**Normalisation**: every new column is made causal-normalised in its builder (the 035 R1 rule: a
trailing 2880-candle z-score clipped to ±5, divided by 5) and declared in `pre_normalized_columns`.
No new fitted scaler, so no artifact can see test candles (Principle XI). The shared 6-column
`main_scaler.pkl` stays as it is for every arm: a pre-existing exposure, identical across arms
(035 R1).

**Alternatives rejected**:
- More capacity, multi-horizon heads, label smoothing: recorded as failed (memory).
- Keras: MLX is the production backend.

## R5 — Baseline and the `stop_aware` arm

**Decision**:
- **Baseline `nnp_ridge`:** `NNPredict_Ridge` as is (point target, H = 4, gate z ≥ 1.0 and magnitude
  ≥ 0.10, grace 2.6 h).
- **`nnp_ridge_sa`:** the same with `target_mode = "stop_aware"`, judged against `nnp_ridge`.
- **Coeff family excluded:** `utils/Wavelets.py` has uncommitted user edits, and 038 found wavelet
  inputs add nothing.

## R6 — Framework port (FR-006, FR-007)

**Decision**:
- `NNPredictStrategy(EntryFrameworkMixin, BaseNNStrategy)` with `ENTRY_MODE = "model"`,
  `MODEL_ENTRY_FAMILY = None` and `HELD_PAIR_ADD = None`. Every framework feature is off, so the
  entry is exactly `BaseNNStrategy.get_entry_conditions` (the model's `predict_buy`) and tags stay
  BaseStrategy's.
- **Identity proof:** re-run the baseline as `nnp_ridge_fw0` and require an equal `trades_hash` on
  all 15 folds.
- **Feature arms:** run on the model chosen by R7, each judged against that model's own fw0. All of
  these act after the model, so they reuse the trained weights (the framework's attributes are
  model-key IGNORED):

| Arm | Change |
|---|---|
| `_g24` | `stoploss_grace_hours` 24.0 (sell_params merged) |
| `_hpa` | `HELD_PAIR_ADD = {"in_profit_only": True}` |
| `_famm` | `MODEL_ENTRY_FAMILY = "momentum"` with the ARM_D_FAMILY momentum exit set declared as `FAMILY_EXIT_OVERRIDES = {"momentum": {"take_profit": 10.0, "unclog_12h_hours": 24.0}}` (`SimpleTrigger/presets.py`). ARM_D_FAMILY's reversion set is empty, i.e. today's shared exits, so a reversion arm would be identical to fw0 and is not run |
| `_gate` | `ENTRY_MODE = "trigger_and_model"`, `TRIGGER_NAME = "gbb"` (the model must fire on a gbb candidate) |

## R7 — Which model the framework and sizer arms use

**Decision (pre-registered)**: the model arm (`nnp_ridge`, `nnp_ridge_sa`, `nnp_lstm_*`) with the
highest pooled total over f00–f14 among those with no WORSENED half against `nnp_ridge`. Ties go to
`nnp_ridge` (cheapest).

**Caveat**: this selection uses both halves, so later arms' halves are not blind. Labelled
PROVISIONAL, as everything since 023.

## R8 — Sizer screen and arm (FR-008)

**Signal tables**: `research/nnnc_framework/signals.py` gains a plan entry for the selected model and
keeps `%predict_gain` as the score column. It runs one forward pass of each fold's trained weights
over the fold's test window, in strict mode, with no training.

**Screen** (`research/nnpredict/screen.py`): on the 412 early width-book trades
(`research.new_inputs.trades.trades()`, which reproduces 412 trades and the 0.1004 control gap):
- the score's within-fold top-half vs bottom-half pp gap (the `cs_aro` analogue at share 0.5);
- the partial Spearman with trade pp given bb_width;
- sign folds.

**Pre-registered pass**:
- gap ≥ 0.05 pp per trade (half of ARO's 0.1004);
- partial > 0;
- at least 5 of 8 folds with that sign.

Early half only, so **the late half stays blind for the sizer arm**.

**Sizer arm `nnp_cs`**:
- **Class:** `WidthSizerMixin, WidthBookMixin` over the selected NNPredict class (framework on),
  `AGREE_RULE = "p_buy"`, `sizer_score` = `%predict_gain`.
- **Threshold:** per fold, the 0.5 agree-share threshold from the fold's training window
  (`research.control_sizer.pick`).
- **Verdicts:** `judge_one(nnp_cs, w_aro)` must PASS; `judge_one(nnp_cs, cs_aro)` is reported.

## R9 — Pre-registrations

Three files under `research/nnpredict/`, each committed before its own results:

| File | Covers | Committed |
|---|---|---|
| `preregistration_models.json` | baseline report, model arms, the R7 rule | before any 039 fold trains |
| `preregistration_framework.json` | the identity rule, the feature arms | before fw0 runs (it names the R7 model by rule, not by name) |
| `preregistration_sizer.json` | the screen pass rule, the sizer arm | before the screen runs |

**Rule maps** are copied from `research/holt_trigger/preregistration.json` (paired and risk maps),
with min_control_trades 200 and min_arm_trades 100.

## R10 — Compute and order

- **Ridge folds:** seconds to train; about 3–5 min to evaluate per arm.
- **MLX LSTM folds:** training time measured on f00 first, then extrapolated. With strict ordering,
  only one training job runs at a time.
- **Framework arms:** backtest only (no training).
- **Expected wall time:** roughly 4–10 h, dominated by 3 × 15 LSTM trainings. It runs as background
  chains with a check that both bots are still up.
