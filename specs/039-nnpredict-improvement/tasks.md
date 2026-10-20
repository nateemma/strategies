---

description: "Task list for 039 NNPredict improvement study"
---

# Tasks: NNPredict Improvement Study

**Input**: `specs/039-nnpredict-improvement/` (spec, plan, research R1–R10, data-model, contracts/cli.md, quickstart)

**Tests**: included. The spec requires tests for every fix (SC-001), plus causality and identity
proofs.

All paths are relative to `user_data/strategies/`. Runs happen from the freqtrade root with
`~/.freqtrade_env` loaded. One chain at a time, checking both bots after each arm. Never stage
`NNNC/NNNCStrategy.py`, `research/live_pnl/{drift,gate}.json`, `utils/Wavelets.py` or
`utils/test_fft.py`.

## Phase 1: Setup

- [X] T001 Create `research/nnpredict/__init__.py`, `research/nnpredict/tests/__init__.py`, and `NNPredict/tests/__init__.py` (if missing)
- [X] T002 [P] Run the existing suites as a baseline before any change: `pytest NNPredict/tests Framework/tests/test_model_key.py utils/tests -q`. Record the pass count

## Phase 2: User Story 1 — Fixes, proven by tests (P1)

**Independent test**: the new tests pass. Post-model settings share a model key, and a Ridge fold
passes the artifact check.

- [X] T003 [US1] Test `NNPredict/tests/test_model_key_classes.py`:
  - subclasses differing only in each of `entry_z`, `min_magnitude`, `rolling_window`, `magnitude_floor_mode`, `magnitude_pctile`, `mag_std_mult`, `entry_quantile`, `exit_quantile` have equal model keys;
  - differing in `target_mode`, `target_max_gain`, `atr_floor` or `LABEL_SMOOTH_WINDOW` gives different keys.

  Use `Framework.ModelKey` the way `Framework/tests/test_model_key.py` does, and confirm the test FAILS before T004
- [X] T004 [US1] Fix R1 in `NNPredict/NNPredictStrategy.py`: move the 8 post-model names to `MODEL_KEY_IGNORED`. Then T003 and `Framework/tests/test_model_key.py` pass
- [X] T005 [US1] R2: read `evaluation/run_folds.py` (`model_exists`, `fold_has_model`) and `Framework/ModelKey.locate*`. Write a test `evaluation/test_run_folds_joblib.py` showing that a fold directory holding only `*.joblib` counts as a trained model, then make the minimal family-agnostic fix so the test passes (only if one of the checks rejects `.joblib`)
- [X] T006 [US1] Rewrite `NNPredict/README.md` (FR-003) to cover:
  - the z-score + magnitude-floor gate;
  - the targets (point / excursion / stop_aware);
  - backends, including the new LSTM_RAW;
  - the inputs;
  - status (never on the harness before 039; 038's evidence);
  - a pointer to this study

---

## Phase 3: User Story 3 — model variants: code (P2, built before the model-stage run)

**Independent test**: each variant trains on a small window and gives finite, non-constant
predictions. Every input column is causal.

- [X] T007 [P] [US3] Add `RegressorTypeMLX.LSTM_RAW` to `NNPredict/NNPredictRegressorMLX.py`: an `nn.Module` of features → `nn.LSTM(num_features, H)` (last time step) → `nn.Linear(H, 1)`. Mirror `NNNClassifierMLX_LSTM_Raw`'s hidden size, and reuse the file's existing `MLXRegressor` subclass pattern
- [X] T008 [P] [US3] Write `NNPredict/_inputs.py`:
  - `Raw8InputsMixin` adds the raw8 columns (`NNNC/_raw8_features.add_raw8_features`);
  - `LaggedReturnsMixin` adds `lret_{8,16,32,64,96}` and `rvol_{16,96}`;
  - every new column goes through a causal rolling normalisation (a trailing 2880-candle z-score, clipped to ±5, divided by 5) and is registered in `include_list` and `pre_normalized_columns`;
  - `include_list` is set per class: raw = raw8 + `current_gain`, `recent_gain`; raw+ind = the parent's list + raw8; raw+ret = raw8 + lagged + gains.

  Also make sure `rolling_dataframe_normalise` works when no column needs the fitted scaler, or keep the 6 scaled columns and drop them from the model input, whichever is minimal. Verify on SOL
- [X] T009 [P] [US3] Test `NNPredict/tests/test_inputs.py`:
  - truncation: every new column is identical up to a cut when recomputed on truncated data;
  - each class's `include_list` is what research R4 lists;
  - all new columns are finite after the warm-up;
  - the three LSTM classes and `NNPredict_Ridge` resolve to four distinct model keys
- [X] T010 [US3] Strategy classes:
  - `NNPredict/NNPredict_LSTM_Raw.py`, `NNPredict_LSTM_RawInd.py` and `NNPredict_LSTM_RawRet.py`: MLX `LSTM_RAW`, with the matching input mixin;
  - `NNPredict/NNPredict_Ridge_SA.py`: `NNPredict_Ridge` with `target_mode = "stop_aware"`.

  Each declares nothing else, merges parameters from its parent, and carries the same `.json` params as `NNPredict_Ridge` if one exists (otherwise none)
- [X] T011 [US3] Smoke test: train `NNPredict_LSTM_Raw` on 30 days of SOL in a scratch run (`test_strat.sh` training mode) and confirm it saves weights and gives non-constant predictions. Record the training time (R10)

---

## Phase 4: User Story 2 + 3 — model-stage run (P1/P2)

- [X] T012 [US2] Fold plans `evaluation/folds/plan_{nnp_ridge,nnp_ridge_sa,nnp_lstm_raw,nnp_lstm_ind,nnp_lstm_ret}.json`: the plan_cs_aro shape, family `NNPredict`, `base_strategy` the class, distinct `run_id`s, arm_tag `<tag>_fNN`
- [X] T013 [US2] Write `research/nnpredict/preregistration_models.json`:
  - arms and comparators (data-model);
  - rule maps from `research/holt_trigger/preregistration.json`;
  - min trades 200/100;
  - the R7 selection rule verbatim;
  - instrument: all 15 records present, and trades in both halves (else INCONCLUSIVE);
  - "seen before": 038, NNPredict memory, no harness result for NNPredict.

  Write `research/nnpredict/judge.py` (the `models` stage: instrument = all 15 records, plus a check per arm that trades are not 0; `judge_one` vs `nnp_ridge`; the R7 selection; output `verdicts_models.json`; baseline reported beside `w_aro`/`cs_aro`/`st_g24_mom` pooled totals) and its test `research/nnpredict/tests/test_judge.py` (R7 on synthetic verdicts; refusal when the pre-registration is dirty).

  Commit T001–T013 (named files) before any fold runs
- [X] T014 [US2] Run `nnp_ridge` on f00 only first, and confirm training, the artifact check and evaluation all succeed. Then run the full chain of the 5 arms (Ridge arms first, then the 3 LSTM arms), checking the bots after each
- [X] T015 [US2] Run `judge models` and record the R7-selected model `<M>`

---

## Phase 5: User Story 4 — framework port (P1)

**Independent test**: `<M>_fw0` trades are identical to `<M>` on 15/15 folds.

- [X] T016 [US4] Port: `NNPredictStrategy(EntryFrameworkMixin, BaseNNStrategy)`, `ENTRY_MODE = "model"`, `MODEL_ENTRY_FAMILY = None`, `HELD_PAIR_ADD = None`. Test `NNPredict/tests/test_framework_port.py`:
  - on a synthetic frame, `get_entry_conditions` equals `BaseNNStrategy`'s mask exactly;
  - `enter_tag` is unchanged;
  - the model key is unchanged by the port (framework names are IGNORED)
- [X] T017 [US4] Write `research/nnpredict/build_arms.py framework`: from `verdicts_models.json` `<M>`, write the arm classes `<M>_FW0` (no declaration), `_G24` (sell_params merged with `stoploss_grace_hours` 24.0), `_HPA` (`HELD_PAIR_ADD = {"in_profit_only": True}`), `_FAMM` (`MODEL_ENTRY_FAMILY = "momentum"` + `FAMILY_EXIT_OVERRIDES` momentum set) and `_GATE` (`ENTRY_MODE = "trigger_and_model"`, `TRIGGER_NAME = "gbb"`), plus their plans. Each arm reuses `<M>`'s weights: assert that the model keys are equal
- [X] T018 [US4] Write `research/nnpredict/preregistration_framework.json` (identity rule: equal `trades_hash` on 15/15; feature arms vs fw0; PASS → candidate default; liveness: an arm with `folds_changed == 0` is INERT, a defect, with no verdict; seed robustness: a PASS whose whole-period gain is < 1 pp needs a seed-2 rerun of the arm and its fw0 that also PASSes) and the `judge.py framework` stage (identity first; refuses feature verdicts if identity fails; applies liveness). Commit before running fw0
- [X] T019 [US4] Run fw0, then the 4 feature arms (backtest only), then `judge framework`. For any PASS under 1 pp, run the seed-2 pair (`run_folds --seed 2`, which trains new seed-2 weights) and judge again

---

## Phase 6: User Story 5 — sizer (P1)

- [X] T020 [US5] `research/nnnc_framework/signals.py`: add the plan entry `NNP` → `<M>`'s plan (or the fw0 plan) and keep `%predict_gain` as `predict_gain` in the table when present. Test that the existing models' tables are unchanged (the column is absent there)
- [X] T021 [US5] Write `research/nnpredict/screen.py` (R8) and `research/nnpredict/preregistration_sizer.json` (the pass rule: gap ≥ 0.05, partial > 0, ≥ 5/8 sign folds; early half only; the sizer arm and its comparators). Test `research/nnpredict/tests/test_screen.py`: gap and partial on synthetic trades. Commit before the screen runs
- [X] T022 [US5] Run the signals for NNP (strict mode), then the screen
- [X] T023 (screen FAILED both directions: no sizer arm, per the pre-registration) [US5] Only if the screen passes: `build_arms.py sizer` writes `NNPredict/NNP_CS.py` (`WidthSizerMixin, WidthBookMixin, <M>_FW0`, `AGREE_RULE = "p_buy"`, `sizer_score` = `%predict_gain`, `WIDTH_CUTS` as `cs_aro`), with per-fold thresholds at 0.5 agree share over each fold's TRAINING-window width-book signal rows. Use `research.control_sizer.pick` if it accepts a score column; otherwise extend it minimally (score column parameter, tested, default unchanged). Then run `nnp_cs` and `judge sizer`; a PASS under 1 pp needs a seed-2 rerun that also PASSes

---

## Phase 7: Polish

- [X] T024 Write `research/nnpredict/report.md`: one table of every arm (per half vs comparator), the identity result, the screen, the recommendation (FR-012), prediction spread per arm, and limitations
- [X] T025 (nothing changed: the framework pre-registration makes adoption of the two PASSes -- gbb gate, 24 h grace -- the user's call; report.md recommendation) Adopt (only what PASSed and the rule says): set the recommended default on `NNPredictStrategy` / `<M>`, recorded in the report. If nothing passed, the framework port stays with everything off
- [X] T026 [P] Run all tests (`NNPredict/tests`, `Framework/tests`, `utils/tests`, `evaluation/test_*.py`, `research/nnpredict/tests`); all green
- [X] T027 Update AGENT_GUIDE (a 039 row) and memory (project_039 file, MEMORY.md line). Commit the named files and push

## Dependencies

- Setup → US1 (T003–T006) → US3 code (T007–T011) → the models run (T012–T015) → the framework (T016–T019) → the sizer (T020–T023) → polish.
- T016's port can be coded alongside Phase 3, but fw0 needs `<M>` (T015).
- Runs never go in parallel.

## Implementation strategy

- **MVP:** US1 fixes + the `nnp_ridge` baseline on the harness. That alone makes NNPredict comparable.
- **Then:** the model variants → the framework → the sizer, each stage behind its own committed
  pre-registration.


## Execution notes (2026-10-20)

- T014/T015 ran as the operating-point amendment (preregistration_op.json) after the as-is Ridge made 152 trades; the threshold amendment (preregistration_threshold.json) added flip / excursion / H24 at the user's request. R7 over all eight: nnp_lstm_ind_op.
- T026: 039's tests all pass. Pre-existing failures outside 039 remain in Framework/tests (exit-policy allowlist: NNNC_W_ARO_TP2, _atr14_take_profit, SimpleTrigger_MofN4of8_C2, research/tb_model/*; model-key audit: MIX_KNOTS/_knots/HGB_SIGNALS, _apply_atr_take_profit, _filtered).
- Results: research/nnpredict/report.md.
