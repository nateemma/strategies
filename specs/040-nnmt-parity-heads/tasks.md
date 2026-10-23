---

description: "Task list for 040 NNMT parity and head study"
---

# Tasks: NNMT Parity and Head Study

**Input**: `specs/040-nnmt-parity-heads/`: spec, plan, research R0–R14, data-model, contracts/cli.md,
quickstart.

**Tests**: included. The spec requires proofs: the identity proof (SC-001), the model-key
classification (FR-002) and the transition truncation test (SC-005).

**Conventions**:
- All paths are relative to `user_data/strategies/`. Runs happen from the freqtrade root with
  `~/.freqtrade_env` loaded.
- One chain at a time (MLX training or HGB fits), checking both bots (pid 2661 and
  `user_data/logs/live_nnnc.pid`) after each arm.
- Stage only named files. Never stage the user's files (`NNNC/NNNCStrategy.py`-style WIP, untracked
  fold files).
- Every pre-registration is committed BEFORE the results it judges exist.

**Story order differs from priority.** The baseline `mt` (US2) must be recorded BEFORE the port
(US1) lands, so the identity proof has a pre-port reference. The phases below are in execution order.

## Phase 1: Setup

- [X] T001 Create `research/nnmt/__init__.py`, `research/nnmt/tests/__init__.py` and `NNMT/tests/__init__.py` (if missing)
- [X] T002 [P] Run the existing suites as a baseline before any change: `pytest NNMT Framework/tests utils/tests -q`. Record the pass count and the pre-existing failures in `research/nnmt/baseline_tests.txt`
- [X] T003 [P] Fix the stale `NNMT/README.md` reference to `NNMT/BaseNNMTStrategy.py` (the class is `NNMTStrategy` in `NNMTStrategy.py`). Full rewrite in T080

---

## Phase 2: Foundational (blocks every story)

**Purpose**: the harness pieces every stage uses. No behaviour change to `NNMT_MLX`.

- [X] T004 Write `research/nnmt/judge.py` scaffolding by copying 039's `research/nnpredict/judge.py` pattern:
  - `refusals` via `research.nnnc_switches.judge`;
  - `judge_one` wrappers;
  - an `identity(arm, ref)` stage that compares `trades_hash` per fold over 15 folds;
  - stage dispatch for `identity|models|framework|sizer|heads|transition`;
  - a `model_share(tag)` helper reading signal tables (data-model § model-decided share).

  Leave `research/nnpredict/` untouched
- [X] T005 Write `research/nnmt/build_arms.py` scaffolding: plan writing through `evaluation/foldplan.py` (family `NNMT`, 15 anchored folds, `config_wf`, seed 1, and a per-arm `class_prefix` such as `NNMT_MT_OP_fNN`, so plans never share generated fold classes: foldplan's default `<family>_WF_fNN` would collide) and arm-class writing (parameter dicts merged from the parent; each class gets its own `.json` copied from its parent's, or none if the parent has none)
- [X] T006 [P] Test `research/nnmt/tests/test_judge.py`:
  - `identity` passes on equal hashes and fails on one differing fold;
  - `model_share` = 1 − entries ÷ guard-passing bars on a synthetic table
- [X] T007 R8 head-probability capture: in `NNMT/NNMTStrategy.get_predictions`, store each head's class-probability array (aligned with the dataframe offset) in an instance dict `self._head_probs[task]` before the argmax. No change to the predictions or `%<task>` columns. The shared `Predictors/MLXClassifierMultiTask.py` is unchanged
- [X] T008 [P] Test `NNMT/tests/test_head_probs.py`: on a stub classifier returning fixed probabilities, `_head_probs` has the right shapes, and its argmax equals `%<task>` wherever the threshold does not force neutral
- [X] T013 [P] [US1] Test `NNMT/tests/test_model_key_nnmt.py`:
  - classes differing only in `prediction_threshold`, `profit_prediction_threshold`, `min_consecutive_buys`, `apply_task_filters`, `forward_peak_regressor_threshold`, `use_forward_peak_regressor_filter`, `enable_exit_signal`, `exit_rvol_threshold` or a framework switch (`ENTRY_MODE`, `HELD_PAIR_ADD`, `TRIGGER_NAME`) have EQUAL keys;
  - differing in `_CLASSIFIER_TASK_WEIGHTS`, `classifier_type`, `INPUT_SET` or `PROFIT_HORIZON` gives DIFFERENT keys;
  - an unpinned NNMT class resolves to kind `mainline-absent` under an empty whitelist.

  Confirm it FAILS on `use_forward_peak_regressor_filter` before T014
- [X] T014 [US1] R2 in `NNMT/NNMTStrategy.py`: move `use_forward_peak_regressor_filter` to `MODEL_KEY_IGNORED` and add the other post-model names explicitly. T013 and `Framework/tests/test_model_key_audit.py` pass, apart from the failures recorded in T002.

  This MUST land before `mt` trains (T011): it renames every NNMT key, and the identity arm reuses `mt`'s weights by key in strict mode
- [X] T009 Extend `research/nnnc_framework/signals.py`:
  - NNMT plan entries (PLAN_FAMILY `NNMT`);
  - `fold_table` writes `p_<task>_<class>` columns when the strategy has `_head_probs`;
  - the trading head's p_sell/p_hold/p_buy as now.

  Existing NNNC/NNPredict outputs stay byte-identical (re-run one existing entry on f00 and compare the parquet hash)

**Checkpoint**: harness ready, and NNMT keys final (T013/T014). No NNMT trade has changed.

---

## Phase 3: User Story 2 (part 1) — NNMT baseline on the harness (P1) 🎯 first data

**Goal**: current NNMT's 15-fold record, the pre-port reference.

- [X] T010 [US2] Write `research/nnmt/preregistration_models.json`:
  - arms `mt`, `mt_fw0`, `mt_op`, `mt_raw`, `mt_lr`, `mt_lr_raw`, `mt_lr_ri`, each with class, comparator and change;
  - the identity rule (15/15 `trades_hash`);
  - the R4 model-share minimum 0.5;
  - the R7 selection rule;
  - rule maps and minimums copied from `research/nnpredict/preregistration_models.json` (min_control_trades 200, min_arm_trades 100);
  - early f00–f07, late f08–f14;
  - seed-2 policy (R13);
  - comparators `wf_aro_g24`, `cs_aro`, `w_aro`, `st_g24_rev`.

  COMMIT it before T011
- [ ] T011 [US2] `build_arms models` writes `evaluation/folds/plan_mt.json` (base `NNMT_MLX`). Run f00 alone:
  - time the training and record it in `research/nnmt/compute.md`;
  - verify the Principle XI purge (research R4): the last training candle is before `test_start`, and the label tail is truncated;
  - check both bots.

  Then run f01–f14 in one background chain
- [X] T012 [US2] Verify 15 `mt` records, each with a model artifact and a non-zero trade count. Record the totals in `research/nnmt/report.md` §1 beside the four comparators

---

## Phase 4: User Story 1 — Framework port, key hygiene, identity (P1)

**Goal**: `NNMTStrategy` carries the NNNC framework with everything off and changes no trade.

**Independent test**: `mt_fw0` equals `mt` on 15/15 folds. The model-key tests pass.

(T013 and T014, key hygiene, were done in Foundational, before `mt` trained.)

- [X] T015 [US1] R1 in `NNMT/NNMTStrategy.py`: `class NNMTStrategy(EntryFrameworkMixin, BaseNNStrategy)` with `ENTRY_MODE = "model"`, `MODEL_ENTRY_FAMILY = None`, `HELD_PAIR_ADD = None`. Update `Framework/tests/test_exit_policy_single_source.py`'s allowlist only if it flags NNMT, with a comment
- [X] T016 [P] [US1] Test `NNMT/tests/test_framework_port.py`:
  - `NNMTStrategy` is an `EntryFrameworkMixin`;
  - `ENTRY_MODE == "model"`, `MODEL_ENTRY_FAMILY is None`, `HELD_PAIR_ADD_RESOLVED is None`, `FAMILY_EXITS == {}`;
  - `_model_tags` returns None;
  - `NNMT_MLX`'s model key is unchanged by the port (computed before and after, recorded in the test)
- [X] T017 [US1] Run `plan_mt_fw0` on f00 with `FT_STOP_INVARIANTS=report` and inspect the log for stop-invariant violations. Fix or record any before going on. Then run all 15 folds (no training: same key)
- [X] T018 [US1] `research.nnmt.judge identity`: `mt_fw0` vs `mt` must be 15/15 equal. If not, STOP, diagnose the differing fold (enter_tag, entries, exits) and fix the port. The identity result is written to `verdicts_models.json`

**Checkpoint**: the port is proven inert. Every later arm runs on the framework base.

---

## Phase 5: User Story 2 (part 2) — the operating point (P1)

- [X] T019 [US2] Write `NNMT/_nnmt_operating_point.py`: an `OperatingPointMixin` that merges `NNNC_A_Raw_Open.json`'s buy params (guards open, `prediction_threshold` 0.80) into the class's `buy_params` (039 pattern, `__init_subclass__`). Exits stay NNMT's
- [X] T020 [P] [US2] Test `NNMT/tests/test_operating_point.py`: the merged buy params equal ARO's open values, every parent key survives, and the model key equals `NNMT_MLX`'s
- [X] T021 [US2] `NNMT/NNMT_MLX_OP.py` and `plan_mt_op.json`. Run 15 folds (no training). Run `signals --models mt,mt_op` and record each arm's model-decided share
- [X] T022 [US2] `judge models` (partial): `mt_op` vs `mt_fw0`, with the shares. If `mt_op`'s share is < 0.5, record it. Model and head arms at that operating point are then INCONCLUSIVE by rule, and the report says the guards decide (no further tuning, Principle VI)

---

## Phase 6: User Story 4 — model variants (P2)

**Goal**: four variants at the operating point, the R7 selection, and the trading-head-only twin.

**Independent test**: each variant trains on f00, predicts every head with finite, non-constant
outputs, and has a distinct model key.

- [X] T023 [P] [US4] Add `ClassifierTypeMLX.LSTM_RAW` to `NNMT/NNMTClassifierMLX.py`:
  - a multi-task model with no resize or convolution preamble: features → `nn.LSTM(num_features, H)` (last step) → one `nn.Linear(H, n_classes)` per task;
  - H mirrors `NNNClassifierMLX`'s LSTM_RAW hidden size;
  - registered in the type factory;
  - trained by the existing multi-task loop.
- [X] T024 [P] [US4] Test `NNMT/tests/test_lstm_raw.py`: a forward pass on a (batch, seq, features) tensor returns one output per task in `TASK_NAMES` with the right class count. There are no Conv1d layers in the module tree
- [X] T025 [US4] Variant classes, all `NNMT_MLX` + `OperatingPointMixin`, with `InputSetMixin` from `NNPredict/_inputs.py`:
  - `NNMT/NNMT_Raw.py`: current network, `INPUT_SET = "raw"`;
  - `NNMT/NNMT_LR.py`: `classifier_type = LSTM_RAW`, 24 indicators;
  - `NNMT/NNMT_LR_Raw.py`: LSTM_RAW, raw;
  - `NNMT/NNMT_LR_RI.py`: LSTM_RAW, `raw_ind`.

  If `InputSetMixin`'s `raw` set includes NNPredict-only columns (`current_gain`, `recent_gain`), add an NNMT-side input list that drops them. The set is then exactly raw8, recorded in `research.md` R6
- [X] T026 [P] [US4] Test `NNMT/tests/test_variants.py`:
  - each variant's model inputs are what R6 lists;
  - the four keys are distinct from each other and from `NNMT_MLX`;
  - raw8 columns pass a truncation test (reuse `NNPredict/tests/test_inputs.py`'s helper)
- [ ] T027 [US4] Train f00 for `mt_lr` first, timing it and checking the predictions are finite and non-constant on every head. Then run all four variant plans in one sequential chain (60 trainings)
- [X] T028 [US4] `judge models`: the four variants vs `mt_op`, then the R7 selection `<M>` written to `verdicts_models.json`. Re-run `<M>` and `mt_op` on seed 2 (R13) and record the seed-2 result beside the seed-1 one — seed 1 DONE 2026-10-22: R7 = mt_lr_ri (+0.93 pp vs mt_op, paired UNCHANGED; PASS needs seed 2); seed 2 DONE 2026-10-22 21:18: mt_lr_ri 13.75 vs mt_op 10.04 (+3.08 / +0.63, paired UNCHANGED p 0.60) PASS -> <M> = mt_lr_ri confirmed

**Checkpoint**: `<M>` is selected and has a seed-2 replicate.

---

## Phase 7: User Story 3 — NNNC framework features and the twin (P2)

- [X] T029 [US3] Write `research/nnmt/preregistration_framework.json`:
  - arms `<M>_g24`, `<M>_hpa`, `<M>_gate`, `<M>_tp` vs `<M>`, and `<M>_tr` vs `<M>` (`wf_aro_g24` reported);
  - the INERT rule;
  - `<M>` named by rule, not by name.

  COMMIT it before T030
- [ ] T030 [US3] `build_arms framework` writes the five classes and plans (R5 table; `_tr` sets task weights `{trading: 1, others: 0}`). Run the four feature arms (backtest only), then `_tr` (15 trainings)
- [ ] T031 [US3] `judge framework`: feature verdicts, plus the twin's verdict ("does multi-task help?"). Seed 2 for any PASS. Write `verdicts_framework.json`

---

## Phase 8: User Story 4 (sizer) — the selected model's p_buy as a width-book sizer (P2)

- [X] T032 [US4] Write `research/nnmt/preregistration_screens.json`:
  - R9's sizer screen rule (early half only, gap ≥ 0.05, partial > 0, 5/8 sign folds) and its arm `<M>_cs` (on `w_aro`'s exits by `framework_sha`), judged against `w_aro` (must PASS) and `cs_aro` (reported);
  - R10's two head screens, each head's declared direction, and its arm use (R10 table);
  - the `<M>_ah` task weights (trading 4, every auxiliary 1).

  COMMIT it before T033
- [ ] T033 [US4] Write `research/nnmt/screen.py` (copying `research/nnpredict/screen.py`, parametrised by score column and direction). It refuses unless the 412 early width-book trades and the 0.1004 control gap reproduce. Run `signals --models <M>` and `screen sizer`, writing `screens_sizer.json`
- [ ] T034 [US4] Only if the screen passes: `build_arms sizer` (per-fold 0.5-share thresholds via `research.control_sizer.pick`; `<M>_cs` copies `w_aro`'s framework by `framework_sha`, and its unsized twin must match `w_aro` on 15/15). Run it, then `judge sizer`. Re-run on seed 2 if it PASSes. If the screen fails, record it in `verdicts_sizer.json` with no arm

---

## Phase 9: User Story 5 — auxiliary heads (P2)

**Goal**: a verdict per head (use or don't use, and how).

**Independent test**: `screens_heads.json` has both screens for all five heads in both halves.

- [ ] T035 [US5] `build_arms heads-model` writes `<M>_ah` (all auxiliary weights 1). Run 15 trainings and `signals --models <M>_ah`. Report its trading head against `<M>` (side check, not a verdict)
- [X] T036 [P] [US5] Write `research/nnmt/head_screen.py`:
  - recompute each auxiliary label (the strategy's own target functions on the test windows);
  - compute each head's causal baseline: the same state from a TRAILING window of equal length ending at t (regime: close vs a trailing 200-bar mean ±1.5%; risk: a trailing 50-bar mean of atr_norm; flow and momentum: trailing 24-bar means; profit: the trailing 8-bar realised outcome sign);
  - one-vs-rest AUC per half for head and baseline (screen 1);
  - the R9 metric in the declared direction on the 412 trades (screen 2);
  - `rho_pbuy`.

  Writes `screens_heads.json`
- [X] T037 [P] [US5] Test `research/nnmt/tests/test_head_screen.py`:
  - every trailing baseline is causal (a truncation test);
  - on a synthetic series where the head equals the label, the margin is > 0;
  - where the head equals the trailing state, the margin is about 0
- [ ] T038 [US5] Run `head_screen`. For each passing head:
  - `research.nnmt.build_arms heads tail` writes the out-of-sample tail tables (`user_data/data/nnmt_heads/tail_<M>_ah_fNN.parquet`: the fold strategy over its `TRAIN_DATA_SPLIT` holdout, as 039's `mix.py tables`);
  - `build_arms heads` writes its R10 arm (`hd_<head>`, a table-reading mixin on the declared book, percentiles from the tail tables).

  Run and `judge heads`, with seed 2 for any PASS. Heads that fail are recorded with no arm. Write `verdicts_heads.json`

---

## Phase 10: User Story 6 — regime transition (P1 by value, runs last by dependency)

**Goal**: a verdict on whether an upcoming bear↔bull change can be predicted and traded.

**Independent test**:
- the labels pass the truncation test;
- base rates are reported per fold;
- every grid cell has a screen result.

- [X] T039 [US6] Write `research/nnmt/preregistration_transition.json`:
  - the R11 definition (z, deadband 0.5, σ over 2880 bars, L = H);
  - grid H ∈ {96, 288, 672} × {pair, btc} × {UP, DOWN};
  - the stand-in model settings (037 HGB, purge H, hourly subsample, inputs);
  - both baselines;
  - the pass rule (margin ≥ 0.02 both halves, ≥ 200 events per half, proxy > 0 both halves);
  - the best-cell selection;
  - the R12 arms with thresholds and comparators;
  - the `<M>_trh` replacement rule (AUC within 0.01 both halves).

  COMMIT it before T040
- [X] T040 [US6] Write `research/nnmt/transition.py`:
  - `labels`: S_H(t) and the UP/DOWN/NONE label per pair and for BTC, on tradeable rows; base rates and event counts per fold and half to `screens_transition.json["base_rates"]`;
  - `causality`: 16 truncation cuts with the `research/indicator_screen/causality.py` method, covering S_H, the label's S_H(t+H) and every stand-in input; exit 1 on a mismatch.
- [X] T041 [P] [US6] Test `research/nnmt/tests/test_transition.py`:
  - on a synthetic random walk with an imposed bear→bull ramp, UP fires before the ramp completes and not elsewhere;
  - S_H(t) is unchanged when data after t is dropped;
  - a label is NaN within H of the data end;
  - the base rate of a pure random walk is symmetric (UP ≈ DOWN)
- [X] T042 [US6] Run `transition labels` and `transition causality` (it must PASS before T043). Write base rates into `report.md` §6 — DONE 2026-10-23: causality PASS (288 + 288 comparisons, 32/32 inputs PASS); base rates in §6
- [ ] T043 [US6] `transition screen --fold fNN` for f00–f14, one process per fold, sequential, never alongside an MLX chain:
  - inputs come from 037's panel (`base:*`, `raw8:*`, tradeable rows) plus BTC's raw8 and z;
  - per-fold HGB one-vs-rest per direction and cell, fit on the training window minus its last 90 days (purge H);
  - writes prediction tables for that held-out tail (threshold source) and for the test window. Then `transition judge`: AUCs, baselines, events, the proxy, pass, and the selected cell per direction to `screens_transition.json`
- [ ] T044 [US6] Only for selected passing cells: `build_arms transition` writes the R12 arms (`tr_gate`, `tr_scale`, `tr_exit` on `cs_aro`'s book; `tr_long` standalone on `st_g24_rev`'s exits). Each reads per-fold prediction tables with thresholds from the training tail. Run them and `judge transition`. Seed 2 for any PASS (the HGB is re-fit with seed 2)
- [ ] T045 [US6] Only if a stand-in cell passes: `NNMT/_transition.py` retargets the regime head with the R11 label for that cell (3 classes, task weight 1) as `<M>_trh`. Run 15 trainings and `signals`, and screen it with the same rules. If it matches the stand-in (AUC within 0.01 both halves), re-run the passing arms on its tables (tail tables from its `TRAIN_DATA_SPLIT` holdout), with seed 2 for any PASS. Otherwise record the gap
- [ ] T046 [US6] Write `verdicts_transition.json` with the verdict per cell and arm, including "not predictable at this horizon" for failing cells, with their numbers

### Regime-state amendment (user request 2026-10-22; `research/nnmt/preregistration_regime_state.json`)

- [X] T056 [US6] Pre-register the future-state target Y = S_H(t+H) (3 classes, 6 cells), its Markov / trailing-z baselines, the stand-in, the two NN models (`<M>_rg` regime-only, `<M>_rgm` multi-task), the shared-vs-separate comparison and the arms (`rs_gate`, `rs_scale`, `{M}_rgm`). COMMITTED before any label is built
- [ ] T057 [US6] `transition.py state-screen --fold fNN` (f00–f14, after T042's causality PASS): multiclass HGB stand-in, both baselines fit on the fold's fit rows, `screens_regime_state.json`; then `state-judge` applies the screen rule and selection. Tests first: the Markov baseline uses only fit rows; Y is NaN within H of the end; the target never enters the inputs
- [ ] T058 [US6] `NNMT/_regime_state.py` mixin (regime head label = Y for the selected or fallback cell, masked beyond the training range) and the classes `<M>_rg` / `<M>_rgm` with merged task weights; a test that both get new model keys and that no regime label reads a bar after `train_timerange`. 30 trainings, sequential, after the seed-2 chain and T035; `signals`; screen both on the cell; write shared-vs-separate
- [ ] T059 [US6] Arms per the pre-registration: `rs_gate` / `rs_scale` only for passing classes (tables: stand-in or a qualifying NN); `{M}_rgm` vs `{M}` whenever `<M>_rgm` is trained. `judge` with `judge_one`; seed 2 for any PASS. `verdicts_regime_state.json`, report §7

---

## Phase 11: Polish & cross-cutting

- [ ] T047 Write `research/nnmt/report.md`:
  - one table of every arm's per-half result against its comparator and against `cs_aro`, `w_aro`, `st_g24_rev` and `wf_aro_g24`, with trade counts and model-decided shares;
  - the screens' tables;
  - the three FR-015 verdicts (NNMT defaults recommendation, per-head verdicts, transition verdict).
- [ ] T048 Adopting results is the USER's call. Do not change `NNMT_MLX` / `NNMTStrategy` defaults beyond the inert port. List candidate defaults in the report
- [ ] T049 [P] Full test pass: `pytest NNMT/tests research/nnmt/tests Framework/tests utils/tests -q`. Only the failures recorded in T002 may remain (SC-007)
- [ ] T050 [P] Update `NNMT/README.md` (status, framework, variants, heads, transition, pointer to the report), `AGENT_GUIDE.md` (a 040 row and current state) and `README.md` (the NNMT directory status line)
- [ ] T051 Memory: write `project_040_nnmt.md`; correct `project_nnmt_task_audit.md` (the trading label is NOT task-filtered since 2618e0eb); add the MEMORY.md index line
- [ ] T052 Commit each stage as it completes (named files only). Push only when the user asks
- [X] T054 [US4] Architecture amendment (user request 2026-10-21): option 3 = shared LSTM + an LSTM per task (`ClassifierTypeMLX.LSTM_STACK`, `NNMT/NNMT_LS.py`, `NNMT/tests/test_lstm_stack.py`), pre-registered in `research/nnmt/preregistration_arch.json` vs `mt_lr` (option 1); option 2 (no shared LSTM) = the TR twin (T030-T031). `plan_mt_ls.json`; `judge arch`
- [X] T055 [US4] After the variant chain (T027): run `mt_ls` f00-f07 (screen), `signals --models MT_LS,MT_LR --folds f00..f07`, `judge arch`. Only on a screen PASS: f08-f14 and `judge arch` again (confirm), then seed 2 of `mt_ls` and `mt_lr`. Report in report.md; not an R7 candidate — DONE 2026-10-22: screen FAIL (early gain −1.19 pp vs mt_lr, paired UNCHANGED p 0.84, model share 0.85); no confirm stage
- [ ] T053 AFTER the study (user decision 2026-10-20): fix the walk-forward training window so no training bar or label reaches the test window (research/nnmt/compute.md § Fold boundary). freqtrade trims with `date <= stop`, so `train_timerange` must end at least H (= 48) bars before `test_start`. This is a harness change (`evaluation/foldplan.py` / `run_folds.py`, family-agnostic) with a test, and then a plan for re-running the comparators (`cs_aro`, `w_aro`, `wf_aro_g24`, `st_g24_rev`) so later studies compare like-for-like. Do NOT apply it during 040

---

## Dependencies & Execution Order

- **Setup (T001–T003)** → **Foundational (T004–T009, plus the key hygiene T013/T014)** → **US2
  part 1 (T010–T012)**: `mt` is recorded BEFORE the port and AFTER the key change.
- **US1 (T015–T018)** needs T012. The port (T015) must not land before `mt` is recorded.
- **US2 part 2 (T019–T022)** needs US1 (it runs on the framework base).
- **US4 variants (T023–T028)** need T022. T023–T026 (code and tests) can be written in parallel
  with any earlier run.
- **US3 (T029–T031)** needs T028 (`<M>`).
- **US4 sizer (T032–T034)** needs T028. Its screen is CPU only, but still never alongside an MLX
  chain.
- **US5 (T035–T038)** needs T028 and T032 (the pre-registration covers the head screens).
- **US6 (T039–T046)**:
  - T039–T042 (labels, causality) need only Foundational and can be built at any time;
  - the screen T043 must not overlap an MLX chain;
  - the arms T044 need `cs_aro`/`st_g24_rev` records (existing);
  - the head version T045 needs `<M>` (T028).
- **Polish (T047–T052)** comes after every story it reports.

## Parallel Opportunities

- Code and tests marked [P] within a phase: T006/T008/T013; T016; T023/T024/T026; T036/T037; T041.
- Writing code for a later stage while an MLX chain runs (e.g. T023–T026 and T036–T041 during the
  `mt` chain). Running two chains at once is never allowed.

## Implementation Strategy

- **MVP**: Phases 1–5. This puts NNMT on the common harness: the port proven inert, the baseline and
  the operating point, all comparable with every other family.
- Then the variants (selecting `<M>`), then the framework features and sizer, the heads, and the
  transition.
- Each stage's verdict is committed when it lands. A failing screen ends its branch with a recorded
  negative result. That is a successful outcome under the constitution.
