---
description: "Task list for 032 candidate-only training for NNNC models"
---

# Tasks: Candidate-Only Training for NNNC Models

**Input**: `specs/032-candidate-only-training/` (spec.md, plan.md, research.md R1–R9,
data-model.md, contracts/row_filter.md C1–C5, quickstart.md)

**Tests**: Included, because the spec requires each of the following and each needs a test:
- switch-off identity (FR-001, SC-001);
- filtered windows equal the unfiltered rows (FR-002/003, SC-002);
- validation filtered (FR-004);
- model key only for the mixin (FR-006);
- entry gate (FR-007);
- threshold keep-mask (FR-008);
- judge refusal (FR-013).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`. The harness runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan ...`.
- Training and backtests need `set -a; source ~/.freqtrade_env; set +a`. They run as ONE
  sequential chain: check `ps` for freqtrade, `test_strat`, `run_folds` and `run_arm` first. Do not
  run tests that read fold json while a chain runs.
- Commit to master, staging named files only. Run the full suite and check pytest's own exit code
  before every commit (quickstart §1).
- **No training of any `NNNC_CO_*` class before T014 (the pre-registration) is committed.** No
  `co_aro` / `co_gbr` arm record may exist before it either.
- **Stages are read in order (FR-012).** The early-half books are not looked at before S1 is
  recorded, and the late half is not looked at before S2 is recorded. A failed stage leaves the
  later stages `not_reached`.
- Labels: S2/S3 are PROVISIONAL (one era, 2023–26; research R8).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `research/candidate_training/__init__.py` and `research/candidate_training/tests/__init__.py` (empty package), and add `research/candidate_training/tests` to the suite command in quickstart §1 and in the tasks conventions.

---

## Phase 2: Foundational (blocking)

- [X] T002 [P] Write `Framework/tests/test_row_filter.py::test_mask_*` for contract C1. It must cover:
  - AND of conditions, each op (`lt`/`le`/`gt`/`ge`);
  - NaN → False;
  - a missing column → `KeyError`;
  - an empty spec, a bad op or a bad shape → `ValueError`;
  - the input frame is not mutated.
- [X] T003 Implement `Framework/RowFilter.py::row_filter_mask(df, spec)` per C1. Generic only: no NNNC names or values. T002 must pass.
- [X] T004 [P] Capture the switch-off parity fixture BEFORE touching `Framework/TrainingEngine.py`:
  - `Framework/tests/fixtures/prepare_training_data_pre032.npz` is built by a deterministic
    synthetic frame builder in `Framework/tests/test_row_filter.py`: 2 pairs, 120 rows, columns
    including `guard_metric`, `bb_width`, features and `%pnl_weight`;
  - it covers `tensor_method` 0 and 3, with and without `%pnl_weight`;
  - it uses a minimal host object that provides `seq_len`, `TRAIN_DATA_SPLIT`, `tensor_method`,
    `dataframeUtils`, `scale_dataframe` (identity) and `enhance_training_data` (pass-through).

  In the same step, BEFORE any Framework edit, record the current 030 model keys of
  `NNNC_A_Raw_Open`, `NNNC_GbbRaw_Open`, `NNNC_FS_GBR_GR`, `NNNC_ARO_f00` and `NNNC_FS_GBR_GR_f00`
  (via `Framework.ModelKey.resolve` under `config/config_wf.json`) into
  `Framework/tests/fixtures/model_keys_pre032.json`.
- [X] T005 [P] Fix `evaluation/pick_threshold.py` so `cmd_pick` / `cmd_control_rate` decide "trained?" with `run_folds.fold_has_model(fold, plan["base_strategy"], plan["config"], 1)`, not `model_exists(cls)` (research R6, C4). Add a test in `evaluation/test_pick_threshold.py` that monkeypatches `fold_has_model` and asserts it is consulted.

**Checkpoint**: run the full suite, then commit T001–T005.

---

## Phase 3: User Story 1 — default-off candidate-only training switch (P1) 🎯 MVP

**Goal**: a declared `TRAIN_ROW_FILTER` trains and validates only on eligible windows, with a new model key and an entry gate. Undeclared, nothing changes.

**Independent test**:
- the T004 parity fixture is identical;
- the filtered build keeps exactly the eligible windows, unbroken;
- the existing model keys and fingerprints pinned in tests are unchanged.

- [X] T006 [P] [US1] Tests in `Framework/tests/test_row_filter.py` for C2:
  - (a) with no filter, the output equals the T004 fixture for every case;
  - (b) with a filter, each kept train sample's window, label and weight equal the unfiltered
    arrays at the same row, and no kept sample has an ineligible last bar, for both tensor
    methods;
  - (c) the validation arrays are filtered the same way;
  - (d) zero kept train or validation rows → `ValueError`;
  - (e) `enhance_training_data` changing the row count with a filter → `ValueError`;
  - (f) `get_training_class_weights` on the kept labels with no SELL returns weight 0.0 for SELL
    and finite weights for the others;
  - (g) `_row_filter_report` has the data-model fields and correct counts;
  - (h) the first `seq_len-1` rows (zero-padded windows) are kept or dropped only by their own
    last bar's eligibility, and a kept padded window equals the unfiltered one.
- [X] T007 [US1] Implement the filter in `Framework/TrainingEngine.py::prepare_training_data` (research R1):
  - read `getattr(self, "TRAIN_ROW_FILTER", None)`, with NO class default added;
  - per pair, build train and validation masks from the raw frame `dataframe[i]`, sliced
    `[:train_end][offset:]` and `[test_start:][offset:]`;
  - index tensor, labels and weights with one integer index array, mlx-safe for
    `tensor_method` 3;
  - raise on misalignment or on zero rows;
  - set `self._row_filter_report`.

  In `_train_keyed`, after a successful fresh training, write `row_filter.json` at the model
  key's location when a report exists. T006 and T004 must pass.
- [X] T008 [P] [US1] Tests in `Framework/tests/test_row_filter.py` for C3. Resolve keys via `Framework.ModelKey.resolve` (never instantiate):
  - `CandidateTrainingMixin + NNNC_A_Raw_Open` has a different model key from `NNNC_A_Raw_Open`;
  - every class in `Framework/tests/fixtures/model_keys_pre032.json` (captured in T004, before
    T007) still resolves to its recorded key;
  - the mixin's `get_entry_conditions` returns parent mask AND predicate on a stub frame, with
    `_st_tags` recomputed;
  - `HELD_PAIR_ADD` is None.
- [X] T009 [US1] Implement `NNNC/_candidate_training.py::CandidateTrainingMixin` (C3):
  - `TRAIN_ROW_FILTER = (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))`;
  - `MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)`;
  - `HELD_PAIR_ADD = None`;
  - the `get_entry_conditions` override, following the `NNNC/_width_book.py` pattern.

  Run `Framework/tests/test_model_key_audit.py`; if it flags a name, classify it in the mixin.
  T008 must pass.
- [X] T010 [P] [US1] `evaluation/pick_threshold.py` keep-mask (C4):
  - `effective_buy_probs(df, keep=None)`: rows with `keep` False count as 0.0, and unscored rows
    are still dropped;
  - `predicted_buy_probs` passes `row_filter_mask(trimmed, strat.TRAIN_ROW_FILTER)` only when the
    strategy declares one.

  Tests in `evaluation/test_pick_threshold.py`: with `keep=None` the result is identical to
  today; with `keep` the ineligible bars are 0.0; at a fixed rate the threshold falls (or holds) when bars
  are zeroed.
- [X] T011 [US1] Run the full suite, check pytest's exit code, then commit T006–T010. Then run the SC-001 spot check:
  - re-run `wf_aro_hpon` f00 under a fresh tag (`wf_aro_hpon_p032`, via a copied plan with only
    `run_id` and `arm_tag` changed), one chain. This replaces `wf_aro_g24`, which predates the add
    default (e3a32dc); see verification.md;
  - its `trades_hash` must equal `regime/arms/wf_aro_hpon_f00_F00.json`'s;
  - record it in `specs/032-candidate-only-training/verification.md` and commit.

**Checkpoint**: the switch exists, is off by default, and is proven identical when off (SC-001
code-level, SC-002).

---

## Phase 4: User Story 2 — train, pick, backtest, measure (P2)

**Goal**: candidate-only ARO and GBR are trained on 15 folds, thresholded at rate 0.014735 and backtested. The stage-1 AUC and the instrument checks are reported.

**Independent test**:
- 30 trained folds, each with a `row_filter.json` (kept share about 20%, SELL 0);
- 30 arm records with trades;
- `auc.json`;
- fidelity ≥ 95% and 0 entry violations.

- [ ] T012 [US2] `research/candidate_training/build_arms.py` (C5), modelled on `research/model_calibration/build_width_arms.py`. It writes:
  - `NNNC/NNNC_CO_ARO.py` = `CandidateTrainingMixin + NNNC_A_Raw_Open`;
  - `NNNC/NNNC_CO_GBR.py` = mixin + `NNNC_FS_GBR_GR`;
  - their `.json` copied from the base, with `strategy_name` set;
  - `evaluation/folds/plan_co_aro.json` / `plan_co_gbr.json`, copied from `plan_wf_aro_g24.json`
    / `plan_fs_gbr_gr.json` with:
    - `run_id` `co_aro` / `co_gbr`;
    - `base_strategy` the new class;
    - `strategy_class` `NNNC_CO_<M>_fNN`;
    - `model_dir` `saved_data/NNNC_CO_<M>_fNN`;
    - `arm_tag` `co_<m>_fNN`;
    - `status` planned;
    - `thresholds` removed.

  Run it. Check with `run_folds --dry-run` that there are 15 unique classes and model dirs per
  plan. Fingerprint check, expected result declared here: each arm's `evaluation.fingerprint`
  sha must EQUAL the `framework_sha` recorded in its control's arm records
  (`regime/arms/wf_aro_g24_f00_F00.json`, `fs_gbr_gr_f00_F00.json`), resolved with the arm's own
  `.json`. The reasons:
  - the controls were recorded with the add off, and the mixin sets it off;
  - `TRAIN_ROW_FILTER` is not a fingerprint member, because it acts before the model and is
    covered by the model key;
  - the entry gate is a method, not a hashed attribute.

  Any other difference stops the task until it is explained.
- [ ] T013 [US2] `research/candidate_training/train.py --model ARO|GBR` (C5, research R7):
  - for each fold it calls `run_folds.generate_fold` (snapshot params, no threshold), then
    `fold_has_model`, then `train_fold` if absent, then `fold_has_model` again;
  - it prints the fold's `row_filter.json`;
  - it refuses to start if `ps` shows another freqtrade, `test_strat`, `run_folds` or `run_arm`
    process (the live dry-run bot `freqtrade trade` is excluded from that check);
  - it is resumable.

  Unit-test the process check and the resume skip in `research/candidate_training/tests/test_train.py` with monkeypatched `ps` and fold functions.
- [ ] T014 [US2] Pre-registration: `research/candidate_training/preregistration.json` and `.md`, per data-model.md and research R8:
  - books, control and width tags, folds, early and late halves, `target_rate` 0.014735;
  - the S1/S2/S3 rules verbatim (S1 as AUC difference > 0 in both halves, with the CI for
    information only);
  - the small-margin rule: a stage passing by under 1 pp is INCONCLUSIVE until a paired seed-2
    repeat (T021a) passes;
  - the decision rule: PASS (PROVISIONAL), FAIL or INCONCLUSIVE, never "adopted"; `zero trades` →
    INCONCLUSIVE;
  - the confirmation route for a PASS (the folds after 2026-09-07, or dry-run data);
  - the statement that no outcome is recorded as a finding (constitution II, gate 2);
  - the labels and an empty `exempt_records`.

  Commit it alone (named files) BEFORE any `NNNC_CO_*` training. Record the commit hash in the
  `.md`.
- [ ] T015 [US2] Smoke: `train.py --model ARO` on f00 only (`--folds f00`). Check that:
  - `row_filter.json` has a train kept share between 10% and 35%, SELL count 0 and BUY share
    between 30% and 70%;
  - the model key directory is new and the existing ARO f00 weights are untouched (by file sha);
  - `pick_threshold.py --plan plan_co_aro.json --fold f00 --target-rate 0.014735` returns a
    threshold with an achieved rate within 10% of the target.

  STOP and diagnose before T016 if any check fails. No backtest at this step.
- [ ] T016 [US2] Train ARO f01–f14 (`train.py --model ARO`, background, one chain).
  Then run `pick_threshold.py --plan plan_co_aro.json --target-rate 0.014735 --out
  evaluation/folds/thresholds_co_aro.json` and write the map into `plan_co_aro.json` `thresholds`.
  Then run `run_folds --plan plan_co_aro.json`: every fold must log "model already present" and
  15/15 must be evaluated with trades. Record per-fold achieved rates.
- [ ] T017 [US2] The same for GBR (train, pick into `thresholds_co_gbr.json`, plan, `run_folds`), started only after T016's chain has exited.
- [ ] T018 [P] [US2] Instrument checks, with the result committed in `research/candidate_training/instrument.json`. Read entries only, not P&L:
  - add `CO_ARO` / `CO_GBR` (plans `co_aro` / `co_gbr`, prefixes `NNNC_CO_ARO` / `NNNC_CO_GBR`) to
    `research/nnnc_framework/signals.py` `PLANS` and to `diagnose.MODELS_R2`;
  - run `signals.py --models CO_ARO,CO_GBR`;
  - (a) simulator recall and precision on each arm's recorded entries, ≥ 95% (SC-005);
  - (b) entry violations = 0, i.e. no recorded entry bar fails the predicate;
  - (c) every fold's `row_filter.json` summary.
- [ ] T019 [US2] `research/candidate_training/auc.py` (S1):
  - within-candidate AUC per model and half, for the retrained and the current model, on
    identical eligible bars and labels (reusing `research/model_calibration/within_candidates.py`
    helpers);
  - writes `auc.json`;
  - unit-test the "identical bars" join in `research/candidate_training/tests/test_auc.py`.

  Run it, and record S1 pass or fail per the pre-registration.

**Checkpoint**: S1 recorded. Early and late P&L have not been read.

---

## Phase 5: User Story 3 — pre-registered judgement (P3)

**Goal**: S2 then S3 per model, decided by the committed rules.

**Independent test**: `verdicts.json` with the stages in order, `not_reached` after a failure, and the judge refusing stale or uncommitted inputs.

- [ ] T020 [P] [US3] `research/candidate_training/judge.py` (C5). It refuses (exit 2) if `preregistration.json` is uncommitted or dirty, or if any `co_*` record predates its commit (reuse the pattern in `research/held_pair/judge.py::refusals`). It then evaluates per model:
  - S1 from `auc.json`;
  - S2 from early pp of the arm, control and width book (`evaluation.pool`);
  - S3 from `research.nnnc_switches.judge.judge_one(arm, control)`, plus late pp against the
    width book, plus `judge_one(arm, width)` reported;
  - `fidelity`, `entry_violations` and `liveness` from `instrument.json`;
  - `thirds` (pp per f00–f04, f05–f09, f10–f14, context only);
  - the small-margin check;
  - then the decision: PASS (PROVISIONAL), FAIL or INCONCLUSIVE, with `confirmation_route` on a
    pass.

  Tests in `research/candidate_training/tests/test_judge.py`: a refusal on a stale record; stage
  order (an S1 fail makes S2/S3 `not_reached` without reading pools, asserted by a pool stub that
  raises); a zero-trade fold → INCONCLUSIVE; a 0.5 pp margin → INCONCLUSIVE pending repeat; the
  word "ADOPTED" never appears in the output.
- [ ] T021 [US3] Run `judge.py`. Commit `verdicts.json`.
- [ ] T021a [US3] CONDITIONAL, only if T021 marks a stage INCONCLUSIVE on the small-margin rule. Paired seed-2 repeat over that stage's folds:
  - `run_folds --seed 2` for the arm (train → pick → evaluate, one chain);
  - the same for the control (its seed-2 model);
  - re-run `judge.py --seed-repeat 2` so the stage passes only if the margin is > 0 on both seeds.

  Commit the repeat records and the updated `verdicts.json`.
- [ ] T022 [US3] Write `research/candidate_training/report.md`:
  - the stage table per model and the decision;
  - liveness, fidelity and entry violations;
  - the PROVISIONAL label;
  - a negative result recorded with full evidence;
  - wording: "PASS (PROVISIONAL)" is a provisional result, not a finding. Name its
    forward-confirmation route.

  Add a §7 pointer in `research/model_calibration/findings.md`. Commit.

---

## Phase 6: Polish

- [ ] T023 [P] If any new research script sets an exit parameter, allowlist it in `Framework/tests/test_exit_policy_single_source.py`. Run the full suite and check pytest's exit code.
- [ ] T024 [P] Memory: add a `project_032_candidate_training.md` file and its line in MEMORY.md; update `project_model_check_width_sort.md`.
- [ ] T025 Mark all tasks `[X]` in this file. Final full suite, then commit.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2 → US1 → US2 → US3 → Polish.**
- **T004 before T007:** the fixture must come from the pre-change code.
- **T008's key capture before T009.**
- **Pre-registration before training:** T014 is committed before T015, and T015 must pass before
  T016.
- **One chain:** T011's spot check, T016 and T017 run one after another, never together. T021a
  runs only if it is triggered, and after T021.
- **Order of reading:**
  - T018 (entries only) and T019 (S1) come before T020/T021 read any P&L;
  - T020's code can be written in parallel with T016/T017 (no results read).

## Parallel Opportunities

- T002, T004 and T005 (different files).
- T006, T008 and T010 (test files and the `pick_threshold` change).
- T018 and T020 (code) while a training chain runs. Do not run the suite while a chain runs if it
  reads fold json; the T020 tests use stubs only.

## Implementation Strategy

1. **MVP = US1.** The switch, proven identical when off, with a new key and an entry gate.
   Commit.
2. **US2.** Pre-register, smoke one fold, then the two training chains (about 6 h in total), then
   the instrument checks and S1.
3. **US3.** Judge in stage order, then the report.
