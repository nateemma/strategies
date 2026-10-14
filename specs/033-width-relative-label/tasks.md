---
description: "Task list for 033 width-relative label for NNNC models"
---

# Tasks: Width-Relative Label for NNNC Models

**Input**: `specs/033-width-relative-label/`:
- spec.md, plan.md;
- research.md R1–R10 (with R2b, R6b);
- data-model.md;
- contracts/pooled_label.md (C1) and contracts/width_sizer.md (C2);
- quickstart.md.

**Tests**: Included. The spec requires each of the following, and each needs a test:
- switch-off identity (FR-001, SC-001);
- hand-computed labels (SC-002);
- label sanity bounds (SC-003);
- the model key only for the mixin (FR-007);
- the 4-decimal threshold (FR-009);
- stake weights (FR-008);
- judge refusal and stage order (FR-013/014).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`. The harness runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan ...`.
- Training, picks and backtests need `set -a; source ~/.freqtrade_env; set +a`. They run as ONE
  sequential chain, with picks one fold per process. Check `ps` for freqtrade, `run_folds`,
  `run_arm`, `pick` and `train` first. Do not run tests that read fold json while a chain runs.
- Any script that scores a fold strategy outside a backtest calls `strat.ft_bot_start()` and
  verifies the loaded params against the fold `.json` (032 lesson).
- Commit to master, staging named files only. Run the full suite and check pytest's own exit code
  before every commit (quickstart §1).
- **No training of any `NNNC_WR_*` class before T017 (the pre-registration) is committed.** No
  `wr_*` arm record may exist before it either.
- **Stages are read in order, per arm (FR-013).** A failed stage leaves that arm's later stages
  `not_reached`, and their records unread.
- Results are at most PASS (PROVISIONAL) (one era).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 Create `research/width_relative/__init__.py` and `research/width_relative/tests/__init__.py`. Add `research/width_relative/tests` to the suite command in quickstart §1 and in these conventions.

---

## Phase 2: Foundational (blocking)

- [ ] T002 BEFORE any Framework or NNNC edit, capture the parity fixtures:
  - `Framework/tests/fixtures/model_keys_pre033.json`: the 030 model keys (via
    `Framework.ModelKey.resolve` under `config/config_wf.json`) of `NNNC_A_Raw_Open`,
    `NNNC_GbbRaw_Open`, `NNNC_FS_GBR_GR`, `NNNC_CO_ARO`, `NNNC_CO_GBR`, `NNNC_ARO_f00` and
    `NNNC_FS_GBR_GR_f00`;
  - `Framework/tests/fixtures/train_keyed_labels_pre033.npz`: the labels that `_train_keyed`
    passes to `train_model` for a deterministic 2-pair synthetic frame. Build it with a stub host
    whose `train_model` records its arguments, using the 032 test frame builder.
- [ ] T003 [P] Write `Framework/tests/test_pooled_label.py::test_path_outcome_*` for C1 `path_outcome`. It must cover:
  - the buy side against `labels_gbb_path` with its rule gates disabled: `outcome == target`
    exactly where that label is 1, on bars with a full horizon, on a scripted frame and on one real
    pair-month from `user_data/data`;
  - the sell side against `labels_gbb_path_sell`, the same way, with target −0.02 and take-profit
    0.022;
  - a stop first gives `stop`; neither hit gives the horizon return;
  - near the frame end (fewer than `horizon` bars left): a barrier hit within the remaining bars
    gives a valid outcome; no hit gives NaN and `bars_to_exit` −1;
  - a wrong sign for the side → `ValueError`;
  - the input is not mutated.
- [ ] T004 Implement `Framework/TrainingSignals.py::path_outcome(df, target, stop, horizon, side="buy")` (C1, research R2/R2b), close-based, reusing that module's forward helpers where they fit. Generic only. T003 must pass.

**Checkpoint**: run the full suite, then commit T001–T004 (fixtures included).

---

## Phase 3: User Story 1 — default-off width-relative label (P1) 🎯 MVP

**Goal**: a declared `POOLED_LABEL` relabels each fold's eligible bars relative to width, on both
sides, trains on entry- or exit-eligible rows, and gets its own model key. Undeclared, nothing
changes.

**Independent test**:
- the T002 fixtures are identical;
- hand-computed labels match;
- the label bounds hold on a scripted frame.

- [ ] T005 [P] [US1] Tests `Framework/tests/test_pooled_label.py::test_relabel_*` for C1 `relabel`. They must cover:
  - a scripted 2-pair frame with known outcomes and widths: the buckets, BUY (top share), SELL
    (bottom share of the exit side) and HOLD, hand-computed, are equal;
  - ties at the target broken by `bars_to_exit`, then by position, deterministically;
  - every bucket within ±1 bar of `share × n` on the fit rows;
  - with `fit_split`: the edges and cuts come from the fit rows only (changing only validation
    outcomes leaves them unchanged), and validation rows are labelled by those cuts;
  - the sell side ranks ascending (bottom share is SELL);
  - `%wr_keep` is 1.0 on eligible valid bars and NaN elsewhere, with NaN outcomes excluded;
  - an entry/exit overlap → `ValueError`;
  - `min_bucket` merging reported;
  - a bad spec → `ValueError`, a missing column → `KeyError`;
  - the report fields per data-model (`RelabelReport`, `WidthBucket` with `side`);
  - the Spearman of the label with the reference is within ±0.05 on a frame where the outcome is
    width-correlated by construction (the R3 case);
  - inputs other than the `%wr_keep` column are not mutated.
- [ ] T006 [US1] Implement `Framework/PooledLabel.py::relabel(frames, labels, spec, fit_split=None)` per C1 and data-model. It handles both sides, fits equal-count buckets and per-bucket cuts per side over the pooled eligible valid bars of the fit rows (the first `fit_split` of each frame), applies them to all rows, and uses the ranking key `(outcome, tie-break, pair index, row)`: descending for BUY, ascending for SELL. It works on the label format `maybe_train` buffers (check it in `Framework/BaseNNStrategy.py` around `get_training_labels`) and returns the same format. Generic only. T005 must pass.
- [ ] T007 [P] [US1] Tests for the TrainingEngine hook in `Framework/tests/test_pooled_label.py::test_hook_*`:
  - with no `POOLED_LABEL`, the labels reaching `train_model` equal
    `train_keyed_labels_pre033.npz`;
  - with it declared, they equal `relabel`'s output;
  - `width_relative.json` is written beside the weights after a fresh training (stub model
    location, as 032's `row_filter.json` test);
  - the 032 row filter `(("%wr_keep","gt",0.5),)` keeps exactly the `%wr_keep` rows.
- [ ] T008 [US1] Implement the hook in `Framework/TrainingEngine.py::_train_keyed`. Before `train_model`, when `getattr(self, "POOLED_LABEL", None)` is set, call `PooledLabel.relabel` on the buffered frames and labels with `fit_split=self.TRAIN_DATA_SPLIT`, and set `self._pooled_label_report`. After a fresh training, write `width_relative.json` next to `row_filter.json`. No class default. T007 and T002 must pass.
- [ ] T009 [P] [US1] Tests in `research/width_relative/tests/test_mixins.py` for `WidthRelativeMixin` (C2):
  - the key of `WidthRelativeMixin + NNNC_A_Raw_Open` differs from `NNNC_A_Raw_Open`'s and from
    `NNNC_CO_ARO`'s;
  - changing `buckets` or `share` changes the key;
  - every class in `model_keys_pre033.json` still resolves to its recorded key;
  - `TRAIN_ROW_FILTER == (("%wr_keep","gt",0.5),)`;
  - `HELD_PAIR_ADD` is None;
  - the spec values equal the class's `MIN_BUY_GAIN_THRESHOLD`, `MIN_SELL_LOSS_THRESHOLD` and
    `HORIZON`.
- [ ] T010 [US1] Implement `NNNC/_width_relative.py::WidthRelativeMixin`:
  - `POOLED_LABEL`, built from the class's own label attributes (target, stop −0.05, exit loss,
    take-profit 0.022, horizon) with entry rule (guard_metric < 0.3, bb_width > 0.035), exit rule
    (guard_metric > 0.3, bb_width > 0.035), reference `bb_width`, buckets 10, share 0.5,
    min_bucket 500;
  - `TRAIN_ROW_FILTER`;
  - `MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER", "POOLED_LABEL")`;
  - `HELD_PAIR_ADD = None`.

  Run `Framework/tests/test_model_key_audit.py` and classify any flagged name. T009 must pass.
- [ ] T011 [P] [US1] Tests in `research/width_relative/tests/test_mixins.py` for `WidthSizerMixin` (C2):
  - entries equal `WidthBookMixin`'s on a stub frame, with tags suffixed `:agree` or `:disagree`
    from `predict_buy`;
  - `custom_stake_amount` returns `proposed × 1.0/0.75` for agree and `× 0.5/0.75` for disagree,
    capped by the parent's fillable stake and `max_stake` (stub `_fillable_stake`);
  - `prediction_threshold` on the mixin class keeps 4 decimals (0.53424 → 0.5342) while
    `NNNC_A_Raw_Open().prediction_threshold` still rounds to 2;
  - setting one does not change the other in the same process;
  - `MODEL_KEY_IGNORED` contains `AGREE_SHARE` and `STAKE_WEIGHTS`, and they leave the model key
    unchanged.
- [ ] T012 [US1] Implement `NNNC/_width_sizer.py::WidthSizerMixin` (C2):
  - its own `prediction_threshold` `DecimalParameter(0.0, 1.0, decimals=4, default=0.5,
    space="buy", load=True)`;
  - `AGREE_SHARE = 0.5`, `STAKE_WEIGHTS = (1.0, 0.5)`;
  - `get_entry_conditions` calls super (the width book) and appends the tag;
  - `custom_stake_amount` reads the tag;
  - `MODEL_KEY_IGNORED` extended.

  T011 must pass.
- [ ] T013 [US1] SC-001 spot check: re-run the reference fold `wf_aro_hpon_p032` f00 (recorded trades_hash 846c306531c85348) and record identity in `specs/033-width-relative-label/verification.md`, together with the T002 fixture results.

**Checkpoint**: run the full suite, then commit T005–T013.

---

## Phase 4: User Story 2 — train, run and measure (P2)

**Goal**: width-relative ARO and GBR weights for 15 folds; both sizer arms backtested; the S0
instrument and the S1 separation recorded.

**Independent test**:
- the label audit, instrument and separation JSON per model;
- 30 trained folds;
- 60 arm records.

- [ ] T014 [P] [US2] `research/width_relative/build_arms.py` (C2). It generates:
  - `NNNC/NNNC_WR_ARO.py/.json` (bases `WidthSizerMixin, WidthBookMixin, WidthRelativeMixin,
    NNNC_A_Raw_Open`) and `NNNC/NNNC_WR_GBR.py/.json` (`... NNNC_FS_GBR_GR`). The `.json` files
    are copied from the controls' params snapshots (24 h grace), with `enable_exit_signal` false,
    and `WIDTH_CUTS` from `research/model_calibration/width_twin.json`;
  - plans `evaluation/folds/plan_wr_<m>_sz.json` (exit-off) and `plan_wr_<m>_szx.json` (exit-on;
    `enable_exit_signal` true in its fold params). Same folds and model key; new arm tags and fold
    class names (`NNNC_WR_<M>_SZ_fNN`, `NNNC_WR_<M>_SZX_fNN`); no thresholds yet.

  Tests in `research/width_relative/tests/test_build_arms.py`:
  - each arm's fingerprint inputs equal the control's except the declared mixins and the exit
    switch;
  - both plans resolve to the same model key per fold.
- [ ] T015 [P] [US2] `research/width_relative/pick.py --model M --fold fNN --out PATH` (research R7). It:
  - loads the `_sz` fold class with `ft_bot_start()` and threshold 0;
  - keeps the validation tail (the last `1 − TRAIN_DATA_SPLIT` of each pair's training frame);
  - takes the FIRST bar of each width-book candidate run (`enter_long` rising from 0 to 1 per
    pair): the bars that can open a trade;
  - picks the 4-decimal threshold at the `1 − AGREE_SHARE` quantile of the effective BUY
    probability (argmax BUY ? p_buy : 0) over those bars;
  - reports the achieved share;
  - with `--verify`, rescores at the picked threshold and asserts tags match `p > t` with 0
    mismatches.

  Tests in `research/width_relative/tests/test_pick.py` for the quantile and achieved share on
  synthetic arrays.
- [ ] T016 [P] [US2] `research/width_relative/train.py --model M [--fold fNN]`: the 032 `train.py` pattern (one fold per process, resumable, refuses if another chain runs, ignores `sh -c` wrappers). It trains the `_sz` plan's fold classes and prints each fold's `width_relative.json` and `row_filter.json` summaries.
- [ ] T017 [US2] Write and COMMIT `research/width_relative/preregistration.json/.md` (research R9), BEFORE any training. It covers:
  - S0 bounds per fold and side: bucket share [0.40, 0.60], |Spearman| ≤ 0.05, kept > 0,
    BUY > 0, SELL > 0;
  - the exit-off trades_hash identity with `w_<m>` per fold, with differences reported and
    flagged;
  - 0 entry violations;
  - tag-to-agree 0 mismatches;
  - exit liveness (exit-on);
  - thresholds loaded to 4 decimals;
  - the stake shortfall > 5% flag;
  - S1 (exit-off only, EARLY half only): the new early gap above the current early gap (ARO 0.072,
    GBR 0.063); the late gap is not computed before S3;
  - comparator reads: an arm's records stay readable as the other arm's comparator;
  - S2/S3 per arm with their comparators;
  - the 1 pp small-margin rule;
  - the decision vocabulary;
  - the confirmation route.

  Record the commit id in the `.md`.
- [ ] T018 [US2] Smoke, ARO f00 only (after T017): train → `width_relative.json` within the S0 bounds on both sides → `row_filter.json` has BUY > 0 and SELL > 0 → the existing `NNNC_ARO_f00` and `NNNC_CO_ARO_f00` weights are unchanged (sha) → pick f00 with `--verify` (0 mismatches) → the flat-stake twin (`STAKE_WEIGHTS` (1.0, 1.0), a one-off class) on f00 has trades_hash equal to `w_aro` f00's → the threshold loads to 4 decimals. Record the results in the preregistration `.md` (smoke section). Stop if any check fails.
- [ ] T019 [US2] Chain ARO, then GBR (one chain), each in this order:
  1. train 15 folds;
  2. pick 15 folds, one per process;
  3. write the thresholds into both plans;
  4. `run_folds` on `_sz`;
  5. `run_folds` on `_szx`.

  Save the pick logs to `evaluation/folds/thresholds_wr_<m>_log.txt`.
- [ ] T020 [US2] Register the WR plans in `research/nnnc_framework/signals.py` (as 032's `CO_PLANS`, runtime registration), then run `signals --models WR_ARO,WR_GBR`. The signal tables come from the `_sz` plan (exit-off; the predictions are the same).
- [ ] T021 [P] [US2] `research/width_relative/instrument.py` (S0; entries and labels only, no P&L) and tests. Per model:
  - the label audit from each fold's `width_relative.json` against the bounds;
  - `row_filter.json` class counts;
  - the exit-off per-fold trades_hash against `w_<m>`;
  - entry violations, where the signal bar fails the width-book rule, in both arms;
  - tag-to-agree mismatches against the signal tables (`buy_argmax` and `p_buy > threshold`);
  - exit liveness for the exit-on arm: `model_exit` count per fold, and gate-passing bars
    (rvol > 2.0 and guard_metric > −0.1) inside open trades;
  - the stake shortfall: realised `stake_amount` from each record's result file against the
    intended stake from its tag;
  - the per-fold thresholds and the achieved agree shares on the validation-tail run starts;
  - per fold, from the arm records' tags and stakes only (no profit field): the realised
    test-window agree share among trades and the mean stake, against the width book's 900 (FR-008).

  The output is `instrument.json`.
- [ ] T022 [P] [US2] `research/width_relative/separation.py` (S1) and tests. It applies `research/candidate_training/co_sizing.py`'s method to the identical width-book trades, with agree recomputed from the WR signal tables at the fold thresholds. It reports the new and current gaps and the agree_0.5 pp per capital on the EARLY half (f00–f07) only. It refuses to compute late-half numbers unless `--after-s3` is passed and `verdicts.json` records S3 for that arm. The output is `separation.json`.
- [ ] T023 [US2] Run T021 and T022. Commit their JSON. (Order: instrument first, then S1. No P&L of the arms is read yet.)

**Checkpoint**: run the full suite, then commit T014–T023.

---

## Phase 5: User Story 3 — pre-registered judgement (P3)

- [ ] T024 [P] [US3] `research/width_relative/judge.py` and tests, following 032's `judge.py`. Per model and arm:
  - refusals (an uncommitted or dirty pre-registration; any `wr_*` record older than its commit);
  - S0 from `instrument.json`;
  - S1 from `separation.json` (exit-off only; an S1 fail stops only that arm);
  - S2 from early pp: exit-off against `w_<m>` and the control; exit-on against the exit-off arm
    and `w_<m>`;
  - S3: `judge_one` against the arm's primary comparator (the control for exit-off, the exit-off
    arm for exit-on), plus late pp against `w_<m>`; `judge_one` against `w_<m>` is reported;
  - comparator reads: the exit-off arm's records are read as the exit-on arm's comparator even when
    the exit-off arm's own verdict stopped earlier;
  - the small-margin rule;
  - thirds as context, computed only for stages reached;
  - the decision vocabulary; the word "ADOPT" is never output.

  Tests with stubs: the stage order (unread pools raise); an exit-off S1 fail leaves the exit-on
  arm judged, with the exit-off records readable as its comparator; a small margin is INCONCLUSIVE; a refusal on a stale record.
- [ ] T025 [US3] Run `judge.py`. Commit `verdicts.json`.
- [ ] T025a [US3] CONDITIONAL, only if a stage is INCONCLUSIVE on the small-margin rule: run the paired seed-2 repeat for that arm and its comparator over that stage's folds, then re-judge. Commit.
- [ ] T026 [US3] Write `research/width_relative/report.md`. It contains:
  - the stage tables per model and arm;
  - the label audit;
  - the instrument results (including the exit liveness and stake shortfall);
  - S1 (early); after S3, the late-half gap as context (`separation.py --after-s3`);
  - the realised agree shares and mean stakes;
  - the decisions, labelled PROVISIONAL, with a negative result recorded with full evidence;
  - the confirmation route.

  Add a §8 pointer in `research/model_calibration/findings.md`. Commit.

---

## Phase 6: Polish

- [ ] T027 [P] If any new script names an exit parameter, add it to the allowlist in `Framework/tests/test_exit_policy_single_source.py` (configuring only, with a one-line reason). Run the suite.
- [ ] T028 [P] Memory: write `project_033_width_relative_label.md` and add its line to MEMORY.md. Update `project_model_check_width_sort.md` and `project_032_candidate_training.md` (next-step pointer).
- [ ] T029 Mark all tasks `[X]`. Run the final full suite, then commit.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2 → US1 → US2 → US3 → Polish.**
- **T002 before any edit** to `Framework/` or `NNNC/` (fixtures capture the pre-change state).
- **T004 before T006** (`relabel` uses `path_outcome`). **T006 before T008.** **T010 and T012
  before T014.**
- **T017 committed before T018.** T018 must pass before T019.
- **One chain:** T013, T018 and T019 run one after another, never together. T025a only after T025.
- **Order of reading:**
  - T021 (S0) and T022 (S1) before any arm P&L is read in T024/T025;
  - T024's code can be written while T019 runs (stubs only).

## Parallel Opportunities

- T003 alongside T002 (test-only) once the fixtures are captured.
- T005, T007, T009 and T011 (separate test targets).
- T014, T015 and T016 (separate scripts).
- T021, T022 and T024 (code) while the T019 chain runs. Do not run suite tests that read fold json
  during the chain.

## Implementation Strategy

- **MVP = US1:** a correct, default-off label with its own key. Verified by fixtures and
  hand-computed labels before any training.
- **Then US2:** the smoke on one fold, then the full chain.
- **Then US3:** the judgement. Each stage is a stop point. A failed stage is a result, written up
  with the same weight as a pass.
