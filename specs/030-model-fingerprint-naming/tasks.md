---
description: "Task list for 030 model naming by model fingerprint"
---

# Tasks: Model Naming by Model Fingerprint

**Input**: `specs/030-model-fingerprint-naming/` — spec.md, plan.md, research.md (R1–R10),
data-model.md, contracts/resolver.md, quickstart.md.

**Tests**: included. The spec requires each of these to be shown:
- byte-identical trades (SC-001, SC-002, SC-007);
- key sensitivity and invariance (SC-003);
- the migration report (SC-004);
- that renaming a class doesn't retrain it (SC-005);
- fingerprint neutrality (SC-006);
- strict-mode halts (SC-008).

**Conventions**:
- Paths are relative to `user_data/strategies/`, and Python is `../../.venv/bin/python`.
- Backtests run from the repo root, after `set -a; source ~/.freqtrade_env; set +a`, as **one**
  sequential chain under fresh tags.
- Commit to master, staging named files only.
- **No loader change before the baseline (T008) is committed.**
- `get_model_path` is not switched until migration (T020) reports 0 conflicts.
- **No model is trained** at any step. Grep every chain log for `training on`; the expected count
  is 0.
- **STOP gate at T020:** any migration conflict stops the feature before T023. Report it in
  `verification.md`.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (baseline tooling)

- [X] T001 [P] Create `scripts/model_index.py` with only the `verify-bytes` subcommand for now
  (contracts/resolver.md).
  - `--write <json>` records `{relpath: {size, sha256}}` for every file under `saved_data/` except
    `model_index.json` and `manifest.json` files.
  - `<json>` without `--write` compares against it and exits 1 on any difference, printing the
    differing paths.
- [X] T002 [P] Write `specs/030-model-fingerprint-naming/scan_overrides.py`, an AST scan.
  - For every BaseNNStrategy subclass in NNNC, NNMT, NNPredict, NNWavelet, Anomaly and Sklearn,
    record the class that defines its effective `get_model_path` (Framework base, a mixin or an
    override) and the literal target where it can be parsed.
  - Output: `legacy_redirects_before.json`.
- [X] T003 Create the baseline plans (copies of the existing plans with new `run_id`s and arm tags;
  model dirs unchanged), committed under `evaluation/folds/`:
  - `plan_wf_aro_b030.json`, from `plan_NNNC_A_Raw_Open.json`;
  - `plan_sw_fmr_ctl_b030.json`, from `plan_NNNC_ForwardMAE_Raw.json`;
  - `plan_wf_trgate_b030.json`, from `plan_wf_trgate_b029.json`;
  - `plan_sw_aro_gate_b030.json`, from `plan_sw_aro_gate.json`;
  - `plan_sw_mlx_c2_b030.json`, from `plan_sw_mlx_c2.json`;
  - `plan_sw_gbr_ctl_b030.json`, `plan_sw_gfm_ctl_b030.json` and `plan_sw_mlx_ctl_b030.json`, from
    the `sw_*_ctl` plans (FR-018's GBR, GFM and MLX references, re-run same-day).

---

## Phase 2: Foundational (the baseline, on unchanged loader code)

- [X] T004 Snapshot the weights:
  `scripts/model_index.py verify-bytes --write specs/030-model-fingerprint-naming/weights_before.json`.
  Then run T002 and save its output, `legacy_redirects_before.json`.
- [X] T005 Snapshot the fingerprints:
  `scripts/nnnc_fingerprints.py --out specs/030-model-fingerprint-naming/fingerprints_before.json`.
- [X] T006 Run the baseline chain (one chain, logged to the scratchpad). Export
  `FT_MODEL_EVAL_STRICT=1` for the whole chain (R8; inert on the old loader, so the baseline and
  after chains share one environment).
  - the eight T003 plans through `run_folds`;
  - `regime/run_arm.py --strategy NNMT_MLX_Base --window W1 --tag nnmt_base_b030`;
  - `regime/run_arm.py --strategy NNMT_Ctl_s42 --window W1 --tag nnmt_ctl42_b030`.
- [X] T007 Write `specs/030-model-fingerprint-naming/baseline_check.py`, which compares two sets
  of arm records per fold by `trades_hash` and `n_trades`, and run it on the baseline:
  - `wf_aro_b030` vs `wf_aro_a029`;
  - `wf_trgate_b030` vs `wf_trgate_b029`;
  - `sw_aro_gate_b030` vs `sw_aro_gate`;
  - `sw_mlx_c2_b030` vs `sw_mlx_c2`;
  - `sw_fmr_ctl_b030` vs `wf_fmr`;
  - `sw_gbr_ctl_b030` vs `sw_gbr_ctl`, `sw_gfm_ctl_b030` vs `sw_gfm_ctl`, `sw_mlx_ctl_b030` vs
    `sw_mlx_ctl`;
  - the two NNMT W1 arms vs their latest records, reported only.

  Expect equality for all but the NNMT arms. Any difference is investigated and written up before
  T008 is committed.
- [X] T008 Commit the baseline: `baseline.md` (the table), `baseline.json`, `weights_before.json`,
  `fingerprints_before.json`, `legacy_redirects_before.json` and the T003 plans. Then record the
  baseline commit sha in `baseline.md` in a follow-up commit.

**Checkpoint**: baseline committed. Code changes may begin.

---

## Phase 3: User Story 2 - A different model can never load another's weights (P1)

**Goal**: a deterministic key covering every weight-affecting input; post-model settings don't
move it. This phase makes **no loader change**.

**Independent Test**: `pytest Framework/tests/test_model_key.py Framework/tests/test_model_key_audit.py`.

- [X] T009 [P] [US2] Write `Framework/tests/test_model_key.py` first, against stub strategies
  (`object.__new__`, a fake `dp` whitelist, a tmp storage root).
  - The key is deterministic across processes and insensitive to dict order and class name.
  - Each `MODEL_KEY_ATTRIBUTES` member changes the hash (parametrised over the resolved list).
  - Each `MODEL_KEY_IGNORED` member leaves it unchanged.
  - A training-path method overridden in a subclass changes it; an inference method
    (`get_entry_conditions`) does not.
  - Scaler bytes changing changes it.
  - `MODEL_TRAIN_TIMERANGE`, `MODEL_SEED` and `startup_candle_count` change it.
  - `MODEL_TRAINED_WITH={"startup_candle_count": 64}` on a 4096 class gives the 64 class's key.
  - The prefix format and the hash length (16) are as specified.
- [X] T010 [US2] Implement `Framework/ModelKey.py` per data-model.md and contracts/resolver.md:
  - `resolve`, `key_hash`, `key_prefix`, and canonical JSON (enums by name, sorted sets, tuples
    as lists, floats via `repr`);
  - the `MODEL_KEY_*` merge along the MRO (union, never replace);
  - methods → the defining class qualname;
  - classifier class plus knobs; files by sha16, with a (path, mtime, size) cache;
  - run inputs from the strategy (whitelist from `config["exchange"]["pair_whitelist"]`, asserted
    equal to `dp.current_whitelist()` when `dp` is present; `timeframe`; `startup_candle_count`;
    `MODEL_TRAIN_TIMERANGE` or `"unpinned"`; `MODEL_SEED` → `MLX_SEED` → 42);
  - `MODEL_TRAINED_WITH` pins;
  - **resolution point (R11):** list and dict attributes are read from the class's declared
    values (MRO lookup, deep copy), never from the instance's mutated copies;
  - classifier identity comes from `get_classifier_type()` plus a factory class lookup, without
    building the classifier.

  Add a T009 case: mutating an instance's `include_list` after resolution leaves the key
  unchanged.

  It holds no family names. Make T009 pass.
- [X] T011 [US2] Declare the generic lists in `Framework/BaseStrategy.py` / `BaseNNStrategy.py`
  (no other change):
  - `MODEL_KEY_ATTRIBUTES`, `MODEL_KEY_METHODS` and `MODEL_KEY_IGNORED`, from research R1, R3 and
    R5, holding **only names those two classes own**;
  - no entry-framework or family names in `Framework/`;
  - `MODEL_TRAIN_TIMERANGE = None`, `MODEL_SEED = None`, `MODEL_TRAINED_WITH = {}`,
    `MODEL_CODE_VERSION = 1`.

  These are class attributes only, so no behaviour changes and the 029 fingerprint members are
  untouched.
- [X] T012 [P] [US2] Add the extensions, each a merged tuple in the class that owns the names:
  - `utils/EntryFramework.py` (`EntryFrameworkMixin.MODEL_KEY_IGNORED`): `FAMILY_EXIT_OVERRIDES`,
    `FAMILY_EXITS`, `REVERSION_REGIME_GATE`, `ENTRY_MODE`, `TRIGGER_*`, `MODEL_ENTRY_FAMILY`,
    coexistence and pre-emption names. Other `utils/` mixins declare theirs (stop invariants,
    hold window, position adjust).
  - The `NNNC/` mixins that own training inputs declare them in their own `MODEL_KEY_ATTRIBUTES`:
    `LabelParamsMixin`, `PnlLossStrategyMixin`, `PonderStrategyMixin`, `TrainSeedStrategyMixin`,
    `TriggerOnlyMixin`.
  - `NNNC/NNNCStrategy.py`: label-param mixin names, ponder, pnl-loss, aug-dilation.
  - `NNMT/NNMTStrategy.py`: `PROFIT_*`, `_CLASSIFIER_*`, task thresholds, regime spans, and its
    target methods.
  - `NNPredict/NNPredictStrategy.py`: `target_mode`, `ridge_alpha`, coefficient settings and
    methods.
  - `NNWavelet/NNWaveletStrategy.py`: `wavelet_type`, `n_coeffs`, its `maybe_train`.
  - `Anomaly/NNAnomalyStrategy.py`: `get_trading_classes`.
  - `Sklearn/SklearnStrategy.py`: `train_model`.
- [X] T013 [US2] Write `Framework/tests/test_model_key_audit.py` (R5) and make it pass by
  classifying every name it reports into the family lists.
  - It scans the class bodies of all six families, plus `Framework/` and `utils/` mixins, for
    assigned class attributes and `buy_params`/`sell_params` keys.
  - Every name must be in exactly one of the merged ATTRIBUTES or IGNORED lists.
  - Every class that defines a method named on any family's METHODS list must be a class of that
    family.
  - Record each non-obvious classification in research.md R5 as a one-line note.
- [X] T014 [US2] Commit Phase 3, then confirm that `scripts/nnnc_fingerprints.py --compare
  fingerprints_before.json` shows 0 changed.

**Checkpoint**: keys resolve; nothing loads differently yet.

---

## Phase 4: User Story 3 - Existing models keep working, with no retraining (P1)

**Goal**: every existing model is indexed under a key, and no weight byte changes.

**Independent Test**: `migrate --dry-run` gives 79 migrated entries, about 58 legacy entries and 0
conflicts; `verify-bytes` then reports equal.

- [X] T015 [P] [US3] Add tests to `Framework/tests/test_model_key.py` for the manifest and index:
  - write, read and round-trip;
  - `kind=legacy` resolves only for its own class;
  - a hash in `conflicts` is never served;
  - `check_manifest` raises `ModelKeyMismatch` naming the differing fields.
- [X] T016 [US3] Implement the index and manifest functions in `Framework/ModelKey.py`: `locate`
  (the R8 order, *not yet called by the loader*), `check_manifest`, `record_training` and strict
  mode (`FT_MODEL_EVAL_STRICT`), then make T015 pass.
  - Index updates use an atomic write (temp file, then rename).
  - `record_training` holds an exclusive `<location>/.training.lock` while training (R12). A second
    trainer stops with an error naming the holder. Add a T015 case for it.
  - Report mode: `FT_MODEL_KEY=report` (R11).
- [X] T017 [US3] Add `scripts/model_index.py migrate [--dry-run]` (R7).
  - **Walk-forward sets:** for each fold of `plan_nnnc.json` (seed 1, plus the f00/f04/f08/f12
    `_s7` variants), `plan_NNNC_A_Raw_Open.json`, `plan_NNNC_ForwardMAE_Raw.json`,
    `plan_NNNC_GbbRaw_Open.json` and `plan_NNNC_GbbFwdMae_Open.json`:
    - load the fold class, either by generating it in memory from the TEMPLATE with
      `MODEL_TRAIN_TIMERANGE`/`MODEL_SEED`, or by subclassing with those set;
    - build a stub instance with the `config_wf.json` whitelist and resolve its key;
    - write a manifest (`kind=migrated`, `provenance=reconstructed-from-plan`: plan, run_id,
      params_sha, fold) beside the weights;
    - add an index entry pointing at the existing weight file;
    - add the browsing symlink `saved_data/models/<prefix>__<hash>`.
  - **Single-model directories:** `kind=legacy` with key `legacy:<Class>`, and a manifest listing
    the inputs as `unknown`.
  - Empty directories are reported and skipped.
  - **Conflicts:** two directories with one key and different weight sha256 go to the index's
    `conflicts` (exit 3).
  - `--dry-run` writes nothing and prints the table.
- [X] T018 [P] [US3] Write `specs/030-model-fingerprint-naming/legacy_redirects.json` from
  T002's scan: `{class: {defined_in, target, reason}}`, with reasons taken from the overrides'
  docstrings.
  - Add a test to `test_model_key_audit.py` that re-runs the scan and asserts it equals the
    register (minus the BorrowedModelMixin users, which T024 removes).
- [X] T019 [US3] Run `migrate --dry-run` and review it:
  - every fold set maps to 15 distinct keys (19 for WF including `_s7`);
  - fold keys differ only in `run.train_timerange` (and the seed for `_s7`);
  - ARO and GBR differ in at least one non-run input.

  Then run `migrate`.
- [X] T020 [US3] **STOP gate.** Confirm that `migrate` exited 0 (no conflicts) and that
  `verify-bytes weights_before.json` reports equal (manifests and index are excluded).
  - Commit the index snapshot `model_index_migrated.json`, `legacy_redirects.json` and the CLI.
  - On any conflict: stop. Write it up in `verification.md` and mark T021–T033 `[-]` (polish T034–T036 still runs).
- [X] T021 [US3] Offline borrowed-key check: for every `BorrowedModelMixin` user's fold class (the
  twin TGT, SIP, the Live_* and SW_* classes), resolve its key with the mixin removed in memory
  and `MODEL_TRAINED_WITH` applied (the gate arms) and the plan's fold window and seed. Assert it
  equals its `MODEL_SOURCE` fold's migrated key.
  - Record the result table in `verification.md`.
  - Any mismatch: find which input differs. Either classify that input (T013) or add a
    `MODEL_TRAINED_WITH` pin with a written reason. Never force it.

- [X] T021a [US3] Live-key check (R11). Wire `get_model_path` to call `ModelKey.locate` **in
  report mode only** (`FT_MODEL_KEY=report`). It logs the resolved key and the manifest
  comparison, then returns the pre-030 path.
  - Run arm D f00 (`wf_aro_lk030`) and `sw_aro_gate` f00 (`sw_aro_gate_lk030`, which still has its
    mixin) under report mode, as one chain.
  - Assert three things:
    - the logged live key equals the migrated `NNNC_ARO_f00` key, with no mismatched fields;
    - `trades_hash` equals the baseline;
    - there is 0 × `training on`.
  - Any mismatch: fix the resolution (T010/R11) and repeat. Never relax the manifest check.

**Checkpoint**: all models indexed, weights unchanged, borrowed users proven to resolve to their
sources, and the live key equals the migrated key.

---

## Phase 5: User Story 1 - Variants of a model share its weights automatically (P1)

**Goal**: the loader resolves by key, `BorrowedModelMixin` retires, and walk-forward evaluation is
strict.

**Independent Test**: re-run the baseline arms without the mixin. The trades must be identical,
with 0 training.

- [X] T022 [US1] Switch the loader.
  - `BaseNNStrategy.get_model_path()` returns `ModelKey.locate(self).path`. The suffix is still
    `.keras`; the classifiers rewrite it as today.
  - `get_markov_matrix_path` keeps its derivation.
  - `BaseNNStrategy.model_exists` also checks `.safetensors` in its fallback branch, for both the
    keyed and legacy paths.
  - `TrainingEngine.maybe_train` calls `ModelKey.check_manifest` before use when a model exists,
    and `ModelKey.record_training` after a successful `train_model`.
  - In strict mode, `locate` raises `ModelMissing` before `train_model` is reached.
  - The report-only switch from T021a is removed. `FT_MODEL_KEY=report` stays available as a
    diagnostic.
  - `regime/run_arm.py` writes `model_key` (`hash`, `prefix`, `kind`) into arm records (R12). The
    manifest is not written at load.
  - Update `Framework/test_base_nn_strategy.py`'s path assertions. The class name now appears
    through the legacy fallback for an unindexed class with an existing directory; otherwise the
    path is the keyed location.
- [X] T023 [US1] Make the walk-forward harness pin and enforce the key.
  - In `evaluation/run_folds.py`, TEMPLATE writes `MODEL_TRAIN_TIMERANGE = "<train_timerange>"`
    and `MODEL_SEED = <seed>` into every generated fold class. Update the pre-change fixture test
    with a written reason.
  - `model_exists(model_key(fold))` becomes a key-based check through
    `scripts/model_index.py key` logic (imported, not shelled).
  - `evaluate_fold` sets `FT_MODEL_EVAL_STRICT=1`; `train_fold` doesn't.
  - Tests in `evaluation/tests/` cover all three.
- [X] T024 [US1] Retire `BorrowedModelMixin`.
  - Remove it from `NNNC/NNNC_TriggerGate.py`, NNNC_SIProbe, `NNNC_Live_*.py` and `NNNC_SW_*.py`.
    Keep their `MODEL_SOURCE` lines only as documentation comments.
  - Gate arms (`NNNC_Live_Gate`, `NNNC_Live_GateOff`, `NNNC_SW_*_Gate`, `NNNC_SW_*_GateOff`) gain
    `MODEL_TRAINED_WITH = {"startup_candle_count": 64}`.
  - `NNNC/BorrowedModelMixin.py` becomes a shim whose `__init_subclass__` raises "retired in 030:
    weights resolve by model key".
  - Update `NNNC/tests/test_trigger_gate.py`: assert the key equality, not the path string.
- [X] T025 [P] [US1] Write `NNNC/tests/test_model_key_nnnc.py`:
  - an arm D subclass changing only `FAMILY_EXIT_OVERRIDES`, with no mixin, resolves to
    `NNNC_ARO_fNN`'s path when given f03's window and seed;
  - a renamed copy of arm D (same inputs) resolves to the same key (SC-005);
  - a gain-target change gives a different key, and under strict mode `locate` raises
    `ModelMissing`.
- [X] T026 [US1] Point `scripts/test_strat.sh -f` at the resolved path: delete and post-check the
  file printed by `scripts/model_index.py path <Family> <Class>` (R10). Smoke-test it with a stub
  class in a tmp storage root (no real training).
- [X] T027 [US1] Commit Phase 5, then run `pytest Framework utils NNNC/tests evaluation
  SimpleTrigger/tests`. All must pass, apart from the 3 known training-pipeline golden failures.

---

## Phase 6: User Story 4 - A model directory says what it holds (P2)

- [X] T028 [P] [US4] Add `scripts/model_index.py key <Class> [--fold fNN --plan P]` and `path`
  (FR-007). They print the prefix, hash, canonical key and location, and whether weights exist
  (exit 0 or 1).
  - Smoke-test them on `NNNC_A_Raw_Open --fold f00 --plan NNNC_A_Raw_Open`, on `NNNC_MLX_Open` (a
    legacy redirect) and on `NNMT_Ctl_s42` (legacy).
  - FR-014 check: `NNNC_MLX_Open_s1` and `NNNC_MLX_Open_s7` (a legacy `_s1`/`_s7` pair),
    `NNNC_MLX_L024` (legacy own weights) and `NNNC_WF` f04 at seed 7 (`--plan nnnc --seed 7`) each
    resolve to exactly their pre-030 directory.
- [X] T029 [P] [US4] Write a manifest readability check: confirm that `saved_data/models/` shows a
  readable symlink name for each migrated model, and that each manifest lists its inputs, window,
  seed, scaler hashes and users.

---

## Phase 7: Verification (SC-001 … SC-008)

- [X] T030 Run the after chain: the same ten arms as T006 (eight NNNC plans, two NNMT W1), under
  `_a030` tags, as one chain with `FT_MODEL_EVAL_STRICT=1` exported. Grep for `training on`
  (expect 0).
- [X] T031 Run `baseline_check.py` after vs baseline. Every NNNC arm must match 15/15
  (`sw_aro_gate` and `sw_mlx_c2` now run without the mixin), and the NNMT W1 arms must be equal.
  Then run `verify-bytes weights_before.json` (equal) and `nnnc_fingerprints.py --compare` (0
  changed).
  - FR-022: `git diff --name-only <baseline sha>..HEAD` lists no file of the live bot's strategy
    (`BasketStrategy` and its imports).
- [X] T032 Strict-mode and weightless-family checks (SC-008, R9):
  - for each of NNPredict, NNWavelet, Anomaly and Sklearn, run `scripts/model_index.py key` on
    every class (it resolves, and the hashes are distinct where inputs differ);
  - one `FT_MODEL_EVAL_STRICT=1` backtest per family on a 1-day timerange must halt with
    `ModelMissing` and create no directory;
  - one arm D subclass with a changed gain must also halt.
- [X] T033 Write `verification.md`: tables for T021, T031 and T032; the migration summary
  (migrated, legacy, conflicts, empty); and the spec amendment for FR-018a (R9), stated as a
  deviation.

---

## Phase 8: Polish & cross-cutting

- [X] T034 [P] Update `AGENT_GUIDE.md`:
  - how models are keyed;
  - `MODEL_TRAINED_WITH`;
  - when to bump `MODEL_CODE_VERSION`;
  - the `model_index.py` commands;
  - that `BorrowedModelMixin` is retired.
- [X] T035 Write `report.md` (the SC table and commits), then update memory:
  `project_next_model_naming.md` becomes `project_030_model_naming.md` (DONE), and the MEMORY.md
  index is updated.
- [X] T036 Mark all tasks `[X]`/`[-]` and make the final commit.

---

## Dependencies & Execution Order

- **Phases 1–2** come before any code change (T008 gates everything).
- **US2 (Phase 3)** → **US3 (Phase 4)**: migration needs the resolver.
- **US3's T020 STOP gate** → **US1 (Phase 5)**: the loader is switched only after a clean
  migration.
- **US4 (Phase 6)** can run after T016, in parallel with Phase 5.
- **Phase 7** follows Phases 5–6. **Phase 8** comes last.

**Parallel opportunities**:
- T001, T002 and T003;
- T009 alongside the T012 family lists;
- T015 and T018;
- T025, T028 and T029.

## Implementation Strategy

The MVP is US2 plus US3: keys and a migrated index, with no loader change. That alone gives the
manifests, the audit and the conflict check. US1 then switches the loader. Verification decides
the feature.
