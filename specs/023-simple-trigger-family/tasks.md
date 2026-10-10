---
description: "Task list for 023 SimpleTrigger strategy family"
---

# Tasks: SimpleTrigger Strategy Family

**Input**: Design documents from `specs/023-simple-trigger-family/`. These are `plan.md`,
`spec.md`, `research.md` (R1–R13), `data-model.md`, `contracts/interfaces.md` (C1–C5) and
`quickstart.md` (§0–§8).

**Tests**: INCLUDED. The spec's acceptance criteria are identity tests (trade sets, fingerprints),
and quickstart §0 names the unit gate.

**Paths**: All paths are relative to `user_data/strategies/` (the nested repo). Python is
`../../.venv/bin/python` (3.12). Tests run with `../../.venv/bin/python -m pytest <path> -q`.

**Standing rules for every task**:
- Stage named files only; never `git add -A` or `git add .`. Commit to master.
- Run ONE backtest chain at a time, and check `ps` first (parallel runs caused OOM and Metal
  faults).
- Parameter dicts merge from the parent (`{**Parent.buy_params, ...}`) and are never fresh
  literals.
- Framework/ gets no strategy-specific code.
- A backtest "result" with `returncode != 0` is not a result.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 Create `SimpleTrigger/__init__.py`, `SimpleTrigger/tests/__init__.py` and
  `evaluation/tests/__init__.py`. Add `SimpleTrigger/tests/conftest.py` and
  `evaluation/tests/conftest.py`, each inserting the repo root (`user_data/strategies`) on
  `sys.path` the same way `Framework/tests/` does (read one of its test files for the pattern).
  Add to `.gitignore`: `SimpleTrigger/SimpleTrigger_*__C_*.py` (generated component arms, T042)
  and `evaluation/.scratch/`.
- [ ] T002 **Baseline reproducibility, before any edit.**
  - Re-run two recorded folds on HEAD, one at a time:
    - arm D fold f00: `python -m regime.run_arm --strategy NNNC_ARO_f00 --timerange <f00
      test_timerange from evaluation/folds/plan_NNNC_A_Raw_Open.json> --label F00 --tag
      base023_aro`
    - `wf_gbbo` fold f00: strategy `GbbOpen`, same timerange, tag `base023_gbbo`.
  - Assert each new `trades_hash` equals the recorded one in
    `regime/arms/wf_aro_f00_F00.json` / `regime/arms/wf_gbbo_f00_F00.json`.
  - Write `specs/023-simple-trigger-family/baseline.md` with both hashes and the commands.
  - **If either differs, STOP and report.** Nothing later in this feature can be judged against
    a baseline that doesn't reproduce.

---

## Phase 2: Foundational (instrumentation; blocks all stories)

**Purpose**: The measurements every story depends on. Every shared-file edit here must leave
recorded arms byte-identical (T014 gate).

- [ ] T003 [P] Write `evaluation/tests/test_fingerprint.py`. Assert that:
  (a) `resolve(NNNC_A_Raw_Open)` and `resolve(NNNC_ARO_f00)` give the same `framework_sha`;
  (b) the resolved dict contains every member listed under data-model `FrameworkPreset`;
  (c) `prediction_threshold`, the classifier and label params are EXCLUDED;
  (d) changing one stop attribute on a throwaway subclass changes the sha;
  (e) a JSON `params.sell` override is applied the way freqtrade merges it
      (`freqtrade/strategy/hyper.py:86-100`, nested dict update);
  (f) resolving `NNNC_A_Raw_Open` does NOT instantiate the strategy and does NOT touch
      `saved_data/` (patch `BaseNNStrategy.__init__` to raise).
- [ ] T004 Implement `evaluation/fingerprint.py`:
  - `resolve(strategy_cls, params_json: dict | None, config: dict) -> (sha16, resolved_dict)`
    reads MRO class attributes for the FrameworkPreset members, applies the JSON merge, and adds
    `max_open_trades`, `stake_amount` and `fee` from config. `max_open_trades` comes from
    params_json `max_open_trades` if present (`hyper.py:68`), else from config.
  - The canonical form is `json.dumps(sort_keys=True)` and sha256[:16].
  - Add `load_strategy_class(name)` that imports by file using `regime.run_arm.strategy_group`
    without instantiating.
  - Add a CLI `python -m evaluation.fingerprint --strategy CLS [--config PATH]` (C5).
  - Make T003 pass.
- [ ] T005 [P] Write `evaluation/tests/test_tag_match.py`. Create temp records
  `wf_x_f00_F00.json` and `wf_x_f00_s7_F00.json`, then assert that `pool.arm_record` and
  `run_folds.arm_record` return exactly `wf_x_f00_F00.json` for tag `wf_x_f00`, label `F00`
  (research R6).
- [ ] T006 Fix the tag glob in `evaluation/pool.py` (`arm_record`, ~L47) and
  `evaluation/run_folds.py` (`arm_record`, ~L211) to open `{tag}_{label}.json` by exact name. Make
  T005 pass.
- [ ] T007 [P] Write `Framework/tests/test_entry_condition_sink.py`.
  - Build a small synthetic dataframe with the indicator columns `populate_entry_trend` reads (see
    `Framework/BaseStrategy.py:903-975`), and a minimal BaseStrategy subclass whose
    `get_entry_conditions` returns a fixed mask.
  - Assert the `enter_long` column is identical with `FT_ENTRY_CONDITION_SINK` unset and set.
  - With it set, assert the JSON written to the sink dir has per-condition
    `{"pass": int, "total": int}` for `signal`, `volume` and `rvol`, plus the quality guards when
    `entry_enable_guards` is True, and that the counts are correct.
- [ ] T008 Add the generic condition sink to `Framework/BaseStrategy.populate_entry_trend` (C4):
  - Name each mask as it is ANDed, then at the end, if `os.environ.get("FT_ENTRY_CONDITION_SINK")`
    is set, write `<dir>/<ClassName>_<pair with / replaced by _>.json`.
  - Make no other change. The masks and their combination order stay byte-identical.
  - Make T007 pass.
- [ ] T009 [P] Write `evaluation/tests/test_foldplan_nomodel.py`. Assert that:
  (a) `derive(src_plan, family="SimpleTrigger", base="X", tag="st_x")` copies every fold's
      `train_*`, `test_*` and `*_timerange` fields verbatim from
      `evaluation/folds/plan_NNNC_A_Raw_Open.json`, and copies `config`;
  (b) the derived plan has `no_model: true`, `model_dir: null`, `strategy_class: "X"` for every
      fold, and `arm_tag` `st_x_fNN`;
  (c) `validate()` accepts the duplicate class only when `no_model` is true and every `model_dir`
      is null, and still rejects it otherwise;
  (d) a no-model plan with a non-empty `thresholds` key is rejected;
  (e) `derive(..., family="NNNC", base="NNNC_TriggerGate", tag="wf_trgate",
      model_source="NNNC_ARO")` emits per-fold `strategy_class` `NNNC_TGT_fNN` and
      `model_dir` `NNNC_ARO_fNN`.
- [ ] T010 Implement the `derive` subcommand and the validate rule in `evaluation/foldplan.py`
  (contract C2). Store `params_snapshot`/`params_sha` as null for no-model plans. Make T009 pass.
- [ ] T011 [P] Write `evaluation/tests/test_run_folds_nomodel.py` using `--dry-run` and
  monkeypatched `evaluate_fold`/`train_fold`/`generate_fold`. For a no-model plan, assert that
  `generate_fold`, `train_fold` and `model_exists` are never called, `evaluate_fold` is called 15
  times with the base class, and a class whose StrategyConfig has `needs_training=True` aborts
  before the first evaluation. For an NNNC plan with `model_dir`, assert that `model_exists`
  checks `saved_data/<model_dir>` and not `saved_data/<strategy_class>`.
- [ ] T012 Implement the no-model path in `evaluation/run_folds.py` `main()` (~L220): skip
  generate, train and model check when `plan["no_model"]`, and assert `needs_training` is False
  via `evaluation.fingerprint.load_strategy_class`. Change `model_exists` to take the fold's
  `model_dir` when present, defaulting to the class. Make T011 pass.
- [ ] T013 Extend `regime/run_arm.py` record (contract C3):
  - add `framework_sha` and `framework_resolved` (via `evaluation.fingerprint.resolve` on the
    strategy class, the scratch params JSON and the config);
  - add `exit_reason_counts` (count `exit_reason` over the archive's trades in `summarize`, ~L291);
  - add `entry_condition_counts` (aggregate sink files when `FT_ENTRY_CONDITION_SINK` is set, with
    the sink dir set per run under `regime/.scratch/<tag>_<label>/sink`);
  - add `no_model`, `preset`, `exit_mode`, `trigger`, `label_method` and `coupled_resolution`,
    read from class attributes when present, else null.

  Do not change how `trades_hash` or `exit_config_hash` are computed. Add
  `evaluation/tests/test_run_arm_record.py` covering `exit_reason_counts` on a fixture trade list.
- [ ] T014 [P] Implement `evaluation/halves.py`:
  - `split(arm_tag, ctl_tag, split_after="f07")` returns `{early, late}`, each with per-half summed
    pp, positive folds, and the trades-only paired `bootstrap_delta` from `evaluation/bootstrap.py`
    (~L82) on that half's trades.
  - Add a CLI (C5).
  - Add `evaluation/tests/test_halves.py` on fixture records, asserting f00–f07 / f08–f14
    membership and the sums.
- [ ] T015 **Gate: shared edits are neutral.**
  - Re-run the two T002 folds on the edited code, both with the sink unset and with the sink set.
  - All four `trades_hash` values must equal the T002 baselines, and the sink run's record must
    hold `entry_condition_counts`.
  - Record the results in `baseline.md`.
  - **If any differ, STOP**: a shared edit changed behaviour.

**Checkpoint**: The instrumentation is proven neutral, and stories can start.

---

## Phase 3: User Story 1 — Any causal trigger through the NNNC framework, no model (P1) 🎯 MVP

**Goal**: The SimpleTrigger family runs any measured-causal trigger, and reproduces 021/022
exactly under the legacy preset.

**Independent Test**: quickstart §1–§3. All 7 legacy arms match their recorded per-fold
`trades_hash` in 15/15 folds. `saved_data/` gains nothing.

### Causality (SC-006)

- [ ] T016 [P] [US1] Write `research/outcomes/tests/test_causality.py`.
  - On a short real-feather slice (one pair, ~5k bars), assert that `check_trigger("gbb", ...)`
    returns PASS, and `check_trigger` on a planted forward-reading trigger (registered in-test
    with `replace=True`, reading `close.shift(-3)`) returns FAIL with `first_mismatch` set.
  - Assert the CLI exits 1 when the control does not FAIL. Simulate this by monkeypatching the
    control to a causal trigger.
- [ ] T017 [US1] Implement `research/outcomes/causality.py` (research R11), modelled on
  `research/labelfree/metalabel/audit.py:50-102`.
  - For each trigger, pair and era, compute populator features + trigger on the full history and
    on prefixes cut at K points (default K=8).
  - Compare over each prefix excluding `warmup_bars` (default `startup_candle_count` = 200) with
    exact equality (RTOL 0).
  - Mark `UNSCOREABLE` when `required_columns` are absent from the populator output.
  - Write rows (data-model `TriggerCausality`, including `code_sha` = `git hash-object` of
    `utils/DataframePopulator.py`, `research/outcomes/builtin_triggers.py`,
    `research/outcomes/audit_triggers.py` and `research/outcomes/causality.py`) to
    `research/outcomes/trigger_causality.json` with `--write`.
  - Pairs default to 3 from `config/config_wf.json`'s whitelist. Eras are 3 non-adjacent years in
    2021–2026.
  - Use the same feature-population call the strategies use (find it via `BaseStrategy` /
    `DataframePopulator`).
- [ ] T018 [US1] Run `python -m research.outcomes.causality --write` and commit
  `research/outcomes/trigger_causality.json`.
  - Expected: control `local_min_followthrough` FAIL; `trends` and `bands` UNSCOREABLE.
  - Report every other non-PASS trigger to the user. Do not "fix" a trigger to make it pass
    without asking.
- [ ] T019 [US1] Resolve the `future_*` record conflict (research R10).
  - Read `labels_future_sharpe`, `labels_future_sortino` and `labels_future_expectancy` in
    `Framework/TrainingSignals.py` to decide whether each one's entry condition reduces to the
    `adx` trigger.
  - Fix whichever is stale: `research/outcomes/tests/test_labelmethod_inventory.py` (`:74-76`,
    count at `:154`) or `specs/021-label-exit-coordination/label-method-audit.md` (`:69-71`).
  - Record the reading in `research.md` R10 as an addendum.

### Mapping, presets, family

- [ ] T020 [P] [US1] Write `SimpleTrigger/tests/test_label_map.py`. Assert that
  `SimpleTrigger.label_map.MAPPING` has exactly one row per `LabelMethod` member (iterate the
  enum), that every `status: "unsupported"` row has a non-empty `reason` and a null `trigger`,
  that every non-null `trigger` is a registered trigger name, and that the `empty` rows are
  exactly the 8 of research R10.
- [ ] T021 [P] [US1] Implement `SimpleTrigger/label_map.py`. The data-model `LabelTriggerMapping`
  rows come from research R10, as amended by T019:
  - `local_min_followthrough` is unsupported, reason "centred window reads 3 bars ahead";
  - `trends` and `bands` are unsupported, reason "unscoreable: required columns absent from
    populator";
  - the 3 `future_*` rows follow T019's outcome.

  Make T020 pass.
- [ ] T022 [P] [US1] Write `SimpleTrigger/tests/test_presets.py`. Assert that:
  (a) every preset contains every FrameworkPreset member, and a missing key raises on load (data
      model: "a preset is total");
  (b) `ARM_D_OPEN` resolves, via `evaluation.fingerprint.resolve` on a bare preset-carrying
      `SimpleTriggerStrategy` subclass, to the same `framework_sha` as `NNNC_A_Raw_Open` with its
      JSON;
  (c) `LEGACY_TRIGGERONLY` resolves to the same sha as `SimpleStrategies.TriggerOnly.TriggerOnlyStrategy`;
  (d) the two presets differ in `use_atr_adaptive_stoploss`, `atr_stoploss_{multiplier,floor,cap}`
      and `startup_candle_count` (research R1 table), which pins the finding.
- [ ] T023 [P] [US1] Implement `SimpleTrigger/presets.py` with `ARM_D_OPEN` and
  `LEGACY_TRIGGERONLY`. Take the literal values by reading the resolved attributes (run
  `python -m evaluation.fingerprint` on both reference classes and copy them). Add
  `get(name) -> dict` and `validate(preset)`. Make T022 pass.
- [ ] T024 [P] [US1] Write `SimpleTrigger/tests/test_simple_trigger_strategy.py`. Assert that:
  (a) the StrategyConfig has `needs_training is False` and model_type NONE (G1);
  (b) `get_entry_conditions` returns `triggers.evaluate("gbb", df)` for `TRIGGER_NAME="gbb"`;
  (c) an unset, unregistered, non-PASS or stale-`code_sha` trigger returns an all-False mask and
      prints a line starting `*** SimpleTrigger:` (G2);
  (d) `buy_params`/`sell_params` contain every key of `BaseStrategy`'s dicts plus the preset's,
      and none are dropped (FR-019);
  (e) the class has no pair-name references (grep the source for `/USD` and `pair ==`) (G6).
- [ ] T025 [US1] Implement `SimpleTrigger/SimpleTriggerStrategy.py` per contract C1, G1–G3 and G6
  (G4/G5 come in US4).
  - It subclasses `Framework.BaseStrategy.BaseStrategy`. At class-definition time it applies
    `presets.get(PRESET)` to class attributes and merges the dicts as
    `{**BaseStrategy.buy_params, **preset_buy, **getattr(cls, "BUY_OVERRIDES", {})}` (same for
    sell) in `__init_subclass__`.
  - Entry goes through the registry exactly as `SimpleStrategies/TriggerOnly.py:139-168` does,
    plus the causality gate: it reads `research/outcomes/trigger_causality.json` and requires
    PASS with a matching `code_sha`.
  - Set `enter_tag` to the trigger name.
  - Make T024 pass.
- [ ] T026 [US1] Create the 7 legacy configs in `SimpleTrigger/`, each `PRESET =
  "LEGACY_TRIGGERONLY"`:
  - `SimpleTrigger_Legacy_Breakout` (`breakout`)
  - `SimpleTrigger_Legacy_BreakoutVol` (`breakout_vol`)
  - `SimpleTrigger_Legacy_BreakoutBb` (`breakout_bb`)
  - `SimpleTrigger_Legacy_BreakoutGbb` (`breakout_gbb`)
  - `SimpleTrigger_Legacy_Geometry` (`geometry`)
  - `SimpleTrigger_Legacy_Gbb` (`gbb`)
  - `SimpleTrigger_Legacy_GbbRobust` (`gbb` + `BUY_OVERRIDES` copied from
    `SimpleStrategies/GbbRobustOpen.py:49-60`: atr_pct 0.013, adx 20.0, rvol 1.9, others as
    there).

  Copy each original's class-level overrides from its `SimpleStrategies/*Open.py` file, and
  confirm with `python -m evaluation.fingerprint` that each legacy config's `framework_sha` equals
  its original's.
- [ ] T027 [US1] Derive the 7 no-model plans with
  `python -m evaluation.foldplan derive --from evaluation/folds/plan_NNNC_A_Raw_Open.json
  --family SimpleTrigger --base <cls> --tag <tag> --out evaluation/folds/plan_<tag>.json`. Tags:
  `stl_bo`, `stl_bvo`, `stl_bbo`, `stl_bgo`, `stl_geo`, `stl_gbbo`, `stl_gbbr`. Then confirm the
  fold windows equal those of each original 021 plan (`evaluation/folds/plan_{BreakoutOpen,...,GbbOpen}.json`).
- [ ] T028 [US1] Run the 7 plans **sequentially** (`python -m evaluation.run_folds --plan ...`,
  105 backtests total). Afterwards assert that no `saved_data/SimpleTrigger_*` directory exists.
- [ ] T029 [US1] Write `SimpleTrigger/tests/test_legacy_reproduction.py`, a data test over
  `regime/arms/`. For each of the 7 pairs (`stl_X` vs the recorded `wf_X`), assert that
  `trades_hash` is equal in every fold. Treat `wf_gbbg` f10 (0 trades, no hash) as not
  applicable; that arm is not reproduced here. Also assert `stl_gbbr` totals 458 trades.
  - **On a mismatch, STOP**: localise it by diffing `framework_resolved` and then the first
    divergent trade, and report before changing anything.
- [ ] T030 [US1] Quickstart §3 (SC-008): diff the resolved parameters of
  `Basket/MomentumRegimeBasket15mFast` (via `evaluation.fingerprint.resolve`, or a class attribute
  dump if it falls outside the preset members) between `git stash`-free HEAD~N (pre-023) and now,
  using `git worktree` for the old revision. Expected: empty. Record the result in `baseline.md`.

**Checkpoint**: US1 is complete, so the family is validated against every recorded no-model
result.

---

## Phase 4: User Story 3 — Model contribution vs an identical no-model twin (P2, the live thread)

*Ordered before US2 because it is the project's one live thread (plan phasing). US2 doesn't depend
on it.*

**Goal**: Prove the twin shares arm D's framework, then measure the model's contribution against
post-fix arm D (1420 trades, +42.71pp, 14/15, research R2).

**Independent Test**: quickstart §5–§6.

- [ ] T031 [P] [US3] Create `SimpleTrigger/SimpleTrigger_Gbb.py` (`TRIGGER_NAME = "gbb"`, default
  `ARM_D_OPEN`). Add to `SimpleTrigger/tests/test_presets.py` an assertion that its
  `framework_sha` equals `NNNC_A_Raw_Open`'s (FR-013, TwinPair rule).
- [ ] T032 [P] [US3] Write `NNNC/tests/test_trigger_gate.py` (create `NNNC/tests/__init__.py` if
  absent). For `NNNC_TGT_f03` (generated in T034), assert that `get_model_path()` points at
  `saved_data/NNNC_ARO_f03/NNNC_ARO_f03.keras`, that the model-existence check is True without
  creating `saved_data/NNNC_TGT_f03/`, and that any training entry point raises
  `RuntimeError("NNNC_TriggerGate must never train")`.
- [ ] T033 [US3] Implement `NNNC/NNNC_TriggerGate.py` (research R12), subclassing
  `NNNC_A_Raw_Open`:
  - `get_entry_conditions` calls `super().get_entry_conditions(dataframe)`, which keeps
    `predict_*` columns and diagnostics, and returns
    `pd.Series(triggers.evaluate(self.TRIGGER_NAME, dataframe).to_numpy(), index=dataframe.index)`.
  - Set `TRIGGER_NAME = "gbb"` and `MODEL_SOURCE = "NNNC_ARO"`.
  - `get_model_path` maps a class named `*_fNN` to
    `<storage>/{MODEL_SOURCE}_fNN/{MODEL_SOURCE}_fNN.keras` and raises when the fold suffix is
    missing or the dir is absent.
  - Override the training entry point (`maybe_train` / whatever `BaseNNStrategy.populate_indicators`
    calls when `training_needed`, ~L1374-1380) to raise.
  - The main scaler is shared at `get_storage_location()`, the same one arm D used. Note this in
    the docstring.
  - Make T032 pass after T034.
- [ ] T034 [US3] Derive and run the twin plans sequentially:
  - `SimpleTrigger_Gbb` uses a no-model plan with tag `st_gbb`.
  - `NNNC_TriggerGate` uses `derive --family NNNC --base NNNC_TriggerGate --tag wf_trgate
    --model-source NNNC_ARO` (fold classes `NNNC_TGT_fNN`, generated by `run_folds`, model check
    on `NNNC_ARO_fNN`). Confirm in the run log that no training occurred.
- [ ] T035 [P] [US3] Write `evaluation/tests/test_twins.py` on fixture records, covering contract C5:
  - `check` exits 2 on a `framework_sha` mismatch, 3 on any per-fold `trades_hash` mismatch between
    the equivalence and simple tags, and 0 otherwise;
  - `compare` refuses with no passing `check` artefact;
  - `compare` output has `delta_pp`, `ci`, `verdict`, `halves`, and `shared_fraction` ("shared"
    from `bootstrap_delta` over total trades);
  - an `INCONCLUSIVE_*` or underpowered verdict from `compare_pooled.compare` gives no directional
    verdict (FR-014).
- [ ] T036 [US3] Implement `evaluation/twins.py`:
  - `check` writes `evaluation/twins/<simple_tag>.check.json`.
  - `compare` uses `evaluation/compare_pooled.py:compare` on pooled files (build them with
    `evaluation/pool.py` for `st_gbb` and `wf_aro` if absent), plus `evaluation.halves.split`.
    It writes `evaluation/twins/<simple_tag>.compare.json` and `.md`.
  - Make T035 pass.
- [ ] T037 [US3] Run `python -m evaluation.twins check --nnnc-tag wf_aro --simple-tag st_gbb
  --equivalence-tag wf_trgate`.
  - **On exit 3, STOP**: localise with the `framework_resolved` diff, the `entry_condition_counts`
    diff (does `BaseNNStrategy.add_additional_indicators` change a guard column?) and the first
    divergent trade. Report the parity defect to the user before fixing anything, because fixing
    it may touch NNNC.
- [ ] T038 [US3] Run `python -m evaluation.twins compare --nnnc-tag wf_aro --simple-tag st_gbb` and
  commit the compare artefacts. The reference must be post-fix `wf_aro`; assert its pooled
  `n_trades_pooled == 1420`.

**Checkpoint**: The model's contribution is measured, or explicitly unresolvable at this power.

---

## Phase 5: User Story 2 — Isolate and evaluate each framework component (P2)

**Goal**: Per-component attribution with liveness proof, so an INERT component is a flagged
defect.

**Independent Test**: quickstart §7. Every component has a row. Under `LEGACY_TRIGGERONLY`,
`stop_atr_band` and `stop_volume_confirm` come out INERT (the R1 finding, measured).

- [ ] T039 [P] [US2] Write `SimpleTrigger/tests/test_components.py`. Assert that:
  - every component in `SimpleTrigger.components.COMPONENTS` has non-empty `overrides` and a
    `liveness_metric`;
  - for each component, the fingerprint diff between the full `SimpleTrigger_Gbb` and its
    neutralised variant touches ONLY keys in that component's `overrides`;
  - under `ARM_D_OPEN` every `guard_*` component is listed in `neutral_is_identity_under`.
- [ ] T040 [P] [US2] Implement `SimpleTrigger/components.py`, the data-model `Component` table.
  Components and their neutral overrides:
  - `guard_adx` → `entry_adx_threshold: -1.0`
  - `guard_atr_pct` → `entry_atr_pct: -1.0`
  - `guard_bb_width` → `entry_bb_width_threshold: -1.0`
  - `guard_close_norm` → `entry_close_norm_threshold: 1.1`
  - `guard_guard` → `entry_guard_threshold: 1.1`
  - `guard_rvol` → `entry_rvol_threshold: -1.0`
  - `take_profit` → `cexit_take_profit: 10.0`
  - `max_hold` → `cexit_max_days: 3650`
  - `roi` → `minimal_roi: {"0": 10.0}`
  - `exit_signal` → `enable_exit_signal: False`
  - `stop_atr_band` → `use_atr_adaptive_stoploss: False`
  - `stop_grace` → `stoploss_grace_hours: 0`
  - `stop_volume_confirm` → `use_volume_confirmation_stoploss: False`

  `trigger` → the `empty` trigger. Liveness metrics: guards → `entry_condition_counts.<guard>`;
  exits → `exit_reason_counts.<reason>`; stops → `trades_hash` plus the `stop_loss` /
  `trailing_stop_loss` counts. **Verify each override key name against `BaseStrategy`** before
  committing, because a misspelt key is silently inert. Make T039 pass.
- [ ] T041 [P] [US2] Write `evaluation/tests/test_attribution.py` on fixture records. Assert that:
  - a component whose arm has an identical `trades_hash` in every fold is `INERT`;
  - one listed in `neutral_is_identity_under` for the preset is `ALREADY_NEUTRAL` and has no arm;
  - a zero-trade arm is `ZERO_TRADES`;
  - every `LIVE` row carries `liveness_evidence` with its before/after values;
  - rows carry `halves` and `fold_detail`.
- [ ] T042 [US2] Implement `evaluation/attribution.py` (contract C5):
  - `--full-tag T --base-class C --components all|a,b --preset P` writes one generated subclass
    per component, `SimpleTrigger/SimpleTrigger_<C>__C_<component>.py` (gitignored, T001), with
    overrides merged.
  - It derives no-model plans (tag `<T>__<component>`) and, with `--run`, runs them
    sequentially.
  - It then builds `evaluation/attribution_<T>.json` and `.md` using `compare_pooled.compare`
    and `halves.split`.
  - Make T041 pass.
- [ ] T043 [US2] Run attribution for `SimpleTrigger_Gbb` under `ARM_D_OPEN`, with the sink
  enabled, for all non-ALREADY_NEUTRAL components (sequentially). Commit the report.
- [ ] T044 [US2] Run attribution for `SimpleTrigger_Legacy_Gbb` under `LEGACY_TRIGGERONLY`, stop
  components only. Expected: `stop_atr_band` and `stop_volume_confirm` INERT. If either is LIVE,
  research R1 is wrong; report it.

**Checkpoint**: Every component's contribution and wiring is measured.

---

## Phase 6: User Story 4 — Coupled vs decoupled stop and targets (P3)

**Goal**: Coupled mode as a pure parameter mapping, plus runtime stop invariants.

**Independent Test**: quickstart §8. Coupled with matched targets is hash-identical to decoupled
in 15/15 folds, and there are zero `StopInvariantError`s.

- [ ] T045 [P] [US4] Write `SimpleTrigger/tests/test_coupling.py`. Assert that:
  - `coupling.resolve(LabelMethod.gbb, min_gain=g, horizon=h, fee_per_side=f, timeframe="15m")`
    returns `take_profit_net == utils.ExitPolicy.net_from_price_ratio(1 + g, f)`,
    `max_days == h * 15 / 1440`, and `coupled_stop == "none-in-label"` with the stop fields
    absent;
  - for `forward_mae` with `max_drawdown=d` it returns a static stop `-d` with
    `use_atr_adaptive_stoploss False` and `coupled_stop == "label"`;
  - `resolve` on an unsupported `LabelMethod` (per `label_map`) raises;
  - the round trip holds: choosing `g` from `utils.ExitPolicy.price_ratio_for_net(tp, f) - 1`
    gives back `tp` to 1e-12.
- [ ] T046 [US4] Implement `SimpleTrigger/coupling.py` (research R8). It produces parameter values
  only (G5). Wire it in `SimpleTriggerStrategy.__init_subclass__`: when `EXIT_MODE == "coupled"`,
  merge `resolve(...)` into `sell_params` and the stop attributes, and store the resolution on a
  class attribute `COUPLED_RESOLUTION`, which `run_arm` records (T013). Make T045 pass.
- [ ] T047 [P] [US4] Write `SimpleTrigger/tests/test_stop_invariants.py`. Drive
  `SimpleTriggerStrategy.custom_stoploss` with a stubbed `super()` returning:
  (a) a positive ratio (planted post-grace sign flip, cf.
      `Framework/tests/test_post_grace_stop_signfix.py`), which must raise `StopInvariantError`;
  (b) a ratio whose implied stop price exceeds `current_rate`, which must raise;
  (c) after grace, a ratio looser than `atr_stoploss_floor` (adaptive) or `stoploss` (static),
      which must raise;
  (d) valid values, which must pass through unchanged.
- [ ] T048 [US4] Implement `StopInvariantError` and the `custom_stoploss` wrapper in
  `SimpleTrigger/SimpleTriggerStrategy.py` (research R9, G4). It must return `super()`'s value
  unchanged when valid. Make T047 pass. Then re-run T029's `stl_gbbo` fold f00 to confirm the
  wrapper is behaviour-neutral (identical hash).
- [ ] T049 [US4] Create `SimpleTrigger/SimpleTrigger_GbbCoupled.py` (`EXIT_MODE = "coupled"`,
  `LABEL_METHOD = gbb`). Choose `min_gain` via `price_ratio_for_net(ARM_D_OPEN cexit_take_profit,
  fee) - 1` and `horizon` so that `max_days` equals the preset's `cexit_max_days`. Assert its
  `framework_sha` equals `SimpleTrigger_Gbb`'s.
- [ ] T050 [US4] Run `SimpleTrigger_GbbCoupled` (no-model plan, tag `st_gbbc`). Assert per-fold
  `trades_hash` equals `st_gbb` in 15/15, `coupled_resolution.coupled_stop == "none-in-label"`, and
  that no run log in T028/T034/T043/T044/T050 contains `StopInvariantError`.

**Checkpoint**: Coupling is possible and proven equivalent where it should be. The investigation
itself is follow-up work.

---

## Phase 7: Polish & cross-cutting

- [ ] T051 [P] Correct the false comment at `SimpleStrategies/TriggerOnly.py:72-75` ("the Open
  operating point, verbatim"). State that its STOP is static −0.05 with ATR-adaptive off, unlike
  arm D, and cite 023 research R1. Comment only; no value changes, since the legacy hashes depend
  on it.
- [ ] T052 [P] Run `ruff check` and `ruff format --check` on every new or edited Python file
  (ruff is the formatter of record per `AGENT_GUIDE.md`).
- [ ] T053 Run the full unit gate (quickstart §0) and confirm it passes.
- [ ] T054 Write `specs/023-simple-trigger-family/report.md`. Start with a SUMMARY covering:
  - SC-001 to SC-008 status with evidence paths;
  - the US3 model-contribution result, quoted with its verdict, CI, halves and shared fraction,
    and the Principle II caveat (no distant era; halves split used);
  - the attribution table;
  - research corrections R1 and R2;
  - the **FR-007a follow-up**: restructure NNNC as SimpleTrigger plus a model gate, now that
    FR-008 has passed or failed;
  - filed notes: `evaluation/pick_guards.floor_for_window` has zero callers (L177 still uses the
    absolute 25), and `guard_trackability.py` is referenced by `run_folds.py` but missing.
- [ ] T055 Update project memory: a new `project_023_simple_trigger.md` with the verdict, the
  presets, the arm D twin result and how to run a SimpleTrigger arm. Add a one-line index entry
  in `MEMORY.md`, and update `project_status_2026_10_10.md`'s live thread with the US3 result.

---

## Dependencies & Execution Order

- **Setup (T001–T002)**: T002 must pass before anything else.
- **Foundational (T003–T015)**: blocks every story. T015 is the gate.
- **US1 (T016–T030)**: needs Phase 2. Causality (T016–T019) precedes T025 (the strategy's
  causality gate reads the JSON).
- **US3 (T031–T038)**: needs US1's T023/T025 (presets, base class). T037 gates T038.
- **US2 (T039–T044)**: needs US1's base class and T031 (`SimpleTrigger_Gbb`). Independent of US3's
  results.
- **US4 (T045–T050)**: needs US1's base class. T050 compares against `st_gbb` (T034).
- **Polish**: after the stories it reports on.

Story order: US1 → US3 → US2 → US4. US2 and US4 could run before US3 in code, but backtest runs
are sequential overall, so run order follows priority.

## Parallel Opportunities

Code tasks marked [P] touch different files and can be written in parallel. **Backtest runs
(T002, T015, T028, T034, T043, T044, T050) are never parallel.**

```text
# Phase 2: these tests can be written together
T003 test_fingerprint · T005 test_tag_match · T007 test_entry_condition_sink · T009 test_foldplan_nomodel · T011 test_run_folds_nomodel · T014 halves

# US1: these can be written together once T017 exists
T020 test_label_map · T021 label_map · T022 test_presets · T023 presets · T024 test_simple_trigger_strategy

# US3 / US2 / US4 test files are independent of each other
T032 · T035 · T039 · T041 · T045 · T047
```

## Implementation Strategy

**MVP = Phase 1 + Phase 2 + US1.** This gives a validated no-model family that reproduces every
recorded 021/022 arm, plus measured trigger causality. On its own it already answers "what does
trigger X earn through framework Y with no model".

Then US3, the live thread: the first clean measurement of the model's contribution. Then US2, the
framework audit. Then US4, the coupling capability.

Every STOP in the task list is a report-to-user point. A reproduction or parity mismatch is a
finding to localise and discuss, never something to tune away.
