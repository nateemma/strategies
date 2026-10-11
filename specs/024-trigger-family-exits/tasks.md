---
description: "Task list for 024 trigger-family exit presets"
---

# Tasks: Trigger-Family Exit Presets

**Input**: Design documents from `specs/024-trigger-family-exits/`. These are plan.md,
spec.md, research.md (R1–R10), data-model.md, contracts/interfaces.md (C1–C8) and
quickstart.md.

**Tests**: Included. The spec requires test enforcement (SC-006 and the C1 neutrality tests),
and the acceptance tests are walk-forward reproductions by `trades_hash`.

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`, and run as ONE sequential chain.
- Commit to master, staging named files only.
- Every result is labelled PROVISIONAL (gate 2).

**User Story 3 (tuning) is DEFERRED** (Q1 → A), so it has no tasks here.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 Record the pre-change baseline in `specs/024-trigger-family-exits/baseline.md`:
  - the recorded `trades_hash` of `st_gbb` f00, `regime/arms/wf_aro_f00_F00.json`, `st_bbo` f00,
    `stl_bo` f00, and all 15 folds of `st_gbb` and `st_bgo_mx`;
  - the `exit_reason_counts` totals of `st_bgo_mx`;
  - the output of `python -m evaluation.dump_strategy --strategy MomentumRegimeBasket15mFast
    --config config/config_mom_dryrun_15m.json`, saved to the scratchpad;
  - the commit sha.

---

## Phase 2: Foundational (blocking; NO behaviour change)

**Purpose**: the C1 hook, R2 routing and C5 families. Every recorded arm must stay hash-identical.

- [ ] T002 [P] Write `Framework/tests/test_exit_params_trade_hook.py`:
  - (a) `_exit_params(trade)` equals `_exit_params()` for a stub trade, on `SimpleTrigger_Gbb`,
    `NNNC_A_Raw_Open` and `SimpleStrategies/TriggerOnly`;
  - (b) with params mirroring attributes, the stop path gives the same `atr_stop_fraction` inputs
    as the attribute reads it replaces (R2). Build a stub dataprovider frame with `atr_pct_roll`
    and `volume`, then call `_resolve_normal_stop(pair, trade)` and the after-fill branch of
    `custom_stoploss`.
  - It should FAIL until T003.
- [ ] T003 Edit `Framework/BaseStrategy.py` (contract C1):
  - give `_exit_params` an optional `trade: Trade | None = None`, with the body unchanged;
  - pass `trade` from `custom_exit` and from the after-fill branch of `custom_stoploss`;
  - change `_resolve_normal_stop(pair)` to `_resolve_normal_stop(pair, trade=None)`, and update
    its caller in the grace branch;
  - `_atr_bar_inputs(pair, params)` reads `params.use_volume_confirmation_stoploss`;
  - `_resolve_normal_stop` reads `params.use_atr_adaptive_stoploss` and `params.static_stoploss`;
  - the after-fill branch reads `params.use_atr_adaptive_stoploss`.
  - Add no family, tag or trigger logic (FR-014).
  - Keep the diff surgical: no reformatting of untouched lines.
  - T002 must then pass.
- [ ] T004 Edit `utils/StopInvariants.py` (contract C3):
  - `_check_stop` reads `use_atr_adaptive_stoploss`, `static_stoploss` and `atr_stoploss_floor`
    from `self._exit_params(trade)` when the method exists, and falls back to the current
    attribute reads otherwise;
  - extend `utils/tests/test_stop_invariants.py`, using a strategy whose
    params mirror its attributes, to show the bound computed is unchanged.
- [ ] T005 Neutrality gate (research R9, quickstart 0):
  - run `run_arm` on `st_gbb` f00, `wf_aro` f00, `st_bbo` f00 and `stl_bo` f00 with the plans'
    exact fold settings;
  - each `trades_hash` MUST equal T001's value;
  - re-run the dry-run dump and diff it against T001;
  - record the results in `baseline.md`.
  - **STOP the feature if any hash differs.**
- [ ] T006 [P] Write `research/outcomes/tests/test_trigger_families.py`:
  - (a) every trigger registered by `ensure_builtins()` has `family` in
    {reversion, momentum, unassigned} and a non-empty `family_reason`;
  - (b) the classification equals spec FR-001's table exactly;
  - (c) registering a trigger through a builtin register function without a family fails the
    check (a planted test trigger in the test).
- [ ] T007 Edit `research/outcomes/triggers.py` (contract C5):
  - add `family: str | None = None` and `family_reason: str = ""` to `Trigger`, and validate
    `family` in `__post_init__` when it is set;
  - `--list` prints the family column.
- [ ] T008 Edit `research/outcomes/builtin_triggers.py` and `research/outcomes/audit_triggers.py`:
  - pass `family=` and `family_reason=` for every registered trigger, per spec FR-001;
  - `empty` and `random_matched` → `unassigned`; `gbb_stale` → `reversion`, as in FR-001;
  - unscoreable triggers (`trends`, `bands`) → `unassigned`.
  - T006 must then pass.
  - Re-run `python -m research.outcomes.causality --write`: the code sha changes, and every
    verdict must be unchanged (diff it against the committed `trigger_causality.json`).

**Checkpoint**: the framework hook is in, provably neutral, and every trigger has a family.

---

## Phase 3: User Story 1 — a trigger runs its family's exits (Priority: P1) 🎯 MVP

**Goal**: an arm declared as trigger + `PRESET="ARM_D_FAMILY"` gets its family's exits.

**Independent Test**: `st_fam_gbb` equals `st_gbb`, and `st_fam_bgo` equals `st_bgo_mx`,
trade-for-trade in 15/15 folds.

- [ ] T009 [P] [US1] Write `utils/tests/test_family_exits.py` (contract C2):
  - `family_of("momentum:breakout_gbb") == "momentum"`;
  - `family_of("model_entry ")` and `family_of(None)` return None;
  - `_exit_params(trade)` applies `FAMILY_EXITS[family]` via `dataclasses.replace`;
  - an unknown or missing tag returns the base params and increments `untagged_exits`;
  - a forbidden key (`stoploss_grace_hours`, ROI) or a non-`ExitParams` key in a family set
    raises at class creation;
  - `ExitParams.__post_init__` invariants are enforced after replacement (floor ≤ cap < 0);
  - (F3) a class setting `take_profit` in a family set AND `cexit_take_profit` in
    `SELL_OVERRIDES` is refused;
  - (F5) `unclog_12h_hours` above `max_hold_days*24` is refused; above `unclog_1d_hours` it
    warns; 24/24 is accepted;
  - `family_of("reversion:gbb+ov") == "reversion"`.
- [ ] T010 [US1] Create `utils/FamilyExits.py` with `FamilyExitMixin`, per contract C2 and the
  data model's allowed and forbidden keys:
  - strategy-agnostic, with no SimpleTrigger or registry import;
  - `untagged_exits` counter as an instance attribute.
  - T009 must then pass.
- [ ] T011 [P] [US1] Extend `SimpleTrigger/tests/test_presets.py`:
  - `ARM_D_FAMILY` has the same attributes, buy and sell as `ARM_D_OPEN`, plus the
    `FAMILY_EXITS` = {reversion: {}, momentum: {take_profit: 10.0, unclog_12h_hours: 24.0},
    unassigned: {}} from data-model.md;
  - `ARM_D_OPEN` and `LEGACY_TRIGGERONLY` resolve every family to `{}`.
- [ ] T012 [US1] Edit `SimpleTrigger/presets.py`:
  - add the `ARM_D_FAMILY` preset (it reuses `ARM_D_OPEN`'s dicts by reference/merge, not by
    copy-paste) with its `family_exits` block;
  - `validate()` checks the family-set keys.
  - T011 must then pass.
- [ ] T013 [US1] Edit `SimpleTrigger/SimpleTriggerStrategy.py` (contract C4, single-trigger path):
  - the MRO becomes `(FamilyExitMixin, StopInvariantMixin, BaseStrategy)`;
  - `_apply_preset` resolves `FAMILY_EXITS`, with a subclass override merged per family;
  - after `super().populate_entry_trend()`, `populate_entry_trend` sets `enter_tag` to
    `f"{family}:{trigger}"` on signal rows, using the registry family;
  - update the class docstring, which still says `enter_tag` stays "model_entry ".
  - Every existing SimpleTrigger test must still pass, including `test_legacy_reproduction.py`.
- [ ] T014 [P] [US1] Write a unit test in `SimpleTrigger/tests/test_family_tagging.py`:
  - on a synthetic populated frame, `SimpleTrigger_FamGbb`'s signal rows carry `reversion:gbb`,
    and non-signal rows carry "";
  - an `ARM_D_OPEN` config's `_exit_params(trade)` with a `momentum:` tag equals its base params
    (shared mode);
  - (F8, FR-004) on an `ARM_D_FAMILY` config, a trade tagged `momentum:breakout_gbb` gets the
    momentum params even when the strategy's latest frame fires only `gbb`: exits read the stored
    tag, never the current bar.
- [ ] T015 [US1] Edit `evaluation/fingerprint.py` (contract C7):
  - extend `OPTIONAL_ATTRIBUTE_MEMBERS` handling with `TRIGGERS`, `DISABLED_TRIGGERS` and
    `FAMILY_PRIORITY`;
  - add `FAMILY_EXITS_RESOLVED` (the full `ExitParams` dict per family, computed from the
    class's default params plus overrides, without instantiating);
  - add `TRIGGER_FAMILIES` (the registry family of each of the arm's triggers);
  - (F1) all of these only in family mode (some family set non-empty), multi-trigger mode
    (more than one trigger) or disabled-side mode (non-empty `DISABLED_TRIGGERS`). A
    single-trigger, shared-mode class hashes exactly as in 023.
  - Tests in `evaluation/tests/test_fingerprint.py`:
    - arm D is still `303456c1fa322b4c`;
    - `SimpleTrigger_Gbb` is still `303456c1fa322b4c`, and that check is re-run after T020 adds
      the `TRIGGERS` mapping;
    - (F6) each of these changes the sha: one momentum value, a `FAMILY_PRIORITY` flip, and a
      monkeypatched trigger family in `TRIGGER_FAMILIES`.
- [ ] T016 [US1] Create `SimpleTrigger/SimpleTrigger_FamGbb.py` (`TRIGGER_NAME="gbb"`,
  `PRESET="ARM_D_FAMILY"`) and `SimpleTrigger/SimpleTrigger_FamBgo.py`
  (`TRIGGER_NAME="breakout_gbb"`, `PRESET="ARM_D_FAMILY"`). Derive plans `st_fam_gbb` and
  `st_fam_bgo` with `foldplan derive --from evaluation/folds/plan_NNNC_A_Raw_Open.json`.
- [ ] T017 [US1] Edit `regime/run_arm.py` (contract C8, F2). From the backtest archive's trades
  (the same source `trades_hash` uses), compute:
  - `untagged_trades`: `enter_tag` without a recognised family prefix;
  - `family_trade_counts`;
  - `overlap_entries`: the `+ov` suffix;
  - per-family `exit_reason_counts`.

  All are omitted when no trade carries a family tag. Add a unit test on a synthetic trades frame
  in `evaluation/tests/test_run_arm_tags.py`.
- [ ] T018 [US1] Run `st_fam_gbb`, then `st_fam_bgo` (sequentially). SC-001:
  - their per-fold `trades_hash` equals `st_gbb` and `st_bgo_mx` in 15/15 folds;
  - `untagged_trades` = 0 (SC-007);
  - `st_fam_bgo`'s `unclog_12h` / `unclog_1d` totals equal `st_bgo_mx`'s (liveness, research R5).
  - Record in `baseline.md`.
  - **If `st_fam_bgo` diverges, STOP and diagnose before US2.**

**Checkpoint**: the MVP. Family exits work and reproduce 023 exactly.

---

## Phase 4: User Story 2 — the combined reversion + momentum arm (Priority: P2)

**Goal**: `gbb` and `breakout_gbb` run in one arm, with reversion priority on overlap and
contention reported.

**Independent Test**: each side with the other disabled reproduces its single-family arm. The
full arm is reported against both.

- [ ] T019 [P] [US2] Write `SimpleTrigger/tests/test_combined_entry.py`:
  - on a synthetic frame where `gbb` and `breakout_gbb` both fire on some rows, the entry mask is
    the OR of the two;
  - overlap rows are tagged `reversion:gbb+ov` under the default priority and
    `momentum:breakout_gbb+ov` under the reversed priority, and non-overlap rows carry no `+ov`;
  - `DISABLED_TRIGGERS=("breakout_gbb",)` gives exactly the `gbb` mask and tags;
  - setting both `TRIGGER_NAME` and `TRIGGERS`, disabling every trigger, or naming an
    unregistered or non-runnable trigger each raises or refuses with a `*** SimpleTrigger:`
    message.
- [ ] T020 [US2] Edit `SimpleTrigger/SimpleTriggerStrategy.py` (contract C4, multi-trigger path):
  - `TRIGGERS`, `DISABLED_TRIGGERS` and `FAMILY_PRIORITY`;
  - `get_entry_conditions` runs the `check_runnable` gate per enabled trigger, evaluates each mask
    once, ORs them, and stores the per-row winning tag (by family priority, then trigger order),
    with `+ov` when another family's trigger also fired (F4);
  - `TRIGGER_NAME` maps to `TRIGGERS=(name,)`.
  - T019 must then pass, and so must every earlier SimpleTrigger test.
- [ ] T021 [US2] Create the combined configs:
  - `SimpleTrigger/SimpleTrigger_FamCombined.py` (`TRIGGERS=("gbb","breakout_gbb")`,
    `PRESET="ARM_D_FAMILY"`);
  - `SimpleTrigger_FamCombinedRevOnly.py` (`DISABLED_TRIGGERS=("breakout_gbb",)`);
  - `SimpleTrigger_FamCombinedMomOnly.py` (`DISABLED_TRIGGERS=("gbb",)`).
  - Derive plans `st_fam_combo`, `st_fam_combo_rev` and `st_fam_combo_mom`.
  - The fingerprint test confirms that the three hash differently.
- [ ] T022 [US2] Run `st_fam_combo_rev`, then `st_fam_combo_mom`. SC-002: they reproduce `st_gbb`
  and `st_bgo_mx` in 15/15 folds by `trades_hash`. Record in `baseline.md`. **STOP on divergence.**
- [ ] T023 [P] [US2] Write `evaluation/tests/test_contention.py` with synthetic pooled sets:
  - a lost entry whose pair is held by the other family's open trade → `pair_held_by_other_family`;
  - 10 concurrent open trades → `slots_full`;
  - a same-family hold → `pair_held_same_family`;
  - otherwise → `divergence`;
  - gained entries are counted;
  - the halves split is f00–f07 / f08–f14;
  - exit 2 on a missing input.
- [ ] T024 [US2] Create `evaluation/contention.py` (contract C6, data model `ContentionRecord`):
  - the family is read from each trade's `enter_tag` in the combined set, and `+ov` entries are
    counted per family and half (FR-010);
  - `max_open_trades` comes from `--config`;
  - output is JSON plus a markdown summary.
  - T023 must then pass. If pooled trades lack `enter_tag`, extend `evaluation/pool.py` to carry
    it from the archive (an additive field).
- [ ] T025 [US2] Run `st_fam_combo` (15 folds), then pool `st_fam_combo`, `st_gbb` and
  `st_bgo_mx`, and run `evaluation.contention`. Write `evaluation/contention_st_fam_combo.{json,md}`.
- [ ] T026 [US2] Compare and report (SC-005, FR-011):
  - `PYTHONHASHSEED=0` `compare_pooled` plus `halves` of `st_fam_combo` against `st_gbb` and
    against `st_bgo_mx`;
  - the sum-of-singles per-fold reference;
  - the per-fold correlation between `st_gbb` and `st_bgo_mx`;
  - per-family P&L inside the combined arm, by half, from `enter_tag`;
  - overlap entries resolved by priority (`+ov`) and their P&L, by half.
  - Write `specs/024-trigger-family-exits/report.md`, every figure labelled PROVISIONAL.

**Checkpoint**: the combined arm is measured, and contention is quantified. That scopes the
deferred tuning feature.

---

## Phase 5: Polish & cross-cutting

- [ ] T027 Run the full unit gate:
  `../../.venv/bin/python -m pytest -q Framework/tests evaluation/tests SimpleTrigger/tests
  research/outcomes/tests utils/tests NNNC/tests`.
  The 3 known pre-existing training-pipeline failures are acceptable only if they are unchanged
  from 023. Also run `ruff check` and `ruff format --check` on the touched files.
- [ ] T028 Run the existing exit single-source test
  (`Framework/tests/test_exit_policy_single_source.py`): `utils/FamilyExits.py` must not
  duplicate exit arithmetic. If the allowlist needs it, classify the new files there, as 023 did.
- [ ] T029 Finish `report.md`:
  - the SUMMARY;
  - the SC-001…SC-007 table with evidence paths;
  - the R9 correction (the dry-run never imported `BaseStrategy`);
  - the recommended scope for the deferred tuning feature, from the contention results.
  - Mark every task `[X]` in this file.
- [ ] T030 Update memory:
  - `project_024_trigger_family_exits.md` (results and status), with its hook in `MEMORY.md`
    "Current state";
  - an `index_exits.md` hub line if a new exit finding results.
  - Commit the code, configs, plans, arm records and docs, staging named files only.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2:** T005 is a hard gate. No family work starts until neutrality is proven.
- **Phase 2 → US1:** US1 needs T003 (hook), T007/T008 (families) and T004 (invariants).
- **US1 → US2:** US2 needs T010, T012, T013 and T015, and T018 must pass (exact `st_bgo_mx`
  reproduction).
- **Within phases**, tests are written before their implementation (T002→T003, T006→T007/T008,
  T009→T010, T011→T012, T019→T020, T023→T024).
- **Backtests** (T005, T018, T022, T025) run strictly sequentially, one chain at a time.

### Parallel opportunities

- T002 ∥ T006, which are different files.
- T009 ∥ T011 ∥ T014, which are test files.
- T019 ∥ T023.
- No two backtest tasks ever run in parallel.

## Implementation Strategy

**MVP = Phases 1–3:** family exits proven to reproduce 023, about 10 minutes of backtests.

Then US2 is about 30 minutes of backtests (2 reproduction arms plus 1 combined arm), then polish.
The deferred tuning feature is specified only after `report.md`'s contention results are in.
