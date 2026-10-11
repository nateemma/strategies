---
description: "Task list for 025 family coexistence and reversion late-half fix"
---

# Tasks: Family Coexistence and Reversion Late-Half Fix

**Input**: `specs/025-family-coexistence/` (spec.md, plan.md, research.md R1–R7, data-model.md,
contracts/interfaces.md C1–C5, quickstart.md)

**Tests**: Included. The spec requires liveness counters and an early-only selection that is
enforced by a test (C5). The acceptance tests are walk-forward reproductions by `trades_hash`.

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`, and run as ONE sequential chain.
- Comparisons run with `PYTHONHASHSEED=0`.
- Commit to master, staging named files only.
- Every result is labelled PROVISIONAL (gate 2).
- **Gotcha (024):** any edit to `research/outcomes/*triggers*.py` changes the causality code sha. Run
  `python -m research.outcomes.causality --write` after it, or every SimpleTrigger arm refuses.
- **Gotcha (024):** `run_folds` skips folds that already have records. Use fresh tags for every
  arm.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `specs/025-family-coexistence/baseline.md`. Record:
  - the commit sha;
  - `st_fam_combo`'s 15 per-fold `trades_hash` values and its per-family exit and trade counts
    (from `regime/arms/st_fam_combo_f*_F*.json`);
  - its contention table (copied from `evaluation/contention_st_fam_combo.md`);
  - the fingerprints of arm D (`NNNC_A_Raw_Open`), `SimpleTrigger_Gbb` and
    `SimpleTrigger_FamCombined` (`27617f430beca1cd`).

---

## Phase 2: Foundational (blocking)

**Purpose**: rule A and rule B mechanisms, counters, fingerprint and the stage runner. At neutral
settings nothing may change.

- [X] T002 [P] Write `utils/tests/test_family_coexistence.py` (contract C1). Use a stub parent whose
  `confirm_trade_entry` returns a settable bool, and monkeypatch `Trade.get_trades_proxy`. Cases:
  - (a) cap None → the parent's result is returned unchanged, and the proxy is never called;
  - (b) the parent returns False → False, and the proxy is never called (the volume check runs
    first);
  - (c) cap 2 with 2 open `reversion:` trades and a `reversion:gbb` entry → False, and one JSONL
    line `{"rule": "pair_cap", "family": "reversion"}` is written to `$FT_ENTRY_CONDITION_SINK/
    coexist_events.jsonl` (`tmp_path`);
  - (d) the same cap with 2 open `momentum:` trades → True, because the cap is per family;
  - (e) a `+ov` tag counts as its family;
  - (f) an untagged entry → the parent's result;
  - (g) a cap outside 1..9 is refused at class creation;
  - (h) no sink env var → no file written.
- [X] T003 Create `utils/FamilyCoexistence.py` with `FamilyCoexistenceMixin`, per contract C1:
  - strategy-agnostic (imports only `utils.FamilyExits.family_of` and freqtrade's `Trade`);
  - `FAMILY_PAIR_CAP = None`;
  - `__init_subclass__` validation.

  T002 must then pass.
- [X] T004 [P] Write `SimpleTrigger/tests/test_coexistence_recency.py` (contract C2). Use a
  synthetic frame where `gbb` fires on rows 3, 6 and 9 and `breakout_gbb` on row 2 (set columns to
  get these). With `TRIGGERS=("gbb","breakout_gbb")`:
  - (a) `MOMENTUM_RECENCY_BARS=0` → identical to 024 tags and mask;
  - (b) `=2` → row 3 dropped (momentum 1 bar ago), and rows 6 and 9 kept;
  - (c) `=5` → rows 3 and 6 dropped;
  - (d) a row where momentum fires itself is never dropped;
  - (e) causal: appending future rows never changes the decisions on earlier rows;
  - (f) `MOMENTUM_RECENCY_BARS>0` with no momentum trigger in `TRIGGERS` is refused at class
    creation;
  - (g) with the sink env set, `<pair>_coexist.json` holds `coexist_recency_refused.pass` = the
    number of dropped rows;
  - (h) (G2) with `MOMENTUM_RECENCY_BARS>0`, an overlap row is tagged `momentum:breakout_gbb+ov`,
    and with it at 0 the same row stays `reversion:gbb+ov` (024 behaviour);
  - (i) (G1) a grandchild config that sets `FAMILY_EXIT_OVERRIDES = {"reversion":
    {"unclog_12h_hours": 6}}` under a child that sets `{"reversion": {"take_profit": 0.03}}`
    resolves `FAMILY_EXITS["reversion"] == {"take_profit": 0.03, "unclog_12h_hours": 6}`. The
    momentum set is unchanged.
- [X] T005 Edit `SimpleTrigger/SimpleTriggerStrategy.py` (contract C2):
  - add `MOMENTUM_RECENCY_BARS = 0`;
  - the class MRO becomes `(FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin,
    BaseStrategy)`;
  - rule B goes in `_tags` / `get_entry_conditions`, using a causal rolling max over the past N
    rows of the momentum-family masks;
  - write the sink file in `populate_entry_trend` when the env var is set, using the pair from
    `metadata`;
  - validation in `_validate_triggers`;
  - (G2) under rule B, overlap rows are tagged momentum;
  - (G1) `_apply_preset` merges `FAMILY_EXIT_OVERRIDES` along the MRO (reversed `__mro__`, per
    family, per key).

  T004 must then pass, and so must every existing SimpleTrigger test.
- [X] T006 [P] Write `evaluation/tests/test_run_arm_coexistence.py`:
  - `aggregate_coexistence(tmp_sink)` sums JSONL lines into `{"pair_cap": {"reversion": 2,
    "momentum": 1}}`;
  - it returns `{}` for a missing or empty sink;
  - malformed lines are skipped.
- [X] T007 Edit `regime/run_arm.py` (contract C3): add `aggregate_coexistence(sink)` and record
  `coexistence_refusals`, omitted when empty, next to `entry_condition_counts`. `aggregate_sink`
  stays unchanged; rule B's per-pair file merges through it. T006 must then pass.
- [X] T008 Edit `evaluation/fingerprint.py` (contract C4):
  - `FAMILY_PAIR_CAP` (when not None) and `MOMENTUM_RECENCY_BARS` (when > 0) join the 024 family
    members;
  - both also trigger family mode, so an arm using only a coexistence rule still hashes its
    family members;
  - tests in `evaluation/tests/test_fingerprint.py`:
    - arm D and `SimpleTrigger_Gbb` stay `303456c1fa322b4c`;
    - `SimpleTrigger_FamCombined` stays `27617f430beca1cd`;
    - each of cap 3/5/7 and window 4/16/48 gives a distinct sha.
- [X] T009 [P] Write `evaluation/tests/test_stage.py` (contract C5), with synthetic arm records and
  pooled sets in `tmp_path`:
  - (a) the selection ignores f08–f14: a candidate that is best late and worst early is not
    chosen;
  - (b) a degenerate candidate (a family below 25% of its single-family trades in a half) is
    excluded;
  - (c) a tie on the early sum goes to fewer entries lost to the other family for
    `--stage coexistence`, and to the higher reversion-side early contribution for
    `--stage tuneN` (G4);
  - (d) exit 3 and `adopted: false` when the late verdict is not IMPROVED (stub `halves` result);
  - (e) exit 2 on a missing input;
  - (f) `decisions.md` gets one appended block per call, with the early table, the chosen arm,
    the late verdict and the liveness rows.
- [X] T010 Create `evaluation/stage.py` (contract C5, data model `StageDecision`):
  - it reuses `evaluation.pool`, `evaluation.contention` and `evaluation.halves` (import, don't
    copy);
  - it reads wallet sums from the arm records;
  - selection per research R4 and R5 (the primary key is the whole-arm f00–f07 sum; the
    tie-breaks are stage-specific, G4).

  T009 must then pass.
- [X] T011 Create the neutral configs:
  - `SimpleTrigger/SimpleTrigger_CoexCapOff.py` (`FAMILY_PAIR_CAP = None`, stated explicitly);
  - `SimpleTrigger/SimpleTrigger_CoexWinOff.py` (`MOMENTUM_RECENCY_BARS = 0`);
  - both subclass `SimpleTrigger_FamCombined`.

  Derive plans `st_coex_capoff` and `st_coex_winoff`.
- [X] T012 Run `st_coex_capoff`, then `st_coex_winoff`. SC-001: the per-fold `trades_hash` equals
  `st_fam_combo`'s in 15/15 folds for both. Record in `baseline.md`. **STOP on any divergence**:
  it means the MRO or rule plumbing changed behaviour.

**Checkpoint**: the mechanisms are in and neutral, and the stage runner is enforced by test.

---

## Phase 3: User Story 1 — coexistence (Priority: P1) 🎯 MVP

**Goal**: choose (or reject) a coexistence rule on early folds and judge it on late folds.

**Independent Test**: six active arms each show non-zero liveness, and the stage decision is
recorded.

- [ ] T013 [US1] Create six active configs subclassing `SimpleTrigger_FamCombined`:
  - `SimpleTrigger_CoexCap3/5/7` (`FAMILY_PAIR_CAP = 3/5/7`);
  - `SimpleTrigger_CoexWin4/16/48` (`MOMENTUM_RECENCY_BARS = 4/16/48`).

  Derive plans `st_coex_cap3/5/7` and `st_coex_win4/16/48`.
- [ ] T014 [US1] Run the six arms as one sequential chain. Check liveness:
  - each cap arm has non-zero `coexistence_refusals.pair_cap`;
  - each window arm has non-zero `entry_condition_counts.coexist_recency_refused.pass`;
  - otherwise mark the arm INERT in `baseline.md`.
  - **(G5) Exception:** zero `pair_cap` refusals on `st_coex_cap3` cannot be a true null. It binds whenever a
    family tries a 4th concurrent pair. With 5,000–9,000 trades per family over 10 pairs, that is
    expected to happen often (024 did not measure peak concurrency, so this is an expectation,
    not a measurement). So it means the hook is not reached in backtests (no `entry_tag`, or
    `Trade.get_trades_proxy` not seeing open backtest trades). **STOP and diagnose before T016**;
    do not record INERT.
- [ ] T015 [US1] For each of the six arms, pool it and run `evaluation.contention` against
  `st_gbb` / `st_bgo_mx`. Write `evaluation/contention_st_coex_*.md`.
- [ ] T016 [US1] Run `python -m evaluation.stage --stage coexistence --previous st_fam_combo
  --candidates st_coex_cap3,st_coex_cap5,st_coex_cap7,st_coex_win4,st_coex_win16,st_coex_win48
  --singles reversion=st_gbb,momentum=st_bgo_mx --out specs/025-family-coexistence/decisions.md`.
  Record the adopted arm, or "none adopted, US2 runs on `st_fam_combo`", as `BASE_ARM` in
  `baseline.md`.

**Checkpoint**: `BASE_ARM` is fixed for US2.

---

## Phase 4: User Story 2 — reversion exit tuning inside the combined arm (Priority: P2)

**Goal**: four single-parameter steps on the reversion set, each choosing on early folds and
judging on late folds.

**Independent Test**: each step's decision block in `decisions.md`, with liveness.

Every step's candidate configs subclass the PREVIOUS step's adopted arm (step 1: `BASE_ARM`'s
class). They set ONLY their new parameter: `FAMILY_EXIT_OVERRIDES = {"reversion": {<param>:
<value>}}`. Earlier adopted values are inherited through the MRO merge (G1).

Before running each step, assert that the candidate's resolved `FAMILY_EXITS["reversion"]` contains
every previously adopted key, and record it in `baseline.md`.

- [ ] T017 [US2] Step 1, `take_profit`:
  - configs `SimpleTrigger_Tune1Tp015/Tp030/TpOff` (0.015 / 0.030 / 10.0);
  - plans `st_tune1_tp015/tp030/tpoff`;
  - run the chain, then `evaluation.stage --stage tune1 --previous <BASE_ARM>`;
  - liveness = the change in the reversion `take_profit` exit count.
- [ ] T018 [US2] Step 2, `unclog_12h_hours`:
  - configs `SimpleTrigger_Tune2U6/U24` (6 / 24) on top of step 1's adopted arm (or `BASE_ARM`);
  - plans `st_tune2_u6/u24`;
  - run, then `stage --stage tune2`;
  - liveness = the change in the reversion `unclog_12h` and `unclog_1d` counts.
- [ ] T019 [US2] Step 3, `atr_stoploss_floor`:
  - configs `SimpleTrigger_Tune3F05/F08` (−0.05 / −0.08) on top of step 2's adopted arm;
  - plans `st_tune3_f05/f08`;
  - run, then `stage --stage tune3`;
  - liveness = a change in reversion stop-exit counts or in `trades_hash` (an INERT arm is
    reported, research R5).
- [ ] T020 [US2] Step 4, `use_volume_confirmation_stoploss`:
  - config `SimpleTrigger_Tune4VcOff` (False) on top of step 3's adopted arm;
  - plan `st_tune4_vcoff`;
  - run, then `stage --stage tune4`.

  The arm adopted after step 4, or the last adopted arm, is `FINAL_ARM`. Record it in
  `baseline.md`.

**Checkpoint**: `FINAL_ARM` is fixed.

---

## Phase 5: User Story 3 — final report (Priority: P3)

- [ ] T021 [US3] Pool `FINAL_ARM` and compare it (`compare_pooled` + `halves`, `PYTHONHASHSEED=0`)
  with `st_bgo_mx`, `st_gbb` and `st_fam_combo`. Add:
  - the sum-of-singles per-fold reference;
  - the per-fold correlation between families;
  - per-family contributions by half (from `enter_tag`);
  - contention against 024.
- [ ] T022 [US3] Write `specs/025-family-coexistence/report.md`:
  - SUMMARY;
  - SC-001…SC-005 table;
  - the stage-by-stage decisions (linking `decisions.md`);
  - an explicit answer to the SC-004 questions with paired intervals (momentum late vs +2.50;
    entries lost to the other family vs 3,258 / 1,252; the final arm vs momentum alone per half);
  - every figure labelled PROVISIONAL.

  A negative outcome is reported as a result.

---

## Phase 6: Polish

- [ ] T023 Run the full test gate: `pytest -q Framework/tests evaluation/tests SimpleTrigger/tests
  research/outcomes/tests utils/tests NNNC/tests`. Only the 3 known pre-existing training-pipeline
  failures may remain. Then ruff on the lines this feature added (024's added-lines method). Run
  the exit single-source test and classify the new files there if it asks (CONFIGURING).
- [ ] T024 Check that the 023/024 fingerprints are unchanged (T001 values, including
  `SimpleTrigger_Gbb` = `303456c1fa322b4c`). Re-run via `run_arm` under fresh `t024_*` tags (G3):
  - `st_fam_combo` f00 (024);
  - `st_gbb` f00, the 023 twin, `aa058ba815f48627`;
  - `stl_bo` f00, the 023 legacy preset, `61bfaa185be5164a`.

  Each `trades_hash` must match. Record in `baseline.md`. This covers the T005 MRO change for every
  SimpleTrigger arm, not only the combined one.
- [ ] T025 Update memory:
  - `project_025_family_coexistence.md` (results);
  - the `MEMORY.md` "Current state" hook;
  - an `index_*` hub line if a reusable finding results.

  Mark every task `[X]`. Commit code, configs, plans, arm records and docs, staging named files
  only.

---

## Dependencies & Execution Order

- **Phase 2 gates everything.** T012 (neutral reproduction) must pass before any active arm runs.
- **US1 → US2 → US3, strictly sequential.** Each US2 step's configs depend on the previous step's
  adopted arm, so steps cannot be pre-generated.
- **Tests come before their implementation:** T002→T003, T004→T005, T006→T007, T009→T010; T008
  includes its own tests.
- **Backtests:** T012, T014, T017–T020 are each one sequential chain, never parallel.

### Parallel opportunities

- T002 ∥ T004 ∥ T006 ∥ T009 (separate test files).
- T003 ∥ T007 ∥ T010 after their tests (separate modules).
- T005 touches SimpleTriggerStrategy and must not run while any backtest is running (backtests
  import it per fold).

## Implementation Strategy

**MVP = Phases 1–3:** the mechanisms plus the coexistence decision, about 45 minutes of backtests.
US2 adds up to 8 arms (about 45 minutes). US3 is analysis only.
