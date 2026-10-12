---
description: "Task list for 028 recovery-aware reversion exits"
---

# Tasks: Recovery-Aware Reversion Exits

**Input**: `specs/028-recovery-aware-exits/` (spec.md, plan.md, research.md R1–R9, data-model.md,
contracts/interfaces.md I1–I3, quickstart.md)

**Tests**: Included, because the spec requires the following, each of which needs a test:
- the pre-registration precedes every 2022 record (SC-001);
- the decision rule follows the pre-declared form (SC-002);
- switch-off reproduction and switch-on equivalence (SC-004, FR-010/011);
- stop invariants (FR-012).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`, run as ONE sequential chain, and use
  fresh tags (`run_folds` skips existing records).
- Commit to master, staging named files only.
- **NO 2022 backtest or replay before T009 is committed.** That includes dry probes that run
  `run_arm`. `--dry-run` of `run_folds` is allowed, because it runs nothing.
- **US1 is a STOP gate.** If T012 exits 3 (no candidate passes), mark T013–T022 `[-]` with the
  reason and go to Phase 6.
- **US3 builds only what US2 ADOPTED.** A sub-item for a candidate that was not adopted is marked
  `[-]` in place. If nothing is adopted, mark all of Phase 5 `[-]`.
- Labels:
  - 2023–26 results are EXPLORATORY;
  - 2022 results are CONFIRMATORY and PROVISIONAL under gate 2 (R8).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (2023–26 only; nothing touches 2022)

- [ ] T001 Create `specs/028-recovery-aware-exits/baseline.md`. Record:
  - the commit sha;
  - framework fingerprints for `NNNC_MLX_Open` (arm D), `SimpleTrigger_BreakoutGbb_MomExit`
    (`st_bgo_mx`), `SimpleTrigger_RevOnlyVolreg2` (`st_book_rev2`) and the three XC classes, via
    `evaluation/fingerprint.py`;
  - the per-fold `trades_hash` (15) of `st_book_rev2`, `st_xpl_rev_c4`, `st_xpl_rev_c2` and
    `st_xpl_rev_c1`, read from `regime/arms/`;
  - the R2 audit: count the `regime/arms/*.json` records whose `timerange` starts with `2022`, and
    the `backtest_results/*.meta.json` SimpleTrigger runs whose span overlaps 2022-03-07…2023-01-01.
    Both must be 0; otherwise stop and report (spec edge case "fresh window already seen").
- [ ] T002 [P] Create `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC42.py` per R5:
  - subclass `SimpleTrigger_RevOnlyVolreg2_XC2`;
  - `SELL_OVERRIDES = {**SimpleTrigger_RevOnlyVolreg2_XC2.SELL_OVERRIDES, "stoploss_grace_hours": 12.0}`;
  - a docstring marking it EXPLORATORY.

  Check that its resolved fingerprint equals XC4's (the fingerprint can't see XC2's
  `_exit_params` override).
- [ ] T003 Create `evaluation/folds/plan_st_xpl_rev_c42.json` by copying
  `plan_st_xpl_rev_c4.json`. Set `base_strategy`, `run_id`, every fold's `strategy_class`
  (`SimpleTrigger_RevOnlyVolreg2_XC42`) and `arm_tag` (`st_xpl_rev_c42_fNN`), and `status`
  "planned". Validate with `foldplan.validate_plan` and run `run_folds --dry-run`.
- [ ] T004 Run C4+C2 on f00–f14:
  `python -m evaluation.run_folds --plan user_data/strategies/evaluation/folds/plan_st_xpl_rev_c42.json`
  (repo root, about 3 min). Check:
  - 15/15 records, returncode 0;
  - trades differ from `st_book_rev2` in at least 1 fold.

  Write `specs/028-recovery-aware-exits/baseline_c42.md` in the format of 027 `exploratory.md`:
  - early/late sums;
  - gain vs `st_book_rev2`;
  - paired verdicts and late risk verdict;
  - liveness.

  Label it EXPLORATORY.

---

## Phase 2: Foundational (blocks every story)

- [ ] T005 [P] Write `research/exit_coupling/tests/test_confirm.py`. The tests must FAIL first.
  - `recovery_share` on a hand-built frame (stop exits with and without a later touch; non-stop
    exits ignored).
  - `candidate_passes`:
    - floor applied vs not applied (C2);
    - gain ≤ 0 fails;
    - share exactly at the floor passes.
  - `decide`: every rule letter (a)–(d) failing alone gives NOT ADOPTED with that letter in
    `failed`. Also:
    - control trades < 200 → INCONCLUSIVE;
    - folds_changed 0 → INCONCLUSIVE;
    - all pass → ADOPT;
    - paired UNCHANGED with a positive gain → ADOPT (the rule is not a significance test);
    - a risk WORSENED in either Sharpe or maxDD → NOT ADOPTED.
  - `map_paired` / `map_risk`: every verdict value in the R4 mapping table. In particular:
    - INCONCLUSIVE_ARM_LOSES and INCONCLUSIVE_UNDERPOWERED pass (b) with a flag;
    - INCONCLUSIVE_ABSORBED and UNRESOLVABLE give INCONCLUSIVE;
    - an unknown value raises.
  - Rule (a) uses the per-fold sum, not `delta_pp`: a case where they differ in sign follows (a).
  - `ci_half_width_pp([-36.5, 73.0], 300/365)` = 45.0.
  - The re-run check helper: an equal `trades_hash` keeps the verdict; a different one turns
    ADOPT into INCONCLUSIVE.
  - The refusal helpers (R6):
    - a class constant different from the pre-registration;
    - a record mtime older than the pre-registration commit time;
    - two arms sharing a class.

  Each must give exit 2. Use temporary files and inject the commit time; no git calls in the
  unit tests.
- [ ] T006 Implement `research/exit_coupling/confirm.py` per contract I1, with pure functions
  `recovery_share`, `candidate_passes`, `decide`, plus the refusal helpers. Reuse:
  - `feasibility.trade_rows` and `feasibility.fidelity` (fold ids are free strings;
    `half_of` labels e-folds "late", which confirm.py ignores);
  - `paths.replay` with fixed `Rules` per candidate (C4 `grace_hours=12.0`; C2
    `take_profit_atr_mult=k`; C1 `stop_min_depth=d`; C42 both C4 and C2);
  - `compare_pooled.compare_data(halves.subset(arm, E), halves.subset(ctl, E))` with
    E = {e00, e01, e02}, called directly, for the paired verdict;
  - `risk.compare` on the same subsets for the risk verdict;
  - the R4 mapping via `map_paired` / `map_risk`.

  Add `judge --rerun-check`, which compares an `_rr` record's `trades_hash` with the original (I1).

  Values are read from `preregistration.json`, never hard-coded. T005 must pass.
- [ ] T007 [P] Create the five 2022 plans `evaluation/folds/plan_st_e22_{rev2,c4,c2,c1,c42}.json`
  per I3 and R1:
  - `no_model: true`;
  - folds e00 2022-03-07→2022-06-15, e01 →2022-09-23, e02 →2023-01-01;
  - `test_timerange` `YYYYMMDD-YYYYMMDD`;
  - `train_*` set to 2021-01-01 → the day before test_start, for form only;
  - `arm_tag` `st_e22_<id>_eNN`;
  - classes: the control `SimpleTrigger_RevOnlyVolreg2`, and XC4/XC2/XC1/XC42.

  Validate each with `foldplan.validate_plan`, run `run_folds --dry-run` on each, and check that
  no two plans share a class.
- [ ] T008 Write a warm-up probe and record its output in `baseline.md`. It runs no backtest; it
  reads only the bars before 2022-03-07, from the feathers.
  - For each of the 10 pairs, check that 15m data exists from at least 2022-03-07 minus 4096 bars
    (2022-01-23 18:00 UTC).
  - Check that the volreg instrument (`utils/RegimeState.volreg`, called as
    `SimpleTrigger_RegGateVolreg` calls it), computed on data truncated at 2022-03-07, is non-NaN
    at the last bar for every pair.

  If either fails, move e00's start to the first date where both pass and record the change
  BEFORE T009.
- [ ] T009 Write and commit the pre-registration: `specs/028-recovery-aware-exits/preregistration.json`
  (data-model fields) and `preregistration.md` (readable copy).
  - The candidates' frozen values come from the class constants:
    - XC2 `TP_ATR_MULT` 2.8970711921574543;
    - XC1 `STOP_MIN_DEPTH` −0.03710764139923586;
    - grace 12.0.
  - `early_gain_pp` / `late_gain_pp` come from 027 `exploratory.md` (C4/C2/C1) and T004 (C42).
  - The other fixed values:
    - folds from T007/T008;
    - floor 0.50;
    - fidelity 0.90;
    - min control trades 200;
    - rule (a)–(d) written out;
    - the R4 verdict-value mapping table;
    - the gate-5 re-run rule (an ADOPTED gain under 1 pp needs an identical `trades_hash` on a
      fresh-tag re-run).

  Commit it in ONE commit with the five plan files from T007. Record the commit sha in
  `preregistration.md` in a follow-up commit, which changes only that line.

**Checkpoint**: pre-registration committed. 2022 may now be touched.

---

## Phase 3: User Story 1 — Mechanism check on 2022 (P1) 🎯 MVP

**Goal**: The recovery share, forgone P&L and each candidate's replay gain on 2022, behind a
fidelity gate.

**Independent Test**: `feasibility.md`/`.json` with fidelity, the recovery share and a PASS/FAIL
per candidate; exit code 0, 2 or 3.

- [ ] T010 [US1] Run the control on 2022:
  `python -m evaluation.run_folds --plan user_data/strategies/evaluation/folds/plan_st_e22_rev2.json`.
  Check 3/3 records with returncode 0 and trade counts per fold. Record the total control trades.
- [ ] T011 [US1] Check that every 2022 record's mtime is newer than the pre-registration commit
  time; this is `confirm.py`'s refusal (ii) run once by hand.
- [ ] T012 [US1] Run
  `python -m research.exit_coupling.confirm feasibility --prereg specs/028-recovery-aware-exits/preregistration.json --out specs/028-recovery-aware-exits/feasibility.md`.
  Act on the exit code:
  - Exit 2 on fidelity: diagnose the replay against 2022 (a new mechanic?), fix it in
    `paths.py` with a test, and re-run. This is allowed because the replay is the instrument, not a
    candidate. Record it as a deviation.
  - Exit 3: STOP (see Conventions).

**Checkpoint**: at least one candidate PASSES, or the feature stops.

---

## Phase 4: User Story 2 — Judge each passing candidate on 2022 (P2)

**Goal**: ADOPT / NOT ADOPTED / INCONCLUSIVE per candidate under the pre-registered rule.

**Independent Test**: `verdicts.md`/`.json` with every field in the data model.

- [ ] T013 [US2] Run the 2022 plan of each PASSING candidate, one at a time, in order C4, C2,
  C1, C42 (skip those that failed T012, marking them STOPPED_AT_FEASIBILITY). Check 3/3 records
  per arm with returncode 0.
- [ ] T014 [US2] Run
  `python -m research.exit_coupling.confirm judge --prereg ... --feasibility specs/028-recovery-aware-exits/feasibility.json --out specs/028-recovery-aware-exits/verdicts.md`.
  Check that:
  - each candidate has exactly one verdict;
  - each cites the rule letters that failed;
  - the CI half-width is reported (SC-002).
- [ ] T014a [US2] Gate-5 re-run (constitution quality gate 5), for each ADOPTED candidate whose
  gain is under 1 pp:
  - create `evaluation/folds/plan_st_e22_<id>_rr.json`, holding only the fold with the largest
    absolute per-fold gain, with `arm_tag` `st_e22_<id>_rr_eNN`;
  - run it;
  - run `confirm judge --rerun-check`.

  A hash mismatch turns the verdict to INCONCLUSIVE. If there is no such candidate, mark this
  task `[-]`.
- [ ] T015 [US2] Add a short reading to `verdicts.md`:
  - the replay-to-arm realisation ratio on 2022 vs 027's 44–68%;
  - exit-reason shifts;
  - the recommendation among the adopted per R4: the largest 2022 gain; the combination only if
    C42 itself is ADOPTED.

**Checkpoint**: verdicts recorded. Phase 5 runs only for ADOPTED candidates.

---

## Phase 5: User Story 3 — Promote adopted couplings into the shared exit (P3)

**Goal**: An adopted coupling becomes a default-off switch (C1/C2) or a declared parameter (C4) in
the shared exit, so NNNC inherits it.

**Independent Test**:
- switched off: the fingerprints and the reference trades in 15/15 folds are unchanged;
- switched on: trades match the XC class on f00–f14 and e00–e02;
- the invariants hold.

- [ ] T016 [P] [US3] Tests first (they must FAIL), only for the adopted candidates:
  - `utils/tests/test_exit_switches.py`:
    - `apply_stop_min_depth` (None leaves the stop unchanged; a set depth deepens only stops
      tighter than it);
    - `atr_take_profit` clamps to [0.01, 0.05];
    - `stop_min_depth` outside [atr_stoploss_floor, 0) is refused at class creation;
    - FAMILY_EXITS can set both fields.
  - `evaluation/tests/test_fingerprint_switches.py`: with the new fields at None, EVERY sha
    recorded in T001 is unchanged (arm D, `st_bgo_mx`, `st_book_rev2`, XC4/XC2/XC1, and XC42 from
    T002).
  - A C4 test: a strategy declaring `stoploss_grace_hours` 12.0 resolves to XC4's fingerprint.
- [ ] T017 [US3] C1, if adopted:
  - add `ExitParams.stop_min_depth: float | None = None` and `ExitPolicy.apply_stop_min_depth` in
    `utils/ExitPolicy.py`;
  - call it in `Framework/BaseStrategy._resolve_normal_stop`, on the post-grace path only;
  - extend the bound in `utils/StopInvariants.py` to min(floor, stop_min_depth);
  - make `research/exit_coupling/paths._normal` call the same `ExitPolicy` function (one source).
- [ ] T018 [US3] C2, if adopted (C2 or C42):
  - add `ExitParams.take_profit_atr_mult: float | None = None` and `ExitPolicy.atr_take_profit`;
  - move the clamp constants out of `paths.py` into `ExitPolicy`;
  - resolve it per trade in `Framework/BaseStrategy`, after family overrides (signal-bar
    `atr_pct_roll` at open_date − 1 timeframe, cached per trade, as XC2 does);
  - update the allowlist in `Framework/tests/test_exit_policy_single_source.py` for the new clamp
    site.
- [ ] T019 [US3] In `evaluation/fingerprint.py`, omit `ExitParams` fields whose value is None
  from `FAMILY_EXITS_RESOLVED` and any other `asdict` hash (R7). T016's fingerprint test must
  pass.
- [ ] T020 [US3] Add the promoted classes, one per adopted candidate. Each switches the coupling
  on through the shared exit and replaces nothing by override:
  - `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P<id>.py`;
  - its plans `plan_st_prom_<id>.json` (f00–f14) and `plan_st_e22_prom_<id>.json` (e00–e02);
  - once, `evaluation/folds/plan_st_rev2_recheck.json`: f00 only, class
    `SimpleTrigger_RevOnlyVolreg2`, `arm_tag` `st_rev2_recheck_f00`, for T021's switch-off
    check.
- [ ] T021 [US3] Run each promoted class on f00–f14 and e00–e02. Its `trades_hash` must equal the
  XC class's in 18/18 folds (FR-011). Then check switch-off:
  - re-resolve the fingerprints from T001 (must be unchanged);
  - run `plan_st_rev2_recheck.json`; its trades_hash must equal `st_book_rev2` f00's recorded
    one.
- [ ] T022 [US3] Run `pytest utils/tests Framework/tests research/exit_coupling/tests evaluation`.
  Everything must pass, with no stop-invariant violation in any T021 run log. Write
  `specs/028-recovery-aware-exits/promotion.md`: the switches, how to declare each, and the
  equivalence and fingerprint results.

**Checkpoint**: adopted couplings are live in the shared exit, default off.

---

## Phase 6: User Story 4 — Report and hand-over (P4)

- [ ] T023 [US4] Write `specs/028-recovery-aware-exits/report.md`:
  - **SUMMARY**: per candidate, the verdict and which reversion exit NNNC should inherit.
  - The pre-registration, any deviations, and the R2 freshness caveat (the base exit settled in the
    NN era included 2022).
  - The US1 table.
  - A side-by-side table per candidate: 2023–26 EXPLORATORY (early/late) and 2022 CONFIRMATORY
    (gain, paired, risk, power).
  - Liveness.
  - The promotion summary, if any.

  Label every 2022 figure PROVISIONAL (gate 2: 2022 adjoins f00).
- [ ] T024 [US4] Update memory:
  - `project_028_recovery_aware_exits.md` with the outcome;
  - the MEMORY.md line;
  - `index_exits.md` if a switch was promoted.

  If a new freqtrade mechanic was found, add it to `docs/EXIT_AND_BACKTEST_MECHANICS.md`.
- [ ] T025 Mark tasks done in this file. Commit the report, outputs, code and plans (named files
  only).

---

## Dependencies & Execution Order

- **Phase 1:**
  - T001 first;
  - T002 ∥ T001;
  - T003 → T004.
- **Phase 2:**
  - T005 → T006;
  - T007 ∥ T005/T006;
  - T008 after T007;
  - T009 needs T004, T006, T007 and T008. Nothing in 2022 runs before it.
- **US1:** T010 → T011 → T012. STOP gate.
- **US2:** T013 → T014 → T014a → T015.
- **US3:** T016 → T017/T018 (adopted only) → T019 → T020 → T021 → T022.
- **US4:** T023 → T024 → T025.
- **Backtests:** one chain at a time (T004, T010, T013, T014a, T021).

## Parallel Example

```text
T001 baseline  ∥  T002 XC42 class
T005 tests     ∥  T007 2022 plans
```

## Implementation Strategy

- **MVP:** Phases 1–3. Pre-register, then the mechanism check on 2022. A STOP there is a valid
  recorded outcome.
- **Then:** US2 gives the adoption verdicts.
- **US3 only for what is ADOPTED.** C4 alone needs no framework code.
