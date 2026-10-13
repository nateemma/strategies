---
description: "Task list for 031 held-pair position add (DCA)"
---

# Tasks: Held-Pair Position Add (DCA)

**Input**: `specs/031-held-pair-dca/` (spec.md, plan.md, research.md R1–R11, data-model.md,
contracts/interfaces.md, quickstart.md)

**Tests**: Included, because the spec requires each of the following and each needs a test:
- switch-off byte-identity (FR-007, SC-001);
- add behaviour and clocks (US1 scenarios 2–8);
- multi-fill accounting (FR-008);
- fidelity ≥ 95% (FR-009, SC-002);
- pre-registration precedence (FR-010, SC-003);
- stop invariants (SC-006).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`. The harness runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan ...`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`. They run as ONE sequential chain
  (check with `ps` first) and use fresh tags.
- Commit to master, staging named files only. Run `Framework/tests/test_exit_policy_single_source.py`
  before every commit that touches a file naming an exit parameter.
- **No late-half (f08–f14) DCA record for any NNNC model, and no 2022 (e00–e02) or 2023–26
  SimpleTrigger DCA record, before T024 is committed.** The only DCA backtests allowed before
  T024 are `hp_aro_dca` on f00–f07 (T020, the fidelity run) and the one-pair smoke test (T015).
- **US2 is a STOP gate.** If T022 fails (< 95% matched, or `row_misaligned` > 0), diagnose and fix
  the switch (re-running T015–T022) before any pre-registration. Never pre-register an arm that
  failed fidelity.
- Labels:
  - NNNC late half is CONFIRMATORY and PROVISIONAL (gate 2, one era);
  - NNNC early half is EXPLORATORY;
  - SimpleTrigger 2022 is CONFIRMATORY;
  - SimpleTrigger 2023–26 is EXPLORATORY.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `specs/031-held-pair-dca/baseline.md` and `baseline.json`. Record:
  - the commit sha;
  - the reference set (research R10): `wf_aro_b030` (arm D, 6.5 h pinned), `wf_aro_g24`,
    `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`, `st_bgo_mx`, `st_prom_c2`;
  - for each reference plan and fold, the `strategy_class` read from the plan (NNNC classes are
    per fold, e.g. `NNNC_ARO_f00`, `NNNC_FS_GBR_GR_f00`), its `framework_sha` (via
    `evaluation/fingerprint.py resolve_by_name`, `config_wf.json`) and its 030 model key;
  - the `framework_sha` of the bases `NNNC_A_Raw_Open`, `NNNC_GbbRaw_Open`, `NNNC_GbbFwdMae_Open`,
    `NNNC_MLX`, `SimpleTrigger_BreakoutGbb_MomExit`, `SimpleTrigger_RevOnlyVolreg2_P2` and both
    `_G24` classes;
  - the recorded per-fold `trades_hash` of every reference arm (from `regime/arms/`).
- [X] T002 Run the test suites (the command in quickstart.md §1) and record the pass count in
  baseline.md.

---

## Phase 2: Foundational (blocking: generic helpers, no behaviour change)

- [X] T003 In `Framework/BaseStrategy.py`, add `trade_clock_start(self, trade) -> datetime`. It
  returns `trade.date_last_filled_utc` when `getattr(self, "position_adjustment_enable", False)`
  is true, and `trade.open_date_utc` otherwise (UTC-aware). Write the docstring per research R4.
- [X] T004 In `Framework/BaseStrategy.py`, add `_fillable_stake(self, pair) -> tuple[float, float]`,
  returning `(quote_volume, quote_volume / QUOTE_VOLUME_HEADROOM_MULT)` from the last analysed
  row. Rewrite `custom_stake_amount` to use it with identical arithmetic (keep the plot/other
  early return).
- [X] T005 Replace the trade-age computations with `self.trade_clock_start(trade)` in:
  - `Framework/BaseStrategy.py`: custom_stoploss grace (~line 1250), custom_exit `age_hours`
    (~1386), confirm_trade_exit `t_open` (~1554);
  - `_apply_atr_take_profit` (~1178–1185): signal bar = `clock_start − timeframe`, cache key
    `(trade.id or (pair, open_date), clock_start, k)`;
  - `utils/StopInvariants.py` (~135);
  - `NNNC/_hold_window.py` (~54).

  Leave `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC2.py` unchanged (a 027 exploratory class not
  used by any 031 arm), and note it in research.md R4 as out of scope.
- [X] T006 Run the quickstart §1 suites and the exit-parameter guard test, and update the allowlist in
  `Framework/tests/test_exit_policy_single_source.py` if any touched file now newly names an exit
  parameter. The pass count must equal the baseline. Commit T003–T006.

**Checkpoint**: the helpers exist, and with every existing class having
`position_adjustment_enable = False`, behaviour is unchanged by construction.

---

## Phase 3: User Story 1 — the default-off switch (Priority: P1) 🎯 MVP

**Goal**: `HELD_PAIR_ADD` declared on any entry-framework strategy adds one stake on a fresh
signal for a held pair. Off, nothing changes.

**Independent Test**: The unit tests pass. The smoke backtest shows a two-fill trade with the
right timing, size and average. The off-identity check is 15/15 on all reference arms.

### Tests for User Story 1 (write first, confirm they fail)

- [X] T007 [P] [US1] `utils/tests/test_held_pair_add.py`, declaration tests:
  - `None` stays off (`position_adjustment_enable` False, `max_entry_position_adjustment` -1);
  - `{}` turns on with the defaults (`max_adds` 1, `min_age_hours` 1.0, `under_water_only` False);
  - a subclass dict merges key by key over its parent's;
  - a subclass `None` turns it off;
  - unknown key, `max_adds` 0, negative age, or non-bool `under_water_only` each raise
    `EntryFrameworkError`;
  - both finalisation paths work: `_ENTRY_FRAMEWORK_AUTO` True (NNNC) and the SimpleTrigger preset
    path that finalises explicitly.
- [X] T008 [P] [US1] Same file, AddDecision tests. Use a fake `dp` returning a small analysed
  dataframe and a fake trade (`nr_of_successful_entries`, `date_last_filled_utc`, `is_short`,
  `has_open_orders`, `open_rate`, `pair`). One test per refusal in data-model.md order: `off`,
  `not_long`, `open_order`, `max_adds`, `row_misaligned`, `no_signal`, `too_young` (including age
  0), `in_profit`, `thin`, `below_min`. Also:
  - the accept path returns `(min(proposed, fillable, max_stake), "hp_add")`;
  - each refusal increments its counter.
- [X] T009 [P] [US1] Same file, clock tests with `position_adjustment_enable` True and a trade whose
  `date_last_filled_utc` is 30 h after `open_date_utc`:
  - the grace branch is active again within 24 h of the last fill;
  - `custom_exit_decision` receives the age from the last fill;
  - the confirm_trade_exit minimum hold uses the last fill;
  - the ATR take-profit reads the row at `last_fill − timeframe` and re-resolves (new cache key);
  - the hold window uses the last fill;
  - with `position_adjustment_enable` False, every site uses `open_date_utc` (parametrised).
- [X] T010 [P] [US1] Same file, stop tests:
  - after an add, the tightened (post-grace) stop price is anchored to the averaged `open_rate`;
  - `custom_stoploss(after_fill=True)` inside the restarted grace window returns `grace_level`;
  - `StopInvariantMixin` raises nothing across the sequence: entry → grace → tightened → add
    (after_fill) → grace → tightened.

### Implementation for User Story 1

- [X] T011 [US1] In `utils/EntryFramework.py`, add:
  - `HELD_PAIR_ADD = None` and `HELD_PAIR_ADD_DEFAULTS`;
  - MRO merge and validation in `_finalise_entry_framework`, setting `position_adjustment_enable`
    and `max_entry_position_adjustment` on the class when on;
  - `HELD_PAIR_ADD`, `HELD_PAIR_ADD_DEFAULTS` and `adjust_trade_position` in `MODEL_KEY_IGNORED`.

  Check that the SimpleTrigger preset path calls the same finalisation. If it does not, call it
  there.
- [X] T012 [US1] In `utils/EntryFramework.py`, implement `adjust_trade_position` per
  contracts/interfaces.md and the data-model AddDecision order:
  - refusal counters in an instance dict;
  - stake via `self.wallets.get_trade_stake_amount(pair, self.config["max_open_trades"],
    update=False)` (fallback `self.config["stake_amount"]`) and `_fillable_stake`;
  - `MIN_QUOTE_VOLUME` floor;
  - return `(stake, "hp_add")`.

  Counters (`adds` plus one per refusal) are written to the run's entry-condition sink as
  `<StrategyClass>_hp_add.json`, at process exit via `atexit` (registered on the first counted
  call), and only when the switch is on (data-model AddDecision). T008 gains a test that the file
  is written with the counts.
- [X] T013 [US1] In `evaluation/fingerprint.py`, add `"HELD_PAIR_ADD"` to
  `OPTIONAL_ATTRIBUTE_MEMBERS`. Add a test (`evaluation/test_fingerprint_held_pair.py`) checking
  that an off class hashes as before and an on class hashes differently.
- [X] T014 [US1] Run T007–T010 and T013 until they pass. Then run the full suites from T002 plus
  the model-key audit test. The pass count must be the baseline plus the new tests.
- [X] T015 [US1] Smoke backtest. Write `research/held_pair/build_arms.py` (spec in T020) first and run
  it with `--only ARO --folds f00`. That generates `NNNC/NNNC_HP_ARO_DCA_f00.py/.json` (per-fold
  class, `model_dir` `saved_data/NNNC_ARO_f00`, `HELD_PAIR_ADD = {}`), `plan_hp_aro_dca.json`
  and a one-fold copy `plan_hp_smoke.json` (`run_id`/`arm_tag` `hp_smoke`, f00 only). Run
  `run_folds --plan .../plan_hp_smoke.json`. From the
  archive and record, assert:
  - at least one trade has two filled entry orders;
  - every add's fill date equals a candle where the signal candle (fill − 15 min) has `enter_long`
    1;
  - `open_rate` equals the amount-weighted average of the fills;
  - the add stake is ≤ 900 and ≤ the liquidity cap;
  - the record's `hp_counters.row_misaligned` is 0 (this needs T017's sidecar pickup, so T017 runs
    before T015).

  Record the result in `specs/031-held-pair-dca/smoke.md`. The `hp_smoke` record is f00, early
  half, and is allowed before T024. The pre-registration lists it as exempt along with the
  fidelity records.
- [X] T016 [US1] Off-identity check (FR-007, SC-001). Re-run the 7 reference arms of T001 under
  `_c031` tags as one chain, all 15 folds. Copy each plan with only `run_id`/`arm_tag` renamed, so
  classes, snapshots and model dirs are unchanged. Compare each per-fold `trades_hash`,
  `framework_sha` and model key with baseline.json. Every off record must carry no `hp_counters`
  and no `n_adds`. Write `specs/031-held-pair-dca/verification.md`. All must be 15/15 equal
  with zero training (no `ModelMissing`, no new weights under `saved_data`). Commit T007–T016.

**Checkpoint**: the switch works on a real backtest, and off it is byte-identical.

---

## Phase 4: User Story 2 — fidelity and harness accounting (Priority: P2)

Execution note: T017 and the builder part of T020 run BEFORE T015 (Phase 3). They are listed here
because their deliverables belong to US2. See Dependencies.

**Goal**: Prove the backtest adds are the screen's adds, and that multi-fill trades are accounted
for correctly.

**Independent Test**: `fidelity.json` shows ≥ 95% of adds matched, and the accounting tests pass.

- [X] T017 [US2] `regime/run_arm.py summarize`:
  - add `n_adds` and `n_trades_with_adds` from the exported trades' `orders` (entry-side, filled),
    or `nr_of_entries` if present, emitted only when > 0;
  - copy the `hp_` keys of `entry_condition_counts` (the T012 sink file, merged by
    `aggregate_sink`; the sink directory is per run) into the record as `hp_counters`.

  Add a test (in `regime/tests/` if it exists, else `evaluation/test_run_arm_adds.py`) on a
  synthetic export with and without adds, and with and without a sidecar: no new keys when off,
  `trades_hash` definition unchanged. **Run this before T015**, which needs it.
- [X] T018 [P] [US2] `evaluation/pool.py`: add a per-trade `n_entries` (additive, `None` when
  absent). Add `evaluation/test_pool_multifill.py`: a synthetic record with one two-fill trade
  checks that `stake_amount` is the total, `profit_abs` passes through, the trade pairs with a
  one-fill control trade on `(fold_id, pair, open_date)` in `compare_pooled`, and `risk.compare`
  runs.
- [X] T019 [US2] `research/held_pair/screen.py`: record `add_e` (bar index of the add, −1 if none)
  on each simulated trade, without changing any other arithmetic. Add a `--dump-adds MODEL GRACE`
  option that writes `adds_<model>.json` (fold, pair, add date, trade open date, add stake)
  for DCA:all:1h. Re-run the early-half screen for ARO and confirm that `screen_nnnc_r2.json`'s
  ARO numbers are unchanged.
- [ ] T020 [US2] Build the fidelity arm. Create `research/held_pair/build_arms.py`, modelled on
  `research/nnnc_framework/build_arms.py`:
  - for each model and fold, write `NNNC/NNNC_HP_<M>_DCA_fNN.py` (per-fold class as
    `research/nnnc_framework/build_arms.py` line ~123 does, subclass of the base,
    `HELD_PAIR_ADD = {}`) and a copy of the fold's `.json`; then write
    `evaluation/folds/plan_hp_<m>_dca.json` derived from the control plan (`wf_aro_g24`, `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`): same folds, thresholds,
    model_dir and 24 h snapshot pin; new class and arm tag;
  - options `--only M` and `--folds`, and `--smoke` writing `plan_hp_smoke.json` (used by T015);
  - for SimpleTrigger, write `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P2_G24_DCA.py` and
    `SimpleTrigger_BreakoutGbb_MomExit_G24_DCA.py`, plus plans `st_hp_rev`, `st_hp_mom` (f00–f14)
    and `st_e22_hp_rev`, `st_e22_hp_mom` (e00–e02), derived from `st_g24_*` / `st_e22_g24_*`.

  Verify that `run_folds --dry-run` resolves every model key with no training. Then run ONLY
  `hp_aro_dca` on f00–f07.
- [ ] T021 [US2] Create `research/held_pair/fidelity.py`. It loads `hp_aro_dca`'s f00–f07 archives,
  extracts adds (pair, fill date) and add-candle exits, and matches them to `adds_ARO.json`. It
  writes `research/held_pair/fidelity.json`, containing per fold:
  - matched / backtest adds / screen adds, i.e. precision (the gate) and recall (reported);
  - add-candle exits and their pp;
  - screen pp vs arm pp (vs the `wf_aro_g24` control), with the gap and the add-candle-exit part;
  - `hp_counters` from the arm records (T017), including `row_misaligned`.

  Exit 2 if matched < 95% of backtest adds or `row_misaligned` > 0.
- [ ] T022 [US2] Run fidelity. **STOP gate**: on exit 2, diagnose the mismatches (list the first 20
  with both sides' context), fix the switch, and repeat T015/T020–T022 under fresh tags. Record
  the result, and any fixes, in `specs/031-held-pair-dca/fidelity.md`. Commit T017–T022 (the
  generated classes and plans included; no late-half or 2022 record exists yet).

**Checkpoint**: the instrument is audited.

---

## Phase 5: User Story 3 — pre-registered judgement (Priority: P3)

**Goal**: One recorded verdict per book under a rule fixed before the judged data is touched.

**Independent Test**: The pre-registration commit is older than every judged record, and
`verdicts.json` holds 6 decisions.

- [ ] T023 [US3] Write `research/held_pair/preregistration.json` and `.md`:
  - `exempt_records`: `hp_smoke` f00 and `hp_aro_dca` f00–f07, each labelled EXPLORATORY;
  - setting `HELD_PAIR_ADD = {}` (DCA:all:1h, one add) and grace 24 h;
  - the books table from research R10;
  - folds;
  - window roles and labels;
  - the rule: NNNC `judge_one` with `paired_map`/`risk_map` copied from
    `research/nnnc_framework/preregistration_r2.json`; SimpleTrigger the 2022 rule from
    `research/st_grace/preregistration.json`;
  - `min_control_trades` 200 for 2023–26, and none for 2022 (st_grace convention), stated
    explicitly;
  - INCONCLUSIVE conditions: zero adds (an override on both rules), thin control (2023–26), and an
    inconclusive map;
  - quality gate 5 for adopted gains under 1 pp;
  - reporting: exposure, max drawdown, leverage control;
  - the screen's early-half predictions and T022's fidelity result;
  - "switch stays default-off".
- [ ] T024 [US3] Commit the pre-registration (named files only), and record its commit timestamp
  in the `.md`.
- [ ] T025 [US3] Run the judged chain as ONE sequential script in the scratchpad (`chain_hp.sh`,
  log `.log`): `hp_aro_dca` f08–f14; `hp_gbr_dca`, `hp_gfm_dca` and `hp_mlx_dca` f00–f14;
  `st_e22_hp_rev` and `st_e22_hp_mom` e00–e02; `st_hp_rev` and `st_hp_mom` f00–f14. Monitor it to
  completion. Expect 0 errors and every record newer than T024.
- [X] T026 [P] [US3] Create `research/held_pair/exposure.py` per research R9. From the pooled arm and
  control trades (with add fill times from the archives), compute:
  - average and peak committed stake;
  - adds per fold;
  - crowd-out: the 24 h grace study's split into same-entry Δpp / control entries forgone (n,
    pair-held n, pp) / new entries (n, pp), keyed on (fold, pair, open_date);
  - the leverage control pp.

  Add a unit test on a synthetic two-trade book with a known answer, in
  `research/held_pair/tests/test_exposure.py`.
- [ ] T027 [US3] Create `research/held_pair/judge.py`. It refuses (exit 2) if the pre-registration is
  dirty or uncommitted, or if any DCA record not in `exempt_records` predates its commit (pattern:
  `research/st_grace/judge.py refusals`). It then:
  - applies `judge_one` per NNNC model and `judge_2022` per SimpleTrigger book;
  - applies the `n_adds == 0` → INCONCLUSIVE override to both;
  - adds exposure, max drawdown, the leverage control and the decomposition per book;
  - sets `beats_leverage`;
  - writes `research/held_pair/verdicts.json`.

  Tests in `research/held_pair/tests/test_judge.py`: an older non-exempt record is refused; an
  older exempt record is accepted; zero adds gives INCONCLUSIVE.
- [ ] T028 [US3] Run the judge. For any ADOPTED book whose gain is under 1 pp, re-run one fold under
  a fresh tag and confirm an identical trade set (gate 5). Commit `verdicts.json`, the judge,
  exposure and tests.

---

## Phase 6: User Story 4 — report and hand-over (Priority: P4)

- [ ] T029 [US4] Write `research/held_pair/report.md`. Per book, include:
  - the screen prediction, arm result and leverage control side by side;
  - early/late (or 2022 and 2023–26) gains;
  - paired and risk verdicts;
  - max drawdown and exposure, arm vs control;
  - adds;
  - evidence labels.

  Also include:
  - the fidelity result and any add-candle exit effect;
  - a book whose gain ≤ its leverage control, stated as "explained by extra capital";
  - a recommendation on enabling the switch for NNNC, stated as the user's decision. The switch
    stays default-off.
- [ ] T030 [US4] Update memory: create `project_031_held_pair_dca.md` and add its line to MEMORY.md
  "Current state". Update the st_grace memory pointer. Add a short section on position
  adjustment mechanics (R1/R2) to `docs/EXIT_AND_BACKTEST_MECHANICS.md`.

---

## Phase 7: Polish

- [ ] T031 Mark the tasks `[X]` and fill the checklist status in `specs/031-held-pair-dca/report.md`
  (a pointer to the research report plus a requirement → evidence table for FR-001..014 and
  SC-001..006).
- [ ] T032 Run the full suites and the exit-parameter guard test one last time, then commit the
  report, docs and spec updates.

---

## Dependencies

- Phase 1 → Phase 2 → US1 (T007–T016) → US2 (T017–T022, STOP gate) → US3 (T023–T028) → US4 →
  Polish.
- T016 (off-identity) and T015 (smoke) both need T011–T014. T015 also needs T017 (sidecar pickup)
  and the builder part of T020, so run in the order T017 → builder → T015 → T016. A defect then
  shows on one fold first.
- T019 and T020 are independent. T021 needs both. T026 can be written while T025 runs.
- No judged backtest may start before T024.

## Parallel opportunities

- T007–T010 (one test file, separate test functions; can be drafted together).
- T018 / T019 (different files). T017 runs early, before T015.
- T026 while the T025 chain runs.

## Implementation strategy

- **MVP** = Phases 1–3: a working, default-off switch, byte-identical when off.
- US2 is the gate to any result.
- US3/US4 produce the verdicts.
