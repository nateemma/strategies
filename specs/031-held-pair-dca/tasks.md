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

- [ ] T001 Create `specs/031-held-pair-dca/baseline.md` and `baseline.json`. Record:
  - the commit sha;
  - `framework_sha` (via `evaluation/fingerprint.py resolve_by_name`, `config_wf.json`) for every
    class used by the reference arms: `NNNC_MLX_Open` (wf_aro), `NNNC_A_Raw_Open`,
    `NNNC_GbbRaw_Open`, `NNNC_GbbFwdMae_Open`, `NNNC_MLX`, `SimpleTrigger_BreakoutGbb_MomExit`,
    `SimpleTrigger_RevOnlyVolreg2_P2`, and both `_G24` classes;
  - the 030 model key for each NNNC class and fold;
  - the recorded per-fold `trades_hash` of `wf_aro`, `wf_aro_g24`, `fs_gbr_gr`, `fs_gfm_gr`,
    `fs_mlx_gr`, `st_bgo_mx` and `st_prom_c2` (from `regime/arms/`).
- [ ] T002 Run the test suites and record the pass count in baseline.md:
  `../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests evaluation -q`.

---

## Phase 2: Foundational (blocking: generic helpers, no behaviour change)

- [ ] T003 In `Framework/BaseStrategy.py`, add `trade_clock_start(self, trade) -> datetime`. It
  returns `trade.date_last_filled_utc` when `getattr(self, "position_adjustment_enable", False)`
  is true, and `trade.open_date_utc` otherwise (UTC-aware). Write the docstring per research R4.
- [ ] T004 In `Framework/BaseStrategy.py`, add `_fillable_stake(self, pair) -> tuple[float, float]`,
  returning `(quote_volume, quote_volume / QUOTE_VOLUME_HEADROOM_MULT)` from the last analysed
  row. Rewrite `custom_stake_amount` to use it with identical arithmetic (keep the plot/other
  early return).
- [ ] T005 Replace the trade-age computations with `self.trade_clock_start(trade)` in:
  - `Framework/BaseStrategy.py`: custom_stoploss grace (~line 1250), custom_exit `age_hours`
    (~1386), confirm_trade_exit `t_open` (~1554);
  - `_apply_atr_take_profit` (~1178–1185): signal bar = `clock_start − timeframe`, cache key
    `(trade.id or (pair, open_date), clock_start, k)`;
  - `utils/StopInvariants.py` (~135);
  - `NNNC/_hold_window.py` (~54).

  Leave `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC2.py` unchanged (a 027 exploratory class not
  used by any 031 arm), and note it in research.md R4 as out of scope.
- [ ] T006 Run T002's suites and the exit-parameter guard test, and update the allowlist in
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

- [ ] T007 [P] [US1] `utils/tests/test_held_pair_add.py`, declaration tests:
  - `None` stays off (`position_adjustment_enable` False, `max_entry_position_adjustment` -1);
  - `{}` turns on with the defaults (`max_adds` 1, `min_age_hours` 1.0, `under_water_only` False);
  - a subclass dict merges key by key over its parent's;
  - a subclass `None` turns it off;
  - unknown key, `max_adds` 0, negative age, or non-bool `under_water_only` each raise
    `EntryFrameworkError`;
  - both finalisation paths work: `_ENTRY_FRAMEWORK_AUTO` True (NNNC) and the SimpleTrigger preset
    path that finalises explicitly.
- [ ] T008 [P] [US1] Same file, AddDecision tests. Use a fake `dp` returning a small analysed
  dataframe and a fake trade (`nr_of_successful_entries`, `date_last_filled_utc`, `is_short`,
  `has_open_orders`, `open_rate`, `pair`). One test per refusal in data-model.md order: `off`,
  `not_long`, `open_order`, `max_adds`, `row_misaligned`, `no_signal`, `too_young` (including age
  0), `in_profit`, `thin`, `below_min`. Also:
  - the accept path returns `(min(proposed, fillable, max_stake), "hp_add")`;
  - each refusal increments its counter.
- [ ] T009 [P] [US1] Same file, clock tests with `position_adjustment_enable` True and a trade whose
  `date_last_filled_utc` is 30 h after `open_date_utc`:
  - the grace branch is active again within 24 h of the last fill;
  - `custom_exit_decision` receives the age from the last fill;
  - the confirm_trade_exit minimum hold uses the last fill;
  - the ATR take-profit reads the row at `last_fill − timeframe` and re-resolves (new cache key);
  - the hold window uses the last fill;
  - with `position_adjustment_enable` False, every site uses `open_date_utc` (parametrised).
- [ ] T010 [P] [US1] Same file, stop tests:
  - after an add, the tightened (post-grace) stop price is anchored to the averaged `open_rate`;
  - `custom_stoploss(after_fill=True)` inside the restarted grace window returns `grace_level`;
  - `StopInvariantMixin` raises nothing across the sequence: entry → grace → tightened → add
    (after_fill) → grace → tightened.

### Implementation for User Story 1

- [ ] T011 [US1] In `utils/EntryFramework.py`, add:
  - `HELD_PAIR_ADD = None` and `HELD_PAIR_ADD_DEFAULTS`;
  - MRO merge and validation in `_finalise_entry_framework`, setting `position_adjustment_enable`
    and `max_entry_position_adjustment` on the class when on;
  - `HELD_PAIR_ADD`, `HELD_PAIR_ADD_DEFAULTS` and `adjust_trade_position` in `MODEL_KEY_IGNORED`.

  Check that the SimpleTrigger preset path calls the same finalisation. If it does not, call it
  there.
- [ ] T012 [US1] In `utils/EntryFramework.py`, implement `adjust_trade_position` per
  contracts/interfaces.md and the data-model AddDecision order:
  - refusal counters in an instance dict;
  - stake via `self.wallets.get_trade_stake_amount(pair, self.config["max_open_trades"],
    update=False)` (fallback `self.config["stake_amount"]`) and `_fillable_stake`;
  - `MIN_QUOTE_VOLUME` floor;
  - return `(stake, "hp_add")`.

  Log the counters once at `bot_loop_start`/backtest end if a hook exists; otherwise expose them
  for T019.
- [ ] T013 [US1] In `evaluation/fingerprint.py`, add `"HELD_PAIR_ADD"` to
  `OPTIONAL_ATTRIBUTE_MEMBERS`. Add a test (`evaluation/test_fingerprint_held_pair.py`) checking
  that an off class hashes as before and an on class hashes differently.
- [ ] T014 [US1] Run T007–T010 and T013 until they pass. Then run the full suites from T002 plus
  the model-key audit test. The pass count must be the baseline plus the new tests.
- [ ] T015 [US1] Smoke backtest. Create `NNNC/NNNC_HP_SMOKE.py` (subclass of `NNNC_A_Raw_Open`
  with `HELD_PAIR_ADD = {}`), then run one fold of ARO (f00, using the control plan's model
  key/threshold) through `regime/run_arm.py`. From the archive, assert:
  - at least one trade has two filled entry orders;
  - every add's fill date equals a candle where the signal candle (fill − 15 min) has `enter_long`
    1;
  - `open_rate` equals the amount-weighted average of the fills;
  - the add stake is ≤ 900 and ≤ the liquidity cap;
  - `row_misaligned` is 0.

  Record the result in `specs/031-held-pair-dca/smoke.md`. Delete the smoke class after the check
  and do not commit it (f00 is early half; this run is allowed before T024).
- [ ] T016 [US1] Off-identity check (FR-007, SC-001). Re-run the 7 reference arms under `_c031`
  tags as one chain: `wf_aro`, `wf_aro_g24`, `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`, `st_bgo_mx`,
  `st_prom_c2`, all 15 folds. Compare each per-fold `trades_hash`, `framework_sha` and model key
  with baseline.json. Write `specs/031-held-pair-dca/verification.md`. All must be 15/15 equal
  with zero training (no `ModelMissing`, no new weights under `saved_data`). Commit T007–T016.

**Checkpoint**: the switch works on a real backtest, and off it is byte-identical.

---

## Phase 4: User Story 2 — fidelity and harness accounting (Priority: P2)

**Goal**: Prove the backtest adds are the screen's adds, and that multi-fill trades are accounted
for correctly.

**Independent Test**: `fidelity.json` shows ≥ 95% of adds matched, and the accounting tests pass.

- [ ] T017 [P] [US2] `regime/run_arm.py summarize`: add `n_adds` and `n_trades_with_adds` (from the
  exported trades' `orders` with `ft_order_side == "buy"`/entry side and filled status, or
  `nr_of_entries` if present), emitted only when > 0. Add a test in `regime/tests/` (or
  `evaluation/`) on a synthetic export with and without adds: no new keys without adds,
  `trades_hash` definition unchanged.
- [ ] T018 [P] [US2] `evaluation/pool.py`: add a per-trade `n_entries` (additive, `None` when
  absent). Add `evaluation/test_pool_multifill.py`: a synthetic record with one two-fill trade
  checks that `stake_amount` is the total, `profit_abs` passes through, the trade pairs with a
  one-fill control trade on `(fold_id, pair, open_date)` in `compare_pooled`, and `risk.compare`
  runs.
- [ ] T019 [US2] `research/held_pair/screen.py`: record `add_e` (bar index of the add, −1 if none)
  on each simulated trade, without changing any other arithmetic. Add a `--dump-adds MODEL GRACE`
  option that writes `adds_<model>.json` (fold, pair, add date, trade open date, add stake)
  for DCA:all:1h. Re-run the early-half screen for ARO and confirm that `screen_nnnc_r2.json`'s
  ARO numbers are unchanged.
- [ ] T020 [US2] Build the fidelity arm. Create `research/held_pair/build_arms.py`, modelled on
  `research/nnnc_framework/build_arms.py`:
  - for each model, write `NNNC/NNNC_HP_<M>_DCA.py` (subclass of the base, `HELD_PAIR_ADD = {}`),
    a copy of the base `.json`, and `evaluation/folds/plan_hp_<m>_dca.json` derived from the
    control plan (`wf_aro_g24`, `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`): same folds, thresholds,
    model_dir and 24 h snapshot pin; new class and arm tag;
  - for SimpleTrigger, write `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P2_G24_DCA.py` and
    `SimpleTrigger_BreakoutGbb_MomExit_G24_DCA.py`, plus plans `st_hp_rev`, `st_hp_mom` (f00–f14)
    and `st_e22_hp_rev`, `st_e22_hp_mom` (e00–e02), derived from `st_g24_*` / `st_e22_g24_*`.

  Verify that `run_folds --dry-run` resolves every model key with no training. Then run ONLY
  `hp_aro_dca` on f00–f07.
- [ ] T021 [US2] Create `research/held_pair/fidelity.py`. It loads `hp_aro_dca`'s f00–f07 archives,
  extracts adds (pair, fill date) and add-candle exits, and matches them to `adds_ARO.json`. It
  writes `research/held_pair/fidelity.json`, containing per fold:
  - matched / backtest adds / screen adds;
  - add-candle exits;
  - screen pp vs arm pp (vs the `wf_aro_g24` control);
  - `row_misaligned` and the refusal counters if exported.

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
  - setting `HELD_PAIR_ADD = {}` (DCA:all:1h, one add) and grace 24 h;
  - the books table from research R10;
  - folds;
  - window roles and labels;
  - the rule: NNNC `judge_one` with `paired_map`/`risk_map` copied from
    `research/nnnc_framework/preregistration_r2.json`; SimpleTrigger the 2022 rule from
    `research/st_grace/preregistration.json`;
  - `min_control_trades` 200 for 2023–26 (the 2022 controls use the st_grace convention);
  - INCONCLUSIVE conditions (zero adds, thin control, inconclusive map);
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
- [ ] T026 [P] [US3] Create `research/held_pair/exposure.py` per research R9. From the pooled arm and
  control trades (with add fill times from the archives), compute:
  - average and peak committed stake;
  - adds per fold;
  - fresh entries refused for the wallet (if exported);
  - the leverage control pp.

  Add a unit test on a synthetic two-trade book with a known answer, in
  `research/held_pair/tests/test_exposure.py`.
- [ ] T027 [US3] Create `research/held_pair/judge.py`. It refuses (exit 2) if the pre-registration
  is dirty, uncommitted, or newer than any judged record (pattern: `research/st_grace/judge.py
  refusals`). It applies `judge_one` per NNNC model and `judge_2022` per SimpleTrigger book, adds
  the exposure, max drawdown and leverage control per book, sets `beats_leverage`, and writes
  `research/held_pair/verdicts.json`. Add a test that a record older than the pre-registration
  is refused, in `research/held_pair/tests/test_judge.py`.
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
- T016 (off-identity) and T015 (smoke) both need T011–T014. Run T015 before T016 so a defect shows
  on one fold.
- T019 and T020 are independent. T021 needs both. T026 can be written while T025 runs.
- No judged backtest may start before T024.

## Parallel opportunities

- T007–T010 (one test file, separate test functions; can be drafted together).
- T017 / T018 / T019 (different files).
- T026 while the T025 chain runs.

## Implementation strategy

- **MVP** = Phases 1–3: a working, default-off switch, byte-identical when off.
- US2 is the gate to any result.
- US3/US4 produce the verdicts.
