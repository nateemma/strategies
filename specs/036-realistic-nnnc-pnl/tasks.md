---
description: "Task list for 036 realistic P&L for NNNC"
---

# Tasks: Realistic P&L for NNNC

**Input**: `specs/036-realistic-nnnc-pnl/`: spec.md, plan.md, research.md (R1–R12),
data-model.md, contracts/execution_recorder.md, contracts/cost_budget_stake.md,
contracts/study_cli.md, quickstart.md.

**Tests**: Included. The spec requires each of these, and each needs a test:
- default-off byte identity and model-key neutrality for both mixins (FR-014, SC-006);
- recorder idempotence across restart (FR-005);
- refusal to train live (FR-003);
- launcher isolation from the momentum bot (FR-001);
- walking the book and curve inversion (FR-009, FR-013);
- parity cause classification (FR-007);
- the recording gate (FR-019).

**Conventions**:
- **Paths and interpreter:** paths are relative to `user_data/strategies/`, and Python is
  `../../.venv/bin/python`. Study tools run as
  `PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.<tool>`.
- **Fold runner:** `run_folds` runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/run_folds.py --plan user_data/strategies/evaluation/folds/<plan>.json`.
- **Chains:** training and backtests need `set -a; source ~/.freqtrade_env; set +a` and run as ONE
  sequential chain. Check `ps` first. On a low-memory kill, report and wait for the user.
- **The momentum dry-run** (pid 2661, port 8080, `tradesv3.dryrun.sqlite`,
  `config_mom_dryrun_15m.json`) is never stopped, restarted or reconfigured. Never use
  `pkill -f freqtrade`.
- **Recorded data** goes to `user_data/data/live_nnnc/` (never committed). It is parsed with
  json/gzip/pandas and never executed.
- **Commits:** to master, staging named files only. Run the full suite and check pytest's exit
  code before every commit:
  `../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests research/held_pair/tests research/candidate_training/tests research/width_relative/tests research/label_screen/tests research/control_sizer/tests research/new_inputs/tests research/live_pnl/tests evaluation -q -p no:cacheprovider`.
- **Gate:** US2 and US3 judging tools refuse until `gate.json` is open (FR-019).
- **Results** are at most PROVISIONAL (FR-020). Promotion is the user's decision.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `research/live_pnl/__init__.py`, `research/live_pnl/tests/__init__.py` and
  `research/live_pnl/parity/.gitkeep`, and confirm that `research/live_pnl/spec_036_draft.md` is
  present (committed affb6069). Add `research/live_pnl/tests` to the
  suite command in AGENT_GUIDE.md's Tests line.
- [X] T002 [P] Confirm `user_data/data/live_nnnc/` resolves outside the nested repository, and
  that `tradesv3.live_nnnc.sqlite*` at the repo root is not tracked by the strategies repository.
  Record both checks in `research/live_pnl/report.md` (new file, "Setup" section).

---

## Phase 2: Foundational (blocks US1–US3)

- [X] T003 [P] Tests first, in `research/live_pnl/tests/test_recorder.py`, for the contract
  `contracts/execution_recorder.md`:
  - (a) with `RECORD_EXECUTION = False`, and with runmode backtest, every overridden method returns
    super()'s value and writes no file;
  - (b) a class with the mixin resolves to the same feature-030 model key as without it
    (`Framework/ModelKey.resolve` on `NNNC_A_Raw_Open` with and without the mixin);
  - (c) duplicate keys across two recorder instances (a simulated restart) are written once;
  - (d) a raising orderbook call yields `depth_missing = true` and the trade decision is unchanged;
  - (e) a `RECORD_DIR` inside the strategies repository is refused.
- [X] T004 Implement `utils/ExecutionRecorder.py` (`ExecutionRecorderMixin`) to pass T003. The
  overrides are `custom_stake_amount`, `confirm_trade_entry`, `order_filled`,
  `confirm_trade_exit`, `bot_loop_start` and `populate_entry_trend`. All call super() first, and
  all I/O goes through one append+fsync writer with daily files: `orders-`, `fills-`, `exits-`,
  `signals-` (`.jsonl`) and `depth-` (`.jsonl.gz`). Record fields are exactly data-model.md's
  OrderRecord, FillRecord, ExitRecord, DepthSnapshot and SignalRecord. Depth uses
  `dp.orderbook(pair, maximum=RECORD_DEPTH_LEVELS)` with a 3 s timeout, and scheduled snapshots are
  "at most one per pair per candle".
- [X] T005 [P] Tests in `research/live_pnl/tests/test_live_fold.py`:
  - (a) the generated `plan_live_aro.json` has one fold, `train_timerange` `20210101-<cut>`, and
    the control's params snapshot and sha;
  - (b) a cut later than the last downloaded 15m candle of any whitelist pair is refused;
  - (c) an existing run_id is refused;
  - (d) `NNNC_LIVE_<id>_REC` resolves to the same key as `NNNC_LIVE_<id>`;
  - (e) the tool refuses while another trainer is running (the 032 driver's busy check, injected).
- [X] T006 Implement `research/live_pnl/live_fold.py` (`--cut`, `--run-id`) to pass T005. It:
  - builds the plan from `plan_wf_aro_g24.json` (base `NNNC_A_Raw_Open`, the same params snapshot,
    seed 1, `test_timerange` `<cut>-<cut+90d>`, class `NNNC_LIVE_<id>`);
  - calls `run_folds.generate_fold` and `train_fold`, then `fold_has_model` (research R1);
  - picks the threshold with `evaluation/pick_threshold.py --target-rate 0.014735` on the fold
    (using `fold_has_model`, not `model_exists`), and regenerates the fold with the threshold;
  - writes `NNNC/NNNC_LIVE_<id>_REC.py` (ExecutionRecorderMixin + `NNNC_LIVE_<id>`;
    `RECORD_EXECUTION = True`; `RECORD_DIR` = `user_data/data/live_nnnc`; `RECORD_RUN_ID = <id>`)
    and its `.json` copied from the fold's with `strategy_name` set. The threshold is scored on the
    training window (`TrainWindow.from_fold`), as for every control;
  - appends a LiveModelRecord to `research/live_pnl/live_models.json` and retires the previous one.
- [X] T007 [P] Write `config/config_live_nnnc.json`:
  - from `config_wf.json`: whitelist (10 pairs), StaticPairList, stake 900, max_open 10, wallet
    10,000, 15m;
  - from `config_mom_dryrun_15m.json`: `entry_pricing`, `exit_pricing`, `order_types`,
    `unfilledtimeout`;
  - `dry_run: true`, `api_server.listen_port` 8081, `bot_name` "nnnc_live", telegram off;
  - no secrets (they come from env).

  Add a test in `research/live_pnl/tests/test_config.py` that it parses, that its port and
  bot_name differ from the momentum config's, and that dry_run is true.
- [X] T008 [P] Tests in `research/live_pnl/tests/test_launcher.py` (run the script with a
  `--check` mode that performs only the refusals). It must refuse when:
  - the port is in use (bind a socket in the test);
  - the db path equals `tradesv3.dryrun.sqlite`;
  - the pid file names a live process;
  - the strategy key does not match `live_models.json`.
- [X] T009 Implement `scripts/live_nnnc.sh` (zsh; run from the repo root) with `start`, `stop`,
  `status` and `--check`, to pass T008. It:
  - sources `~/.freqtrade_env` and exports `FT_MODEL_EVAL_STRICT=1`;
  - reads the current `live` record from `research/live_pnl/live_models.json`;
  - runs `nohup .venv/bin/freqtrade trade --dry-run -c user_data/strategies/config/config_live_nnnc.json --db-url sqlite:///tradesv3.live_nnnc.sqlite --strategy-path user_data/strategies/NNNC -s <class>_REC --logfile user_data/logs/live_nnnc.log`;
  - writes `user_data/logs/live_nnnc.pid`; `stop` signals only that pid, after checking that its
    command line contains `config_live_nnnc.json`.

**Checkpoint**: the full suite passes, and existing arms are unchanged (T003 b; plus a byte check:
re-run `wf_aro_g24` f00 with the current code and compare the trades hash with the record made
with HELD_PAIR_ADD declared off, as in 034/035 SC-003).

---

## Phase 3: User Story 1 - Run NNNC live and record execution (Priority: P1) 🎯 MVP

**Goal**: the live NNNC control running on a refreshed model, with complete records and a daily
parity report.

**Independent Test**: after one week, at least 99% of orders have records and at least 95% have
depth; parity reports exist for each day with unexplained differences resolved; the momentum bot
is unchanged.

- [X] T010 [US1] Download fresh 15m data for the 10 pairs (repo root):
  `.venv/bin/freqtrade download-data -c user_data/strategies/config/config_wf.json -t 15m --days 10`.
  Do not run it while any chain is running. Record the last candle per pair in `report.md`.
- [X] T011 [US1] Train the live model:
  `PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.live_fold --cut <first undownloaded day> --run-id live1`
  (one chain, `ps` first). Verify that:
  - `scripts/model_index.py key NNNC_LIVE_live1_REC` equals the record;
  - the threshold lies in the controls' historical range (0.73–0.79), or the difference is
    explained in `report.md` before going on;
  - the `_REC` class runs the current NNNC defaults (FR-002): dump `HELD_PAIR_ADD` (expect
    `{'in_profit_only': True}`) and the stop-loss grace (expect 24 h) from an instantiated class,
    and record them in `report.md`.

  Commit `live_models.json`, the plan and the generated classes.
- [X] T012 [US1] Start the bot with `zsh user_data/strategies/scripts/live_nnnc.sh start` (repo
  root). Within 30 minutes, verify:
  - `status` reports running on 8081;
  - the momentum bot is unchanged (pid 2661, same start time, port 8080 responds);
  - `user_data/data/live_nnnc/live1/` has one scheduled depth snapshot and one signal row per
    pair per candle;
  - the bot log has no training and no `ModelMissing`.

  Record the start in `report.md` and live_models.json (`started`).
- [X] T013 [P] [US1] Tests in `research/live_pnl/tests/test_parity.py` for each R4 cause on
  synthetic live and backtest trade sets and signal logs:
  - matched;
  - prediction_difference (live and backtest p_buy differ by more than 0.01 for the candle);
  - data_revision (a different close in the signal log);
  - gap_fill (a zero-volume bar);
  - timing (an entry one candle apart, or an unfilled limit);
  - wallet_state, including a position carried from an earlier day's difference;
  - UNEXPLAINED.

  The cause order is applied as declared, with prediction_difference checked first. A further test
  checks that only the day's entries and exits are compared, while the backtest range starts at
  the live run's start.
- [X] T014 [US1] Implement `research/live_pnl/parity.py --day` to pass T013. It:
  - downloads 15m candles for the day (with the download guard of T010);
  - backtests the live class from the live run's `started` date through the end of the day
    (cumulative; plus startup candles) with `FT_MODEL_EVAL_STRICT=1` (repo root,
    `freqtrade backtesting`). Record the result path;
  - gets the backtest's p_buy for every candle of the day from the strategy's own forward pass in
    backtest runmode (`research.nnnc_framework.signals.fold_table(cls, timerange)`, which writes p_buy per
    bar), for the prediction_difference cause. Backtest signal exports cover only candles that
    signalled, so they cannot price live-only entries;
  - reads the live trades from `tradesv3.live_nnnc.sqlite` read-only (`?mode=ro`);
  - matches on (pair, open candle);
  - writes `research/live_pnl/parity/<day>.json` per data-model ParityReport.
- [X] T015 [P] [US1] Implement `research/live_pnl/drift.py` (R11). It computes the weekly live BUY
  rate from the signal logs against 0.014735, flags outside ×0.5–×2, and writes
  `research/live_pnl/drift.json`. Add a test in `tests/test_drift.py` on a synthetic signal log.
- [X] T016 [P] [US1] Implement `research/live_pnl/gate.py` (R10), with tests in
  `tests/test_gate.py`:
  - closed at 27 days, or with any pair under 5 orders;
  - open at 28 days with every pair at 5 or more orders;
  - open at 84 days with thin pairs listed.

  It writes `research/live_pnl/gate.json`.
- [ ] T017 [US1] Run parity daily (from the first full day) and drift and gate weekly. Commit a
  weekly summary (`parity/week-<n>.json` plus gate.json and drift.json) and a `report.md` "Live
  log" entry. Investigate any UNEXPLAINED difference before any P&L read, and record defects in
  `report.md`.
- [ ] T018 [US1] Retrain procedure (FR-004): at 90 days after `live1`'s cut, run T010–T012 with
  `--run-id live2` and the new cut, after `live_nnnc.sh stop`. Open trades complete under the old
  record's attribution: the recorder stamps `run_id` on each order. Document this in `report.md`.
  It is scheduled, not run now.

**Checkpoint**: US1 is running. The 4-week clock (FR-019) starts at T012.

---

## Phase 4: User Story 2 - Price execution from recorded depth (Priority: P2)

**Goal**: a validated cost model, and realistic baselines for the control and width book.

**Independent Test**: held-out error per pair is reported against the tolerance; the restated
`wf_aro_g24` and `w_aro` have gross, cost and net per fold and half.

- [X] T019 [P] [US2] Tests in `tests/test_depth.py`:
  - cost(S) from a synthetic book: a buy walks the asks, a sell walks the bids, against the mid,
    in bps;
  - a stake beyond the book is marked extrapolated;
  - a monotone fit and its inverse round-trip within 0.5 bps;
  - tier assignment by terciles.
- [X] T020 [US2] Implement `research/live_pnl/depth.py` (walking the book, the stake grid
  100…5,000 USDT, participation p = S ÷ rolling 24 h quote volume up to and including the
  snapshot's candle) to pass T019.
- [X] T021 [US2] Write and commit `research/live_pnl/preregistration_cost.json` BEFORE fitting.
  It declares:
  - the holdout (the last 25% of recorded days);
  - the tolerance (per-pair median absolute error ≤ 5 bps at the 900 USDT stake);
  - the choice rule (participation if its pooled held-out error ≤ the per-pair constant's,
    otherwise per_pair, with the re-statement EXTRAPOLATED);
  - the tier rule;
  - the coverage floor (pairs under 80% depth coverage excluded).
- [X] T022 [US2] Implement `research/live_pnl/cost_model.py --fit`, which refuses while the gate is
  closed or T021 is uncommitted. It fits per R5, applies the pre-registered rule, and writes
  `research/live_pnl/cost_model.json` (data-model CostModel, with its sha recorded). Add a test
  of the refusals in `tests/test_cost_model.py`.
- [X] T023 [US2] Implement `research/live_pnl/restate.py --arm` (R6), with a test in
  `tests/test_restate.py`:
  - trade identity is unchanged;
  - the cost uses the causal rolling volume at the entry and exit candles;
  - a zero-cost model returns gross = net.
- [ ] T024 [US2] Run `cost_model --fit` and `restate` on `wf_aro_g24` (historical records, held-pair
  add off) and `w_aro` (after the gate opens). `cap_aro_ref` is restated in T030; `report.md`
  labels which baseline each number belongs to. Commit the outputs and add a `report.md`
  "Execution cost" section: the cost curves per tier, the held-out error per pair, the excluded
  pairs, and the restated baselines per half. Label the baselines as execution measurements on
  fold models that inherit the full-history scaler (FR-020; the plan.md Principle XI exception).

- [ ] T025 [US2] Implement `research/live_pnl/model_check.py` (research R13; FR-016), with a test
  in `tests/test_model_check.py`:
  - a synthetic order set with a constant offset sets the flag on the second consecutive check;
  - with no offset, the flag is never set.

  Run it every 4 weeks once `cost_model.json` exists, and commit `model_check.json`.

**Checkpoint**: realistic baselines exist; US3 can run.

---

## Phase 5: User Story 3 - Cost-budget stake cap and capacity curve (Priority: P3)

**Goal**: net dollar P&L across cost budgets, judged against the current cap.

**Independent Test**: off is byte-identical; each budget arm is live; `capacity.json` has paired
and risk verdicts per budget.

- [X] T026 [P] [US3] Tests in `tests/test_cost_budget.py` per `contracts/cost_budget_stake.md`:
  - (a) off: `custom_stake_amount` and `confirm_trade_entry` return the parent's values exactly,
    and the model key is unchanged;
  - (b) on: the stake is min(proposed, S_max), with S_max from the inverse curve;
  - (c) a missing model, a sha mismatch, or a pair without a tier is refused;
  - (d) the floors are kept.
- [X] T027 [US3] Implement `utils/CostBudgetStakeMixin.py` to pass T026, recording the requested
  and granted stake per trade (custom_data or entry tag suffix; the record must be readable from
  the backtest zip).
- [X] T028 [US3] Write and commit `research/live_pnl/preregistration_capacity.json` BEFORE any
  capacity backtest. It declares:
  - the budgets {5, 10, 20, 40} bps;
  - the reference `cap_aro_ref` (the control re-run with current defaults);
  - net P&L via restate;
  - judge_one on net per fold (early and late), with the risk verdict not WORSENED;
  - the liveness rule (the realised stake distribution differs from the reference, KS p < 0.01,
    or the arm is INERT);
  - PROVISIONAL labelling, and the execution-measurement label (FR-020; the Principle XI
    exception);
  - quality gate 5 (paired seed robustness) is met by determinism: fixed weights and
    deterministic backtests.
- [ ] T029 [US3] Create the classes `NNNC/NNNC_CAP_ARO_B{5,10,20,40}.py`/`.json` (control +
  CostBudgetStakeMixin, with params merged from the control) and the plans
  `evaluation/folds/plan_cap_aro_{ref,b5,b10,b20,b40}.json`, copied from `plan_wf_aro_g24.json`
  with the same thresholds. Verify that every class resolves to the control's model key for each
  fold (no training).
- [ ] T030 [US3] Run the 5 capacity plans, 15 folds each, as one chain with
  `FT_MODEL_EVAL_STRICT=1` (repo root). Then `restate` each arm.
- [ ] T031 [US3] Implement and run `research/live_pnl/capacity.py`, which refuses while the gate
  is closed or T028 is uncommitted, and withholds any adoption offer while `model_check.json` has
  an open bias flag. It writes `capacity.json` (data-model CapacityPoint), with a
  test for the refusal and the INERT flag in `tests/test_capacity.py`. Add a `report.md`
  "Capacity" section: net P&L per budget per half, the verdicts, and the realised/requested stake.

**Checkpoint**: the capacity answer is ready for the user's decision.

---

## Phase 6: User Story 4 - Shared wallet with the momentum basket (Priority: P4, last)

**Goal**: deployment information on running NNNC and the momentum basket in one wallet.

**Independent Test**: the shared-wallet report lists blocked and shrunk entries per strategy and
the combined P&L against the sum.

- [ ] T032 [US4] Check whether fold-range backtests of `MomentumRegimeBasket15mFast` exist. If
  not, create the plan `evaluation/folds/plan_mom_basket.json` (no_model; the same 15 test ranges;
  config `config_mom_15m.json`, a static 75-pair universe) and run it (one chain).
- [ ] T033 [US4] Replay `cap_aro_ref` and the momentum basket through `evaluation/books.py
  --shared-wallet`, and write `research/live_pnl/shared_wallet.json`. Add a `report.md` "Shared
  wallet (deployment information)" section that changes no default.

---

## Phase 7: Polish & cross-cutting

- [X] T034 [P] Update AGENT_GUIDE.md: the "Current state" section (live NNNC dry-run, where its
  records live, the launcher, never touching the momentum bot) and the feature table row 036.
- [ ] T035 [P] Update `research/model_calibration/findings.md` with a §12 (036) when US2 and US3
  results exist, and the memory status entry.
- [ ] T036 Final check:
  - the full suite passes;
  - the byte check from the Phase 2 checkpoint is repeated;
  - `git status` shows no files from `user_data/data/live_nnnc/` and no sqlite files;
  - spec SC-001 to SC-007 are checked off in `report.md` with evidence.

---

## Dependencies & Execution Order

- **Setup (T001–T002)** → **Foundational (T003–T009)** → **US1 (T010–T018)**.
- **US2** needs US1 to be running plus the gate (T016) open, which is at least 4 weeks after T012.
  T019–T021 and T023's code and tests can be written while recording runs. T022 and T024 wait for
  the gate.
- **US3** needs US2's cost model (T024). T026–T029 can be written early. T030–T031 wait for T024 and T025.
- **US4** runs last (Q1 = B). T032 can run any time no other chain is running.
- **Chains:** T011, T030 and T032 each train or backtest, so they never overlap each other. The
  live bot is not a chain: it does inference only, and it runs throughout.

## Parallel opportunities

- Phase 2: T003, T005, T007 and T008 are different files.
- US1: T013, T015 and T016 can be written in parallel once T012 is running.
- During the recording wait: T019–T021, T023, and T026–T029 (code, tests, pre-registrations).

## Implementation Strategy

**MVP = US1.** Get the live bot recording as early as possible; the 4-week clock is the critical
path. Build the US2 and US3 code during the wait, so that when the gate opens only T022, T024 and
T030–T031 remain. US4 is last, and is deployment information only.
