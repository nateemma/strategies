# Implementation Plan: Realistic P&L for NNNC

**Branch**: `036-realistic-nnnc-pnl` (spec directory; committed to master) | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/036-realistic-nnnc-pnl/spec.md`

## Summary

Run the NNNC control live as a dry-run on a freshly trained model, and record what execution would
really cost. Then use that record to price size and set the stake cap from cost rather than a
chosen constant. The stages:

1. **US1 (start first; it is the long pole).**
   - A live fold `live1` is trained to the latest data, with its threshold picked by the standard
     rule (target rate 0.014735).
   - It runs as a second dry-run bot with its own port, database and config. Strict model mode
     means it never trains live.
   - A default-off recorder mixin writes orders, fills, exits, signals and order-book depth.
   - A daily parity report checks the live trades against a backtest of the same day.
   - The model is retrained every 90 days.
2. **US2 (after the recording gate: ≥ 28 days AND ≥ 5 orders per pair, or day 84).**
   - Execution cost = walking the recorded book against the mid, modelled as a function of
     participation (stake ÷ rolling 24 h quote volume) per depth tier, so it applies to 2023–26
     history. It is validated on held-out days.
   - The 15-fold control and width book are re-stated at net P&L.
3. **US3.**
   - A default-off cost-budget stake cap replaces the 1/10-candle-volume rule in the capacity arms.
   - A capacity curve at 5/10/20/40 bps (pre-registered) is run on the existing weights and judged
     on net P&L with judge_one and the risk verdict.
4. **US4 (last).** The momentum basket's fold backtests are replayed with the control in one wallet
   (`books.py --shared-wallet`), as deployment information only.

**Technical approach:**
- No Framework/ change. The two mixins live in utils/ and are off by default and key-neutral.
- Study code lives in `research/live_pnl/`.
- The launcher is `scripts/live_nnnc.sh`.
- Recorded market data goes to `user_data/data/live_nnnc/`, outside the repository.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`); zsh scripts.

**Primary Dependencies**:
- freqtrade (in-tree): `trade --dry-run`, the strategy callbacks `confirm_trade_entry`,
  `order_filled`, `confirm_trade_exit`, `bot_loop_start` and `custom_stake_amount`, and
  `dp.orderbook`;
- pandas and numpy;
- the existing harness: `evaluation/run_folds.py`, `pick_threshold.py`, `risk.py`, `books.py`,
  `research/nnnc_switches/judge.py`, and `Framework/ModelKey.py`.

**Storage**:
- append-only JSONL (depth gzipped) under `user_data/data/live_nnnc/<run_id>/`, not committed;
- the freqtrade sqlite `tradesv3.live_nnnc.sqlite`, at the repo root and not committed;
- committed JSON outputs in `research/live_pnl/`.

**Testing**: pytest (`research/live_pnl/tests`, plus the existing suites). Tests cover:
- mixin pass-through, byte identity and key neutrality;
- dedupe across restart;
- walking the book;
- curve monotonicity and inversion;
- parity cause classification;
- the gate rule;
- launcher refusals (port, database path, already running).

**Target Platform**: macOS (Apple silicon), the same machine as the momentum dry-run.

**Project Type**: a research harness plus a long-running dry-run process.

**Performance Goals**:
- the recorder adds less than 1 s per 15-minute candle per pair;
- depth requests are capped at 3 s and never retried in the loop.

**Constraints**:
- the momentum bot (pid 2661, port 8080, `tradesv3.dryrun.sqlite`) is never touched;
- one trainer or backtest chain at a time;
- no pair-specific code;
- default off and byte-identical when off;
- recorded data is not committed and is parsed, never executed.

**Scale/Scope**:
- 10 pairs;
- about 1.5 orders a day;
- about 960 scheduled snapshots a day (about 1–2 MB gzipped);
- a recording period of at least 4 weeks;
- 15 folds × 5 capacity arms, on existing weights with no training.

## Constitution Check

*GATE: checked before Phase 0 and re-checked after Phase 1.*

| principle | status |
|---|---|
| I. Deployment regime | PASS. The live record is the deployment regime itself, and the capacity curve reports early and late halves. |
| II. Distant era (NON-NEGOTIABLE) | **Justified exception, recorded.** The 15 folds are one era, and the depth record is one recent period. Results are labelled PROVISIONAL (FR-020), and nothing is promoted without the user. The cost model is an execution measurement, not an edge claim, and its 2023–26 use is labelled EXTRAPOLATED if the participation fit fails (R5). |
| III. Learnability ≠ edge | PASS. Every verdict is on net P&L. |
| IV. Parameters live | PASS. The capacity arms need a liveness check (realised stakes differ, R7); a parity report proves the live class matches its backtest. |
| V. Realistic execution | This feature's purpose. Cost comes from recorded depth, not from dry-run fills. |
| VI. Diagnose before tuning | PASS. The recording and parity diagnostics come before any cap change. |
| VII. Validate cheaply | PASS. US2 and US3 are gated on the recording; the capacity arms need no training. |
| VIII. One space at a time | PASS. Only the stake cap varies. |
| IX. Orthogonality | Not applicable (no new factor). US4 measures contention only. |
| X. No per-pair overrides | PASS. Cost tiers are terciles of a structural measure, and refusals cover missing tiers. |
| XI. No leakage | The live fold trains only before the cut and reuses the scaler fitted on 2026-09-10, before any live bar (R12). The known walk-forward scaler exposure is unchanged and stays recorded. |
| Repository constraints | PASS. No Framework/ change, mixins in utils/, params merge, staging by name. |
| Quality gates | Pinned ranges (fold plans), trade counts reported, liveness proof (R7). Scalers unchanged, since the include list is unchanged. |

**Post-design re-check:** PASS. Same table, with no new exceptions. The spec's SC-007 band was
amended to the BUY-rate form (R11); it is a measurement definition, not a principle conflict.

## Project Structure

### Documentation (this feature)

```text
specs/036-realistic-nnnc-pnl/
├── plan.md
├── research.md          # R1–R12
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── execution_recorder.md
│   ├── cost_budget_stake.md
│   └── study_cli.md
└── tasks.md             # /speckit-tasks
```

### Source Code (user_data/strategies)

```text
utils/
├── ExecutionRecorder.py          # ExecutionRecorderMixin (default off, key-neutral)
└── CostBudgetStakeMixin.py       # cost-budget stake cap (default off)
NNNC/
├── NNNC_LIVE_live1.py/.json      # GENERATED by run_folds.generate_fold (live fold)
├── NNNC_LIVE_live1_REC.py        # recorder mixin + live fold class (the bot's strategy)
└── NNNC_CAP_ARO_B<x>.py/.json    # control + CostBudgetStakeMixin (capacity arms)
config/
└── config_live_nnnc.json         # dry-run, port 8081, own bot_name; whitelist/stakes from config_wf
scripts/
└── live_nnnc.sh                  # start|stop|status, pid file, refusals
evaluation/folds/
├── plan_live_aro.json
└── plan_cap_aro_{ref,b5,b10,b20,b40}.json
research/live_pnl/
├── live_fold.py  parity.py  drift.py  gate.py
├── depth.py      cost_model.py  restate.py  capacity.py
├── preregistration_capacity.json   # budgets + rules, committed before any capacity run
├── live_models.json  cost_model.json  gate.json  capacity.json  restated_*.json
├── parity/                         # weekly summaries committed
├── report.md
└── tests/
```

**Structure Decision**: follow 032–035: study code in `research/<study>/`, plans in
`evaluation/folds/`, and arm classes in `NNNC/`. The new reusable runtime pieces (the recorder and
the cost cap) are mixins in `utils/`, keeping Framework/ strategy-agnostic and untouched.

## Complexity Tracking

| item | why needed | simpler alternative rejected because |
|---|---|---|
| a second long-running bot | live depth and parity can only be measured live | dry-run fills are simulated, and a backtest cannot see depth |
| participation-based cost curves | apply recorded cost to 2023–26 history, where no depth exists | a fixed bps cost cannot price size, which is the question |
| a recording gate (4 weeks + 5 orders per pair) | the user's choice (Q2 = C); guards against a thin cost model | — |
