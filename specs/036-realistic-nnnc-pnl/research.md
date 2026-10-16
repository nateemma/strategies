# Research: Realistic P&L for NNNC (036)

Facts were checked in the repo on 2026-10-16. Each decision names the code it builds on.

## R1. The live model: one generated "live fold", trained and picked like a walk-forward fold

**Decision**: a plan `evaluation/folds/plan_live_aro.json` with one fold, `live1`. Its fields:
- `train_timerange` = `20210101-<cut>`, where `<cut>` is the first day not yet downloaded when
  training starts;
- base `NNNC_A_Raw_Open`;
- the control's params snapshot (`params_20261013-g24_NNNC_A_Raw_Open.json`);
- `test_timerange` = `<cut>-<cut+90d>`, which is the live period;
- seed 1.

The steps:
- `run_folds.generate_fold` writes `NNNC/NNNC_LIVE_live1.py`/`.json`. The class pins
  `MODEL_TRAIN_TIMERANGE` and the threshold.
- `run_folds.train_fold` trains it, with one trainer at a time.
- The threshold is picked by `evaluation/pick_threshold.py --target-rate 0.014735` on the
  training window. This is the rule that produced every control threshold
  (`thresholds_NNNC_A_Raw_Open.json`; specs/032 research R6). The pick uses
  `run_folds.fold_has_model`, not the old class-name check (the known 032 fix).

**Rationale**: the live model then has exactly the identity of a walk-forward fold, so it can be
compared with the folds, and the parity backtest (R4) is just this class on the day's range.
Feature 030 keys resolve the weights. A missing key in strict mode raises `ModelMissing` and never
trains.

**Alternatives**:
- Reusing f14's weights: rejected. The forward window showed drift at 3–4 months.
- A hand-written live class: rejected. It could drift from the fold machinery.

**Retrain** (FR-004): every 90 days, with a new fold `live2` and so on, and the cut moved forward.
The `.py`/`.json` of each retired live fold stay in the repo for attribution.

## R2. Isolation of the live bot

**Decision**: a launcher `scripts/live_nnnc.sh` (zsh, repo root) that sources `~/.freqtrade_env`,
exports `FT_MODEL_EVAL_STRICT=1`, and runs `freqtrade trade --dry-run` with:
- its own config, `config/config_live_nnnc.json`;
- its own database `sqlite:///tradesv3.live_nnnc.sqlite`;
- its own log file;
- `--strategy-path user_data/strategies/NNNC`;
- strategy `NNNC_LIVE_live1_REC` (R3).

**The config:**
- **Copied from `config_wf.json`:** the 10-pair static whitelist, stake 900, max_open 10, wallet
  10,000 and 15m.
- **Copied from `config_mom_dryrun_15m.json`:** pricing, order types and timeouts, so live pricing
  matches the running dry-run.
- **Its own:** `api_server.listen_port` 8081 and `bot_name` "nnnc_live".

**Safety:** the launcher refuses to start if:
- port 8081 is taken;
- the database path equals the momentum bot's;
- a process with the same config is already running.

It never signals any other process: no pkill, and it finds its own pid from a pid file.

**Rationale**: FR-001. The momentum bot (pid 2661) uses port 8080 and `tradesv3.dryrun.sqlite`.
Live runmode already turns off pair aggregation and training (`BaseNNStrategy.populate_indicators`:
`runmode != backtest` → `aggregate_pairs = training_needed = False`). Strict mode makes a missing
model fatal.

## R3. Recording orders, depth and signals: a default-off mixin, no Framework/ change

**Decision**: `utils/ExecutionRecorder.py`, `ExecutionRecorderMixin`, which is strategy-agnostic.
It is used by subclassing: `NNNC/NNNC_LIVE_live1_REC.py` = mixin + the generated live class. It
overrides these freqtrade callbacks, always calling `super()` first and returning the parent's
value unchanged:

| callback | records |
|---|---|
| `confirm_trade_entry` | order intent (requested stake from `custom_stake_amount`, granted amount × rate), signal candle and price, plus a depth snapshot |
| `order_filled` | fill time and price, order id |
| `confirm_trade_exit` | exit reason and rate, plus a depth snapshot |
| `bot_loop_start` | a scheduled snapshot per pair once per 15-minute candle, plus housekeeping |
| `populate_entry_trend` (after super) | the signal log for the last closed candle per pair: p_buy, threshold, guard pass, entry flag |

- **Active only when** `RECORD_EXECUTION = True` AND `dp.runmode` is live or dry_run. Otherwise
  every method is a pure pass-through, so backtests stay byte-identical (FR-014).
- **Depth:** `self.dp.orderbook(pair, maximum=20)`, wrapped in a timeout. A failure writes a
  record with `depth_missing = true`.
- **Model key:** `RECORD_EXECUTION` and `RECORD_DIR` go into `MODEL_KEY_IGNORED`, and the
  overridden callbacks are not weight-shaping (`MODEL_KEY_METHODS` does not list them). So the
  `_REC` class resolves to the live fold's key; this is checked by a test.
- **Storage:** append-only daily JSONL files under `user_data/data/live_nnnc/<run_id>/`. That is
  outside the strategies repository, so it is never committed (FR-008). The files are `orders`,
  `fills`, `exits`, `depth` (gzip) and `signals`.
- **Dedupe key:** (kind, trade_id, order_id) for orders and fills, and (pair, candle) for signals
  and scheduled depth. On start the recorder reads the day's file and skips existing keys, so a
  restart cannot double-write.

**Rationale**: freqtrade exposes all four callbacks (`freqtrade/strategy/interface.py`), and
`dp.orderbook` works in dry-run. A mixin in utils/ keeps Framework/ strategy-agnostic, and keeps
NNNC's base and the control's key untouched.

**Rate budget:** one order-book request per pair per 15 minutes (10 pairs), plus about 2 per trade.
That is far below Binance.US limits.

## R4. Daily parity report

**Decision**: `research/live_pnl/parity.py --day YYYY-MM-DD`. It:
1. downloads the latest 15m candles for the 10 pairs into the normal data directory;
2. backtests the live class from the live run's start (`started` in live_models.json) through the
   end of the day, with startup candles and `FT_MODEL_EVAL_STRICT=1`. Holding positions over
   several days (24 h+ grace) and one trade per pair make a single-day backtest start from the
   wrong open-trade state, so the backtest is cumulative and only the day's entries and exits are
   compared;
3. reads the live trades from `tradesv3.live_nnnc.sqlite` (read-only) and the signal log;
4. matches trades on (pair, open candle);
5. classifies each difference by a declared order of causes:
   - (p) prediction difference, checked first: the live p_buy in the signal log differs from the
     backtest's p_buy for the same pair and candle by more than 0.01. Live predicts pair by pair,
     while the backtest uses the pair-aggregated, sequential-index path, so a pipeline difference
     must be separated from market causes;
   - (a) data revision: the downloaded candle differs from the one the live bot saw (the signal
     log stores the live close and volume);
   - (b) gap-fill: a zero-volume bar on either side;
   - (c) timing: an entry within one candle, or an unfilled limit order;
   - (d) wallet or slot state: a different open-trade set, including a position carried from a
     difference on an earlier day;
   - (e) UNEXPLAINED, which is flagged.

Output goes to `research/live_pnl/parity/<day>.json` (committed weekly as a summary, not daily
files).

**Rationale**: FR-007 and SC-002. The signal log is what makes cause (a) checkable instead of
guessed.

## R5. Execution cost: walk the book against the mid, then express it as participation

**Decision**: for a stake S (quote) on a buy, cost(S) = VWAP of asks consumed for S ÷ mid − 1, in
bps. For a sell, it is the same against the bids.

**Why the mid:** the backtest fills at candle prices, which sit near the mid, not the touch. So
the realistic extra cost of size is the half-spread plus walking the book. Fees (0.10% per side)
stay as the backtest already charges them, and are not counted twice.

**Making it usable on 2023–26 history** (where no depth exists), the cost is modelled as a
function of participation p = S ÷ (rolling 24 h quote volume of that pair):
- each snapshot gives cost at a grid of stakes (100…5,000 USDT);
- for each depth tier, a monotone curve cost = f(p) is fitted;
- tiers are terciles of a pair's median top-20 quote depth over the recording period (a structural
  rule, Principle X).

**Validation** (FR-010): hold out the last 25% of recorded days, and report the per-pair median
absolute error against walking the held-out books.

**Fallback (declared before fitting):** if participation explains less of the held-out cost than a
per-pair constant curve, the per-pair curve is used for the live period only. The 2023–26
re-statement is then labelled EXTRAPOLATED.

**Rationale**: the backtest history has candle volume but no depth. Depth is expected to scale
with traded volume, and the record tests that rather than assuming it.

**Alternatives**: a fixed bps cost; this cannot price size, which is the whole question.

## R6. Re-statement of existing backtests (US2)

**Decision**: `research/live_pnl/restate.py --arm <tag> [--folds ...]`. It:
- reads each fold's trades from the arm record's result zip;
- for each trade, computes p at its entry and exit candles from the stored OHLCV (rolling 24 h
  quote volume up to and including the signal candle, which is causal);
- applies cost_in + cost_out to the trade's stake;
- reports gross, cost and net per fold, per half and in total.

Trade identity is unchanged. It is run on `wf_aro_g24` and `w_aro` for the baselines (FR-012).

**The approximation:** compounding is ignored, because the stake is a fixed 900 capped per trade,
not a wallet fraction.

## R7. Cost-budget cap (US3)

**Decision**: an option on `BaseStrategy.custom_stake_amount`'s existing fillable-stake path. A
class attribute `STAKE_COST_BUDGET_BPS = None` (off) and `STAKE_COST_MODEL = None` (path) live in a
mixin `utils/CostBudgetStakeMixin.py`, which NNNC fold classes inherit only in the capacity arms.

When it is on:
- stake = min(proposed, S_max), where S_max is the largest S with cost_in(S) ≤ budget, using the
  candle's rolling quote volume and the pair's tier from the committed cost model;
- `confirm_trade_entry`'s headroom check uses the same S_max instead of 1/10 of the candle volume;
- the MIN_QUOTE_VOLUME floor and min-stake checks are kept.

The requested and granted stakes are logged into the arm record's per-trade data, as
`custom_data`/tag, so realised against requested can be measured.

**Liveness proof** (Principle IV): the realised stake distribution must differ from the current
cap's on at least one budget. Otherwise the arm is INERT.

**Rationale**: FR-013/014. The cap is the existing hook (`Framework/BaseStrategy.py:1480`). The
mixin keeps the change out of Framework/ and off by default.

**Alternatives**: changing `QUOTE_VOLUME_HEADROOM_MULT`. Rejected: that is the sizing dial known to
inflate backtest returns without a cost model (memory `project_liquidity_and_capacity`).

## R8. Capacity curve and judging

**Decision**:
- **Plans** `plan_cap_aro_b{5,10,20,40}.json`: copies of `plan_wf_aro_g24.json` with base
  `NNNC_CAP_ARO_B<x>` (control + the cost-budget mixin), the same thresholds, and the same weights.
  The model key must be unchanged, so the weights are reused and nothing trains.
- **The current-cap reference** is `wf_aro_g24`, re-run now with current defaults. The old records
  predate the held-pair add, so it is re-run as `cap_aro_ref` for a like-for-like comparison.
- **Every arm** is re-stated with R6 for net P&L.
- **Judging:** `research.nnnc_switches.judge.judge_one` on net per-fold P&L (paired early and
  late, risk not WORSENED via `evaluation/risk.py`). The trade count and the realised/requested
  stake ratio are reported.

The budget set {5, 10, 20, 40} bps is committed in a pre-registration before any capacity
backtest.

**Two baselines (analyze I2):** `restate` runs on the old `wf_aro_g24` records (held-pair add
off, the historical comparator) AND on `cap_aro_ref` (current defaults). The capacity verdicts use
`cap_aro_ref` only, and `report.md` labels each.

**Gate 5 (paired seed robustness):** met by determinism. The capacity arms use fixed weights, and
backtests are deterministic, so there is no run-to-run variance to isolate. This is stated in the
capacity pre-registration.

## R9. Shared wallet with the momentum basket (US4, last)

**Decision**: backtest `MomentumRegimeBasket15mFast` on the same 15 fold test ranges. It is
model-free. The static universe `config_mom_15m.json` is used, because the dry-run's dynamic
VolumePairList cannot be backtested. Then replay it with the `cap_aro_ref` trades through
`evaluation/books.py --shared-wallet` (026), which already reports blocked entries and the
combined P&L against the sum. It is reported as deployment information only.

**Open fact to check at task time:** whether the momentum basket's fold-range backtests exist.
None are among the `regime/arms` records found. If they don't exist, they are new model-free
runs.

## R10. Recording period gate (FR-019, Q2 = C)

**Decision**: `research/live_pnl/gate.py` reads the order records and reports days elapsed and
orders per pair.
- **OPEN** when days ≥ 28 AND every pair has ≥ 5 orders.
- **At day 84,** the pairs still under 5 are listed as thin and handled by the tier pooling (R5),
  and the gate opens.

US2 and US3 judging scripts refuse while the gate is closed.

## R11. Operating-point drift (SC-007, amended)

**Amendment to SC-007:** for the control there is no 0.5 share. The operating point is the live
BUY rate: the share of the signal log's candles with p_buy > threshold, against the target rate
0.014735 the threshold was picked for. It is flagged weekly if outside ×0.5–×2 of target.

**Rationale**: the 0.3–0.7 band in the spec was written for the sizer's 0.5 share. The control's
threshold is picked to a rate, so drift is measured against that rate.

## R12. Scaler (Principle XI)

**Decision**: the live fold reuses `saved_data/main_scaler.pkl` unchanged. It was fitted on
2026-09-10, before any live bar. The model key then matches the validated pipeline (the key hashes
the scaler file). Re-fitting would change the key and the model, which is out of scope (FR-018).
The walk-forward exposure in Principle XI is unchanged by this feature and stays recorded.

## R13. Live record against the cost model (FR-016)

**Decision**: `research/live_pnl/model_check.py`, run every 4 weeks once the cost model exists. For
each recorded order and exit with a depth snapshot, it compares:
- the cost the model predicts from participation at that candle;
- the cost of walking that snapshot for the granted stake.

It reports the median signed error per tier. A sustained bias is |median| > the tolerance on two
consecutive checks. It is flagged in `model_check.json`, and `capacity` refuses to offer a result
for adoption while a flag is open.
