# 036 Realistic P&L for NNNC: report

Spec: `specs/036-realistic-nnnc-pnl/`. **Label: PROVISIONAL.**
- Fold-based outputs (re-statement, capacity) are execution measurements, not edge findings. They
  run on fold models that inherit the full-history scaler (FR-020; plan.md records the Principle XI
  exception).
- Promotion of anything is the user's decision.

## Setup (2026-10-16)

- **Recorded live data** goes to `user_data/data/live_nnnc/`. That path is outside the nested
  strategies repository, and the outer upstream checkout ignores it (`.gitignore: user_data/*`).
  It is never committed (FR-008).
- **The live bot's database**, `tradesv3.live_nnnc.sqlite`, is at the repo root and ignored there
  (`.gitignore: *.sqlite`). It is not in the strategies repository.
- **The momentum dry-run** (pid 2661, port 8080, `tradesv3.dryrun.sqlite`,
  `config_mom_dryrun_15m.json`) is out of bounds for this feature.
- **Tests:** AGENT_GUIDE's suite line uses the generic `research/<study>/tests`, which covers
  `research/live_pnl/tests`, so no edit was needed.

## Phase 2 checkpoint (2026-10-16)

- **Tests:** the full suite passes (1122 passed, 9 skipped), including 19 new `research/live_pnl`
  tests.
- **Existing arms:** byte identity holds by construction. Phase 2 adds only new files
  (`utils/ExecutionRecorder.py`, `research/live_pnl/*`, `config/config_live_nnnc.json`,
  `scripts/live_nnnc.sh`) and edits no existing code, so no existing class imports or inherits
  anything new. The recorder's backtest pass-through and key neutrality are tested directly
  (`test_recorder.py`).
- **Launcher:** `live_nnnc.sh --check` correctly refuses before a live model exists.

## Live model live1 (T010–T011, 2026-10-16)

- **Data:** 15m candles for the 10 pairs were downloaded at 08:40; the last candle is 2026-10-16
  15:15 UTC on every pair.
- **Training:** `live_fold --cut 20261016 --run-id live1` trained `NNNC_LIVE_live1` on
  20210101-20261016 (seed 1). Validation MCC is 0.632.
- **Threshold:** 0.7783, picked on the training window at target rate 0.014735 (achieved 1.4736%
  over 1,837,601 bars). It is inside the controls' range (0.734–0.783).
- **Model key:** `1cb71608d8ba4ef1` under `config_live_nnnc.json`, and the `_REC` class resolves
  to the same key. `scripts/model_index.py key` reports a different, absent key because it uses
  the default `config.json`, whose whitelist includes SUI. The bot and the launcher use the live
  config.
- **FR-002, the current defaults:** `HELD_PAIR_ADD = {'in_profit_only': True}`;
  `stoploss_grace_hours` 24.0 (class and fold json); grace level −0.27.
- **Launcher check:** `live_nnnc.sh --check` passes.

## Live bot started (T012, 2026-10-16)

- **Start:** `live_nnnc.sh start` at 15:50 UTC, then a restart at **15:56 UTC**, which is the
  effective start and the FR-019 clock. It runs as pid 41849 on port 8081, with database
  `tradesv3.live_nnnc.sqlite` and log `user_data/logs/live_nnnc.log`.
- **Momentum bot:** untouched. It is still pid 2661, started 2026-09-10 14:06, and port 8080
  responds 200 before and after.
- **Incident at the first start:** every pair's `analyze_ticker` raised `ValueError('cannot convert
  float NaN to integer')`, so there were no signals and no trading for 6 minutes.
  - **Cause:** the recorder converted the last row's `enter_long` (NaN on non-signal candles) with
    `int()`, and its error guard re-raised every `ValueError`. The re-raise was meant only for the
    RECORD_DIR refusal. A recording failure therefore broke the strategy, contrary to the contract.
  - **Fix:** a dedicated `RecorderConfigError` is the only error that escapes; values are converted
    NaN-safe. Two regression tests were added. The bot was restarted.
- **After the fix**, by 16:02 UTC:
  - 20 scheduled depth snapshots (2 candles × 10 pairs), 0 missing, 20 levels per side;
  - one signal row per pair per candle with real p_buy values;
  - no strategy exceptions.
- **Effective threshold:** `prediction_threshold` is a 2-decimal `DecimalParameter`, so the picked
  0.7783 is applied as **0.78**. Every walk-forward control ran with the same rounding (0.7732
  became 0.77), so live matches how the controls were backtested. `drift.py` measures against the
  logged, applied threshold.
- **Model-key audit:** once the bot class existed on disk, `confirm_trade_exit` had a second
  definition, and the audit requires every multiply-defined method to be classified.
  `confirm_trade_exit` and `_fillable_stake` were added to BaseStrategy's
  `MODEL_KEY_METHODS_IGNORED`, and both mixins declare their overridden callbacks as ignored.
  That list is not hashed, so every model key is unchanged (the pre-034 key fixture test passes).
  This is the only edit to existing Framework code in 036, and it is classification only, with no
  behaviour change.

## Daily duties scheduled (T017, 2026-10-16)

- **Cron:** `40 17 * * *` local time (00:40 UTC in summer, 01:40 UTC in winter, so always after the
  UTC day has closed) runs `scripts/live_pnl_daily.sh`. It runs parity for yesterday, then drift,
  then gate, and logs to `user_data/logs/live_pnl_daily.log`. It commits nothing; weekly
  summaries are committed by hand.
- **Smoke run on the partial start day** exposed a parity defect: the cumulative backtest starts
  at the start day's 00:00 UTC, so a backtest trade at 03:30 UTC, before the bot existed, was
  reported UNEXPLAINED. Parity now compares only trades opened at or after `started`, with a
  regression test. The re-run had 0 differences. The smoke file was deleted, because the start
  day is partial.
- **A failed day** (e.g. a download refused while a chain runs) is logged and can be re-run with
  `zsh scripts/live_pnl_daily.sh YYYY-MM-DD`.

## US4: NNNC and the momentum basket in one wallet (T032–T033, 2026-10-16)

**DEPLOYMENT INFORMATION ONLY.** It changes no NNNC default. It is PROVISIONAL: one era, an
approximate replay, and fold models that inherit the full-history scaler.

**Runs (one chain, no training):**
- `cap_aro_ref`: the NNNC control with current defaults, on the existing fold weights, 15/15 folds.
  It sums to 56.47% of wallet over the folds; `wf_aro_g24`, with the held-pair add off, made 49.97.
  It is also the US3 reference.
- `mom_basket`: `MomentumRegimeBasket15mFast` on the same 15 test windows (static 75-pair universe,
  `config_mom_15m.json`, model-free, via `regime.run_arm`). Fold f12 made 0 trades (risk-off).

**Replay** (`research/live_pnl/shared_wallet.py`): every recorded fill of both books goes through
one 10,000 USDT wallet per fold, in time order.
- An order is cut to the free cash, and refused if that leaves it under 10 USDT.
- Exits return their proceeds, so the wallet compounds as the basket's does.
- A trade's P&L scales with its filled share.
- **Control checks:** each book alone reproduces its own backtest exactly, with 0 refused (NNNC
  5,646.6 = 5,646.6; basket 65,913 = 65,913).
- **Two replay defects were found and fixed before these numbers,** each with a regression test:
  - exits returned only the cost basis, so the compounding basket was starved;
  - the 10 USDT minimum refused the basket's genuinely small top-up orders (35% of its orders are
    under 10 USDT).

| sum of folds (USDT) | NNNC alone | NNNC shared | basket alone | basket shared | both, 2 wallets | both, 1 wallet |
|---|---|---|---|---|---|---|
| early f00–f07 | 3,847 | 2,089 | 46,965 | 46,621 | 50,812 | 48,710 |
| late f08–f14 | 1,799 | 1,389 | 18,948 | 18,724 | 20,747 | 20,113 |
| all | 5,647 | 3,477 | 65,913 | 65,344 | 71,559 | 68,821 |

**Reading:**
- **The basket crowds NNNC out; NNNC barely touches the basket.**
  - Shared, NNNC loses 38% of its P&L (2,170 USDT): 214 of 1,237 entry orders were refused and 154
    shrunk, because the basket is fully invested whenever it is risk-on.
  - The basket loses 0.9%.
  - The loss is larger early (−46%) than late (−23%).
- **Per unit of capital, one wallet is far better:** 68,821 on 10,000, against 71,559 on 20,000
  for two wallets. The one wallet keeps 96% of the two-wallet P&L with half the capital.
- **Priority barely matters.** With the basket first on same-time clashes, NNNC gets 3,439
  instead of 3,477.
- **The scale of the basket's returns** (+294% in f04; summed +659%) comes with large drawdowns,
  up to 63.6% in f05. Those risk numbers are the basket's own and are unchanged by sharing.
- **Approximations:**
  - a refused NNNC entry would have left its pair free for a later signal the archive does not
    hold;
  - shrinks are spread evenly over a trade's lots;
  - fees on scaled fills are taken as proportional.
