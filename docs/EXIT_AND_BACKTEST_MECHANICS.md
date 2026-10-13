# Exits, Entries and freqtrade Backtest Mechanics — What Has Been Verified

Reference for anyone (agent or human) changing exit, stop or entry logic, or building an offline
model of what a backtest does. Every item here was verified against freqtrade's source, this repo's
code, or a measurement. Each one cost real time when it was unknown.

Scope and sources:
- freqtrade's pinned in-tree checkout: `freqtrade/optimize/backtesting.py`,
  `freqtrade/strategy/interface.py`, `freqtrade/persistence/trade_model.py`;
- this repo: `Framework/BaseStrategy.py`, `utils/ExitPolicy.py`, `utils/FamilyExits.py`,
  `utils/StopInvariants.py`;
- specs 021–027.

Line numbers drift after an upgrade, so re-verify before relying on a line. Results marked
PROVISIONAL rest on halves of 2023–2026 only (quality gate 2, no distant era).

---

## 1. freqtrade backtest mechanics

### 1.1 What happens on each candle of an open trade

`IStrategy.should_exit` (interface.py, ~1419) runs once per 15m candle `k`, with
`rate = open[k]`. Steps 1–2 happen first, then the exits are checked in this order:

1. **`trade.adjust_min_max_rates(high, low)` runs first.** So `trade.max_rate` already includes
   the current candle's high when your callbacks run. Do not key a "target reached" rule on
   `max_rate`: the same candle's low could then fire a stop placed from its own high, and the
   order of high and low within the candle is unknown. Use closed candles (027 R4).
2. **The stop is adjusted** (`ft_stoploss_adjust`), then checked against the candle's low.
3. **Exits are collected in a fixed order: custom exit / exit signal → stop → ROI.**
   `_check_trade_exit` takes the FIRST one that fills. A `custom_exit` decision therefore wins a
   same-candle tie against the stop, and the stop wins against ROI.

### 1.2 What the strategy sees

- **Callbacks see the last CLOSED candle, `k-1`.** The backtest calls
  `dp._set_dataframe_max_index(required_startup + row_index)`, so `get_analyzed_dataframe` ends at
  the candle before the one being simulated. `custom_exit`'s `last_candle`, the ATR and rvol that
  `custom_stoploss` reads, and `guard_metric` all come from `k-1`.
- **Fill prices:**
  - **Custom exits** (take-profit, unclog, max_hold, metric_overbought) evaluate profit at the
    OPEN of `k` and fill there.
  - **The stop** triggers on the LOW. It fills at the stop price, or at the OPEN when the stop is
    above the candle's high (a gap) (`_get_close_rate_for_stoploss`).
  - **ROI** is detected on the HIGH (`current_profit_best`) and fills at the ROI target price
    (`_get_close_rate_for_roi`), not at the high.
- **A candle's entries are evaluated before its exits,** and a trade can exit on its entry candle.
- **A trade open at the end of the backtest is force-exited at the last candle's close.** Its
  `close_date` is stamped ONE BAR LATER than that candle.

### 1.3 How `custom_stoploss` is called in a backtest

- **`current_rate` is the candle's HIGH, not the open:** `current_rate=(bound or current_rate)`
  with `bound = high` for a long (interface.py `ft_stoploss_adjust`). A stop computed relative
  to `current_rate`, or clamped "at market", is placed relative to the high.
- **It is only called while `trade.stop_loss < low`** (`dir_correct`). Once a candle's low is
  through the stop, the stop is NOT adjusted on that candle, and the existing stop fires.
- **The returned ratio is applied as `new_loss = current_rate * (1 - abs(ratio))`**
  (trade_model.py, `adjust_stop_loss`):
  - a POSITIVE ratio becomes a LOOSER stop (the 2026-10-08 bug: an intended −4% became −46%);
  - `0.0` is falsy and silently ignored;
  - the fix is `ExitPolicy.STOP_AT_MARKET_RATIO` (−1e-6), which fires now.
- **The stop price is rounded UP to the pair's price precision** for a long
  (`price_to_precision(..., ROUND_UP)`). On coarse-tick pairs (ZEC trades in 0.1 steps) this
  decides whether a flat candle's low triggers the stop. A −2% stop of 41.454 becomes 41.5.
- **freqtrade only tightens a stop after entry.** The exception is the `after_fill` call at
  entry, which may set it wider (`allow_refresh`). That is how the grace window gets a −27% stop.
- **`initial_stop_loss_abs` in the archive comes from `strategy.stoploss`, before `after_fill`.**
  It is cosmetic when a custom stop is used.

### 1.4 ROI and entry signals

- **With `ignore_roi_if_entry_signal = True`** (SimpleTrigger presets and BaseStrategy), any candle
  with `enter_long == 1` on that pair suppresses ROI for the trade already open on it.
- **This happens even when the entry would later be refused.** Adding signal rows as a control
  mechanism therefore changes exits (026: a "+late" signal copy broke a null check).
- **Suppression is per candle, not per trade.** An offline replay that cannot recompute entry
  signals should block ROI on the candles of the recorded life where ROI would have fired but the
  trade stayed open.

### 1.5 Positions, sizing, liquidity

- **One open position per pair per bot.** In a single bot running several trigger families, an
  open trade of one family blocks the other family's entries on that pair. That pair-level
  blocking, not exits, was the main interaction between families (024/025); 026 showed it can
  also re-sequence which family gets each pair.
- **`BaseStrategy.custom_stake_amount` caps a stake** at the last candle's quote volume / 10
  instead of refusing the trade.
  - About 75% of SimpleTrigger entries are filled BELOW the 900 stake (median about 310).
  - Full fills read 899.0–899.9999, because of amount precision, not exactly 900.
  - A thin candle gives stake 0, and the entry is skipped WITHOUT `confirm_trade_entry` being
    called.
  - `MIN_QUOTE_VOLUME` is 1000.
- **freqtrade shrinks a stake to the available balance** rather than refusing it
  (`Wallets.validate_stake_amount`). It refuses only below the minimum order.
- **`confirm_trade_exit` vetoes DISCRETIONARY exits in a trade's first hour.** Stops are exempt,
  ROI is not (`ExitPolicy.discretionary_exit_allowed`). A vetoed exit is skipped, and freqtrade
  falls through to the next exit on the same candle (often the stop).

### 1.6 Position adjustment (adding to a held trade; feature 031)

Read from `backtesting.py` `backtest_loop` / `_check_adjust_trade_for_candle` and
`Order.close_bt_order`.

- **Only one trade per pair.** A fresh entry signal on a held pair is ignored, unless
  `position_adjustment_enable` is on.
- **When the add is checked.** On each candle, inside `_check_trade_exit` and BEFORE that candle's
  exit checks, freqtrade calls `adjust_trade_position(current_time = candle date,
  current_rate = open, max_stake = min(exchange max, available wallet))`. It is called on the entry
  candle too (age 0), so require a minimum age.
- **How it fills.** A positive return fills at once at the candle's open, the same timing as a
  fresh entry. It needs no slot. `max_entry_position_adjustment` caps how many adds a trade can
  take.
- **What happens after the fill.**
  - `open_rate` becomes the amount-weighted average (exported rounded to price precision);
  - `stake_amount` becomes the total;
  - the stop is re-anchored to the new `open_rate` with the existing stop percentage;
  - `custom_stoploss(after_fill=True)` is called, so the stop may move either way.
- **freqtrade skips `custom_stake_amount` and `confirm_trade_entry` for an add.** Any liquidity cap
  or volume floor must be applied inside `adjust_trade_position` (031 does this via
  `BaseStrategy._fillable_stake`).
- **Reading the signal.** Inside the callback, `dp.get_analyzed_dataframe` ends at the SIGNAL candle
  (`current_time − timeframe`), with signals unshifted. 031 asserts this on every call
  (`row_misaligned` counter: 0 in every run).
- **Age-based rules.** `trade.open_date_utc` never moves. Code that should restart at an add must
  read `trade.date_last_filled_utc`. In this repo it goes through `trade_clock_start(trade)`,
  which does so only when position adjustment is on.
- **Harness accounting.**
  - `trades_hash` is unaffected: the trade keeps its open date, and its profit changes.
  - Pairing keys on the original `open_date`.
  - Records gain `n_adds` / `hp_counters` only when adds happen.

---

## 2. This repo's exit stack (BaseStrategy / ExitPolicy)

`utils/ExitPolicy.py` is the single source of truth for the stop arithmetic and the custom-exit
decision. `BaseStrategy` only supplies inputs to it.

### 2.1 The stop

- **Grace window.** `stoploss_grace_hours` / `_level`. NNNC default since 2026-10-13: 24 h at
  −27% (was 6.5 h; adopted on four models by research/nnnc_framework/, post-grace stop exits
  roughly halved). The SimpleTrigger preset ARM_D_OPEN still uses 6.5 h; 24 h is untested there.
  - It is set at `after_fill`. During the window `custom_stoploss` returns 1.0 (no change).
  - Most winners reach a +2% target inside it: the median is 9 bars, and the stop in force at
    the target touch is the grace stop (027).
- **After grace: the entry-anchored ATR stop.**
  - The stop is `-K × atr_pct_roll` (`atr_stop_fraction`), clamped to `[atr_stoploss_floor,
    atr_stoploss_cap]` (ARM_D_OPEN: K = 4, [−6%, −4%]).
  - **Volume confirmation:** when rvol > 1, the stop is tightened by 1/√rvol, but never tighter
    than −2% (`VOLUME_CONFIRMED_STOP_FLOOR`).
  - It is recomputed every candle from the LATEST closed candle's ATR. Because freqtrade only
    tightens, the stop in force is the TIGHTEST seen since grace ended. It ratchets up; it is
    not fixed at the entry ATR.
  - **`simulate_forward` (021 outcome tables) uses the entry ATR.** It is an approximation, not
    the live ratchet.
- **The ATR stop gives volatile pairs WIDER stops in %,** because it is in ATR units.
- **Inert levers.** `self.stoploss` and `trailing_*` do nothing once `use_custom_stoploss` is on
  (NNMT, measured byte-identical). For NN strategies `use_atr_adaptive_stoploss = True`.
- **Exit-reason names in archives.** Every custom-stop exit is labelled `trailing_stop_loss`,
  though there is no trailing (`trailing_stop` is False). `stop_loss` is the initial or grace
  stop. Custom exit tags can carry a trailing space; `.strip()` before comparing.
- **Runtime checks.** `utils/StopInvariants.StopInvariantMixin` halts a run on a wrong-sign stop,
  an over-loose stop, or a grace rescale that moves the stop the wrong way.

### 2.2 Custom exits (`custom_exit_decision`, in order, at the candle's OPEN)

1. Only while in profit and `cexit_enable_profit_checks`:
   - `metric_overbought`: `guard_metric` of `k-1` > 0.98;
   - `take_profit`: net profit > `cexit_take_profit` (0.022 at ARM_D_OPEN; momentum has 10.0,
     i.e. off).
2. `unclog_12h`: age ≥ 12 h and net > 0.5%.
3. `unclog_1d`: age ≥ 24 h and net ≥ 0.
4. `max_hold`: age ≥ `cexit_max_days` (10 days).

Two places adjust this:
- **`HoldWindowMixin`** (`NNNC/_hold_window.py`) defers ONLY `unclog_12h`, to
  `HOLD_WINDOW_HOURS`. The momentum family uses 24 h. `exit_config_hash` cannot see this mixin,
  but `trades_hash` can.
- **Per-family exits** (`utils/FamilyExits.FamilyExitMixin`) pick the exit set from `enter_tag`
  `"<family>:<trigger>"`. Grace and ROI are shared by definition. To read what an instance will
  run without instantiating it, use `class_level_exit_params(cls)`: class-as-self reads
  Parameter defaults, not `sell_params`.

Native ROI is `minimal_roi {"0": 0.05}` in the SimpleTrigger presets.

---

## 3. Entry and exit patterns learned (SimpleTrigger, 023–027; all PROVISIONAL)

**Two trigger families:**
- reversion `gbb`: buy the dip in volatility;
- momentum `breakout_gbb`: buy strength.

They want different exits: momentum does best with no take-profit and a 24 h unclog.

| | late result (f08–f14) |
|---|---|
| momentum alone (`st_bgo_mx`) | +2.50 (+25.19 total) |
| `gbb` | −19.28 (+30.33 early) |

**The reversion family's late loss is entry-side (a regime shift), not exits or contention:**
- No coexistence rule or exit setting fixed it (025).
- A causal volatility-regime gate (`volreg`) cuts it.
- Gated reversion alone (`st_book_rev2`) is +4.20 late, IMPROVED against ungated (p 0.05) (026).
- 56.5% of arm D's trades fall outside the `gbb` trigger: the model does more than filter the
  trigger (023).

**What the exit does to trades (027 US1 measurement, 9.5k trades):**
- **Where things sit:**
  - The label target (+2% gross within 48 bars) is usually reached quickly: median 9 bars,
    inside the grace window.
  - The post-grace stop (−4% to −6%) is 2–3× the target.
- **What the stop does:**
  - The stop rarely kills a trade that was on its way to the target: 2–3% of label-positive
    entries.
  - But 45–63% of stop exits later reach +2% within the 10-day max hold, while only 2–7% do so
    within the 12 h horizon.
- **Replay results (the coupling rules did not pass, so no build):**
  - Locking a gain after the target is touched costs money in both families.
  - Extending the grace window to the horizon gains in the replay for reversion in both halves.
    It failed its pre-declared "killed winners" condition, so it was not built.
  - See `specs/027-target-aware-exits/feasibility.md` and `report.md`.

**Patient exits lose on short-horizon labels.** The model predicts direction, not path, so
retraces hit the stop (H=24 study). A perfect-foresight model still loses when label and exit
disagree (oracle ceiling).

---

## 4. Offline replays and evaluation: traps

**A replay must reproduce the recorded exits first.** 027's null-replay gate requires at least
90% of recorded exits to match on reason AND bar. The first version reproduced 79.7%. Five
mechanics from §1 fixed it to 99.6%:
- the stop updated at the HIGH;
- `dir_correct`;
- stops rounded up to the tick;
- per-candle ROI suppression;
- the force-exit stamp.

`research/exit_coupling/paths.py` is the faithful replay. Reuse it rather than writing another.

**Other traps:**
- **Re-sizing trades after the fact must go through the liquidity cap.**
  - A capped trade keeps its stake under a smaller weight.
  - Committed capital comes from the real `stake_amount`, not 900 × weight.
  - Ignoring this produced a spurious "max drawdown IMPROVED" (026 addendum 4: p 0.04 → 0.27).
  - Use `evaluation/books.resize`.
- **A union of separate backtests can repeat a (fold, pair, open_date) key.** The paired
  bootstrap now SUMS same-key trades (`evaluation/bootstrap.py`). It used to overwrite them,
  silently dropping P&L.
- **A max-drawdown verdict between arms with different mean exposure mostly measures position
  size.** Read it beside `evaluation/books.mean_committed`; Sharpe is the exposure-neutral
  measure (`evaluation/risk.py`).
- **Separate books and sleeves:** independent sleeves are just the sum of their isolated runs.
  There's nothing to run to learn that.
- **Harness habits:**
  - identical `trades_hash` means the change never reached the trades;
  - `run_folds` skips existing records, so use a fresh tag;
  - run `stage.py` and `books.py` with `PYTHONHASHSEED=0`;
  - only the `regime` stage exempts the reversion family's late half from the degeneracy rule.
- **Fingerprints hash `ExitParams` for family-mode arms.** New `ExitParams` fields must be left
  out of the hash while off (027 R5), or every recorded sha changes.

---

*Maintained alongside the features that verify it. When a freqtrade upgrade or a framework change
alters any item, update it here, with the source line or measurement.*
