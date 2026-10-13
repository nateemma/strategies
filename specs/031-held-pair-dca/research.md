# Research: Held-Pair Position Add (DCA)

Every decision below was read from the code: freqtrade `freqtrade/optimize/backtesting.py` and
`freqtrade/persistence/trade_model.py` (upstream, read-only), plus this repo's
`Framework/BaseStrategy.py`, `utils/*`, `evaluation/*` and `regime/run_arm.py`.

## R1. How freqtrade backtests a position add

**Decision**: Use freqtrade's native position adjustment. Set `position_adjustment_enable = True`
and `max_entry_position_adjustment = max_adds`, and implement `adjust_trade_position` to return a
positive stake or `None`.

**What the backtest loop does** (`backtest_loop`, per pair and candle):
1. It manages open orders.
2. It processes entries. A pair with an open trade is skipped (no stacking).
3. It fills open entry orders.
4. `_check_trade_exit` runs. **When `position_adjustment_enable` is on, it calls
   `_check_adjust_trade_for_candle` first, then the exit checks.**
5. It processes exit orders.

`_check_adjust_trade_for_candle`:
- calls the strategy with `current_rate = row[OPEN]` and `current_time` = the candle's date;
- passes `max_stake = min(exchange max, wallets.get_available_stake_amount())`, so the wallet
  bound comes free;
- for a positive stake, checks `nr_of_successful_entries <= max_entry_position_adjustment` and calls
  `_enter_trade(..., trade)`. That creates an order at `row[OPEN]` and fills it at once
  (`_try_close_open_order`, since the open is always inside the candle).

The fill runs `Order.close_bt_order`:
- `trade.open_rate` is recomputed as the amount-weighted average (`recalc_trade_from_orders`);
- the stop is re-anchored with `trade.adjust_stop_loss(trade.open_rate, trade.stop_loss_pct)`;
- the backtester then calls `_call_adjust_stop(..., after_fill=True)`, which invokes
  `custom_stoploss(..., after_fill=True)`.

**Consequences**:
- The add fills at the open of the candle after the signal candle, the same timing as a fresh
  entry and as the screen's `open[e]`.
- The add happens BEFORE that candle's exit checks. If the trade's exit fires on the add candle,
  the backtest exits the doubled position. The screen skips the add when the exit falls at or
  before `e` (`tr["x"] <= e`). This is the one known timing difference, and the fidelity check
  (R7) counts it.
- No extra slot is used: adjustment never passes through `trade_slot_available`.
- `custom_stake_amount` and `confirm_trade_entry` are **not** called for adjustments
  (`_get_enter_amount` skips both when `pos_adjust`; `_enter_trade` skips confirmation when
  `pos_adjust`). The add must apply the liquidity cap and the minimum-quote-volume floor itself.
  Wallet and exchange minimum come from freqtrade (`validate_stake_amount`, `min_stake_amount`).
- On the entry candle itself, `_check_trade_exit` also runs, so `adjust_trade_position` is called
  with the trade's age 0. The minimum-age condition (≥ 1 h since the last fill) excludes this.
  The implementation also refuses an add when the last fill is at `current_time`.

**Alternatives rejected**:
- `position_stacking` (separate trades): out of scope (spec), not possible live, and it bypasses
  the slot logic differently.
- A custom re-entry through exit-and-reenter: the screen's ROT policy, which loses on every model.

## R2. Seeing the entry signal inside `adjust_trade_position`

**Decision**: Read `self.dp.get_analyzed_dataframe(pair, timeframe)`. Require its last row's date
to equal `current_time − timeframe` (the signal candle), and use that row's `enter_long == 1`. If
the date does not match, no add happens and a counter records it.

**Rationale**:
- In backtesting the dataprovider returns the cached analysed dataframe (`ft_advise_signals`
  output, signals UNSHIFTED), sliced to `_set_dataframe_max_index`.
- The existing `custom_stake_amount` / `confirm_trade_entry` already read `iloc[-1]` as the signal
  candle for the liquidity check.
- The screen's sizing used `close[e−1] × volume[e−1]` and reproduced the recorded books at 99.7–100%
  fidelity, which confirms that `iloc[-1]` is the signal candle.
- The explicit date check makes this a verified fact on every call, not an assumption. A test pins
  it on a real backtest (T: "signal-row alignment").
- `enter_long` is set by `populate_entry_trend` after every guard, gate and trigger, so the add
  passes through the same guards as a fresh entry (FR-002).

**Entry-confirmation gates**: `FamilyCoexistenceMixin.confirm_trade_entry` caps open trades per
family. It is not called for adjustments, and an add opens no new trade, so the cap correctly
does not apply. The headroom/floor check in `BaseStrategy.confirm_trade_entry` is reproduced by
the add's own sizing (R3).

## R3. Sizing the add

**Decision**: `stake = min(proposed, fillable)`, where:
- `proposed` is the same proposal freqtrade makes for a fresh entry
  (`self.wallets.get_trade_stake_amount(pair, max_open_trades, update=False)`; it is 900 under
  `config_wf.json`);
- `fillable = quote_volume(signal candle) / QUOTE_VOLUME_HEADROOM_MULT`.

The add is skipped (`None`) when `quote_volume < MIN_QUOTE_VOLUME` or `stake < min_stake`. The
result is then capped at `max_stake`, which freqtrade has already bounded by the available wallet.

**Rationale**: This mirrors `custom_stake_amount` plus the floor in `confirm_trade_entry`. The cap
makes the headroom condition hold by construction, so this is the same rule as a fresh entry
(FR-003). The screen used the same expression:
`add = min(STAKE, qv / HEADROOM, WALLET − open_stake)`, with a skip if `qv < MIN_QV or
add < MIN_STAKE`.

**Implementation**: the liquidity arithmetic is factored into one helper in `BaseStrategy`
(`_fillable_stake(pair)`), used by both `custom_stake_amount` and the add. That gives one source,
and `custom_stake_amount` keeps the same values (byte-identical).

## R4. Trade clocks after an add

**Decision**: Add one generic helper to `Framework/BaseStrategy`, `trade_clock_start(trade)`. It
returns `trade.date_last_filled_utc` when `self.position_adjustment_enable` is on, and
`trade.open_date_utc` otherwise. Replace every age computation that feeds an exit or stop rule:

| site | rule |
|---|---|
| `Framework/BaseStrategy.py` custom_stoploss (~1250) | stop grace window |
| `Framework/BaseStrategy.py` custom_exit (~1386) | `custom_exit_decision` age (unclog 12 h / 1 d, max hold, take-profit timing) |
| `Framework/BaseStrategy.py` confirm_trade_exit (~1554) | minimum hold before a discretionary exit |
| `Framework/BaseStrategy.py` `_apply_atr_take_profit` (~1178–1185) | signal bar of the ATR take-profit, and its cache key |
| `utils/StopInvariants.py` (~135) | the invariant checker's own grace test (it must agree with the stop) |
| `NNNC/_hold_window.py` (~54) | HOLD_WINDOW_HOURS deferral |

**Rationale**:
- Gating on `position_adjustment_enable` keeps every existing strategy on `open_date_utc` exactly
  (FR-007). With no add, `date_last_filled_utc == open_date_utc` anyway. The gate makes the
  byte-identity hold by construction rather than by that equality.
- The helper is generic (no strategy names), so it may live in Framework/.
- `date_last_filled_utc` is freqtrade's own property: the max of the open date and the last filled
  order date, and it covers entry fills only for a long-only book with no partial exits.
- The ATR take-profit cache key becomes `(trade.id, clock_start, k)`. A take-profit fixed at entry
  is then re-resolved from the add's signal bar after an add, matching the screen (its replay
  restarts at `e` with `tick = atr(e−1)`).

**Grace after an add**: the backtester calls `custom_stoploss(after_fill=True)` after the add
fills. With the clock restarted, the grace branch returns `grace_level` for `after_fill`, and
freqtrade lets an after-fill stop move in either direction, so the stop widens back to the grace
level from the averaged price. This is exactly the screen's restart. The stop invariants (sign, |v|
≤ 1, never looser than the documented bound outside grace) still apply. Inside grace the checker
uses the same clock (table row above), so it agrees.

**Alternatives rejected**:
- Restart only the grace clock and keep max hold from the original open: not what the screen
  priced. It could be a later variant.
- Always use `date_last_filled_utc`: equal values today, but it would make byte-identity depend on
  a freqtrade property's edge cases instead of a switch.

## R5. Profit reference after an add

**Decision**: No code. Freqtrade averages `trade.open_rate`, `current_profit` is computed from it,
and the custom stop's `_resolve_normal_stop` and the exit decision already use `trade.open_rate` /
`current_profit`. ROI is `{"0": x}`, so it depends on profit only.

**Check**: a unit test asserts that after a simulated add, the tightened stop price is anchored to
the averaged `open_rate`.

## R6. Declaring the switch

**Decision**: a class attribute `HELD_PAIR_ADD` on `EntryFrameworkMixin` (`utils/EntryFramework.py`),
default `None` (off). An enabled declaration is a dict merged key by key along the MRO (like
`FAMILY_EXIT_OVERRIDES`), with keys `max_adds` (int ≥ 1), `min_age_hours` (float ≥ 0) and
`under_water_only` (bool). Defaults come from `HELD_PAIR_ADD_DEFAULTS = {"max_adds": 1,
"min_age_hours": 1.0, "under_water_only": False}`.

At class creation (`_finalise_entry_framework`), an enabled declaration:
- is validated (unknown keys, types and ranges raise `EntryFrameworkError`);
- sets `position_adjustment_enable = True` and `max_entry_position_adjustment = max_adds` on the
  class.

`adjust_trade_position` is defined on the mixin and returns `None` when the switch is off.

**Fingerprint**:
- `evaluation/fingerprint.py` gains `HELD_PAIR_ADD` in `OPTIONAL_ATTRIBUTE_MEMBERS` (hashed only
  when set), so every recorded `framework_sha` is unchanged.
- `position_adjustment_enable` is already a hashed member, and it stays `False` for every existing
  class.

**Model key**: `HELD_PAIR_ADD` and `adjust_trade_position` are added to the mixin's
`MODEL_KEY_IGNORED`. They act after the model, so the 030 model key is unchanged and the
audit test keeps every name classified.

**Rationale**: The entry framework is where the entry signal lives, and both NNNC and SimpleTrigger
inherit it. The generic clock helper sits in Framework/; the strategy-facing switch sits in
utils/, as `FAMILY_EXIT_OVERRIDES` and the regime gate do.

## R7. Fidelity check against the screen

**Decision**:
- `research/held_pair/screen.py`'s `simulate` gains an `add_e` field (the add's bar index, or −1) on
  each trade. This is an additive output; the policy arithmetic is untouched.
- A new `research/held_pair/fidelity.py` loads a DCA arm's fold archives and extracts the trades
  with more than one filled entry order and each add's fill date. It matches them to the screen's
  DCA:all:1h simulation on the same fold's signal table (24 h grace).
- It reports, per fold:
  - adds matched on (pair, candle) / screen adds / backtest adds;
  - trades exiting on the add candle (the R1 timing difference);
  - per-fold pp difference between screen and arm.
- The gate is ≥ 95% of backtest adds matched (SC-002). It is run on ARO's early half (f00–f07).

**Why ARO**: arm D has the highest screen-to-book fidelity (1176/1176), so a mismatch there is the
add's, not the book rebuild's.

## R8. Harness accounting for multi-fill trades

**Findings**:
- `trades_hash` (`regime/run_arm.py summarize`) hashes `pair, open_date, close_date,
  profit_ratio`. An add changes `profit_ratio` (averaged price) but not `open_date`, so the hash
  is a valid liveness signal and stays unchanged for arms with the switch off.
- `evaluation/pool.py` takes `profit_abs`, `stake_amount` (freqtrade reports the total stake after
  adds), `open_date`, and the fold's `wallet_profit_pct`. The per-fold return is wallet-based, so it
  already counts the extra capital.
- Pairing (`compare_pooled`, `bootstrap`) keys on `(fold_id, pair, open_date)`. A DCA trade keeps
  its original open date, so it pairs with its control trade. Its `profit_abs` difference is the
  add's effect plus any re-sequencing.

**Decision**:
- `summarize` adds two fields, `n_adds` and `n_trades_with_adds`, computed from the exported trades'
  `nr_of_entries` (or `orders`) column. They are emitted only when non-zero, so off-arm records
  are unchanged in content as well as hash.
- `pool.py` carries `n_entries` per trade (additive, `None` when absent).
- Tests check pool, pairing and risk on a synthetic two-fill trade (stake = sum of fills, profit_abs
  from the averaged price, key = original open date).

## R9. Exposure, drawdown and the leverage control

**Decision**: a new `research/held_pair/exposure.py` computes, from pooled trades, per arm vs
control:
- average and peak committed stake: an event sweep over open/close and add-fill times, with stake
  per segment;
- maximum drawdown: the existing `evaluation.risk` verdicts;
- **the leverage control**: `extra_stake_capital × base return per unit stake`. Here
  `extra_stake_capital` is the sum over adds of add stake × the trade's remaining holding time,
  normalised to the control's stake-time, and the base return per unit stake-time comes from the
  control's own trades.

This is the arm-level analogue of the screen's leverage check (`DCA gain vs extra stake at the
base return`). The report states "explained by extra capital" when gain ≤ the leverage control
(spec US4.2).

**Alternative rejected**: a doubled-stake control arm. It changes slot/wallet contention for every
trade, not only held ones, so it is not the same capital. It also costs another 60+ fold runs.

## R10. Arms, controls and window

| book | arm tag (15 folds or 3) | class | control | window role |
|---|---|---|---|---|
| ARO | `hp_aro_dca` | `NNNC_HP_ARO_DCA` | `wf_aro_g24` | f08–f14 confirmatory, f00–f07 exploratory |
| GBR | `hp_gbr_dca` | `NNNC_HP_GBR_DCA` | `fs_gbr_gr` | same |
| GFM | `hp_gfm_dca` | `NNNC_HP_GFM_DCA` | `fs_gfm_gr` | same |
| MLX | `hp_mlx_dca` | `NNNC_HP_MLX_DCA` | `fs_mlx_gr` | same |
| ST REV | `st_e22_hp_rev` (e00–e02) + `st_hp_rev` (f00–f14) | `SimpleTrigger_RevOnlyVolreg2_P2_G24_DCA` | `st_e22_g24_rev` / `st_g24_rev` | 2022 confirmatory; 2023–26 exploratory |
| ST MOM | `st_e22_hp_mom` + `st_hp_mom` | `SimpleTrigger_BreakoutGbb_MomExit_G24_DCA` | `st_e22_g24_mom` / `st_g24_mom` | same |

- NNNC arm classes are generated like `research/nnnc_framework/build_arms.py`. Each subclasses the
  model's base class (weights resolve to the same 030 model key, so nothing trains), copies the
  base `.json`, and derives its plan from the control's plan (same folds, thresholds, snapshot pin
  with 24 h grace). The only difference is `HELD_PAIR_ADD = {}` (enabled, defaults).
- Run cost:
  - NNNC: 4 × 15 folds ≈ 13 min (13 s/fold);
  - SimpleTrigger: 2 × 3 folds of 2022 plus 2 × 15 folds exploratory ≈ 15 min (25 s/fold);
  - fidelity uses ARO's early folds of `hp_aro_dca`, which run as part of the ARO arm.
- **Order**: the fidelity arm (`hp_aro_dca` restricted to f00–f07) runs BEFORE the
  pre-registration commit, since it touches only the early half the screen already saw. All
  late-half and 2022 runs come after the commit. The judge refuses any late-half or 2022 record
  older than the commit (as `research/st_grace/judge.py` does).

## R11. Decision rule

As the spec states:
- NNNC uses `research.nnnc_switches.judge.judge_one` (early > 0 AND late > 0 AND paired and both
  risk verdicts map to pass), with the 24 h grace study's maps and minimum control trades 200.
- SimpleTrigger uses `research.st_grace.judge.judge_2022`'s rule on e00–e02.

A per-book verdict is ADOPT / NOT ADOPTED / INCONCLUSIVE:
- INCONCLUSIVE if zero adds, control below minimum trades, or a mapped verdict is inconclusive.
- An adopted gain under 1 pp needs a repeat run of one fold under a fresh tag that reproduces the
  same trade set (quality gate 5). The arms are deterministic: fixed weights, no seed at inference.

Promotion to a default is NOT part of the rule (FR-014).
