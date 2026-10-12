# Interfaces: Target-Aware Exits (027)

## C1 — `research/exit_coupling/feasibility.py` (US1; no backtests)

```
python -m research.exit_coupling.feasibility --momentum st_bgo_mx --reversion st_book_rev2 \
    --out specs/027-target-aware-exits/feasibility.md
```

- Reads the arm archives (via `evaluation.pool` helpers) and price data
  (`research.clfstudy.data.load_pair`, then the strategy's own populator for `atr_pct_roll`).
- Writes `feasibility.md` (tables, per family × half) and `feasibility.json` (FeasibilityRecord +
  WinnersPathProfile).
- Grid values are chosen from f00–f07 rows only. A test plants a late-only effect and checks that it
  cannot change the choice.
- Exit codes: 0 if ≥ 1 coupling is SUPPORTED, 3 if none is (STOP), 2 if an input is missing.
- Pure functions (unit-tested on synthetic paths):
  - `path_metrics(ohlc, entry_idx, open_rate, stop_fn, target, horizon, max_bars)` returns touch
    bar, MAE before touch, ambiguous flag, stop-breach bar;
  - `replay(trade, path, rule)` returns the counterfactual exit price, bar and reason.

## C2 — `utils/ExitPolicy.py` (EDIT)

- `ExitParams` gains the optional fields of data-model ExitCoupling (default `None`), validated in
  `__post_init__`.
- `custom_exit_decision(..., *, bars_open=None, target_touched=None, take_profit_override=None)`:
  - new keyword-only inputs, default `None`, giving today's behaviour exactly;
  - C2 uses `take_profit_override` in place of `params.take_profit`;
  - C3 "close" returns `CLOSE_HORIZON = "horizon_close"` when
    `bars_open >= target_horizon_bars and not target_touched`.
- `coupled_stop_floor(params) -> float | None` returns the C1 bound.
- `lock_price_ratio(params) -> float | None` returns the C5 lock as a multiple of entry
  (`1 + λ·target_gain`).
- `simulate_forward` raises `NotImplementedError` if any coupling field is set.
- `SIMULATION_LOGIC_VERSION` is unchanged, because simulation behaviour is unchanged.

## C3 — `Framework/BaseStrategy.py` (EDIT; strategy-agnostic)

- `_exit_params` maps the new class attributes/overrides into `ExitParams`. Missing attributes
  give `None`.
- `_target_touched(pair, trade) -> bool`: whether any CLOSED candle after entry has high
  ≥ open_rate·(1 + target_gain). It reads the analysed dataframe, never `trade.max_rate` (research
  R4).
- `custom_stoploss`:
  - C4: the grace length comes from `grace_hours_override` when set;
  - C1: the normal stop is `min(atr_stop, stop_min_depth)`;
  - C3 "breakeven" and C5 lock: once their condition holds, the returned stop is at least the lock
    price. If the price is already at or below the lock, it returns `STOP_AT_MARKET_RATIO`, never a
    positive ratio.
- `custom_exit` passes `bars_open`, `target_touched` and the C2 `take_profit_override` (from the
  signal bar's `atr_pct_roll`) to `custom_exit_decision`.
- Each new branch is skipped when its field is `None`. No family or trigger names are used.

## C4 — `utils/StopInvariants.py` (EDIT)

- The bound floor includes `stop_min_depth` when set.
- Tests: C1 within bound; a C1 that is too deep raises; the lock never returns a positive ratio.

## C5 — `evaluation/fingerprint.py` (EDIT)

- `EXIT_TARGETS` and `EXIT_COUPLING` become `OPTIONAL_ATTRIBUTE_MEMBERS`, hashed only when set.
- In `_family_members`, the ExitParams dict drops the new fields while they are `None`.
- Test: the shas of all 023–026 arm classes and of `NNNC_MLX_Open` are unchanged.

## C6 — `regime/run_arm.py` (EDIT)

- The record gains `exit_coupling` (the resolved values, or `None`) and `coupling_liveness`
  (counted from the archive's exit reasons and stop levels, per data-model).

## C7 — Arm classes (US3, only for SUPPORTED couplings)

- `SimpleTrigger/SimpleTrigger_<Ref>_C<n>.py`: subclasses of the reference class setting
  `EXIT_TARGETS` and one coupling via `FAMILY_EXIT_OVERRIDES` (family arms) or `SELL_OVERRIDES` /
  attributes. C4 is set arm-wide.
- Plans are `plan_st_cpl_<ref>_c<n>.json`, copied from the reference plan.
- Judged with `halves.split` (P&L + risk) against the reference arm. Results go to `decisions.md`.
