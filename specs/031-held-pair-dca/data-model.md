# Data Model: Held-Pair Position Add (DCA)

## HeldPairAdd (strategy declaration)

`HELD_PAIR_ADD` class attribute on `EntryFrameworkMixin` (utils/EntryFramework.py).

| field | type | default | rule |
|---|---|---|---|
| (attribute itself) | `None` or dict | `None` | `None` = OFF. A dict, even `{}`, = ON. |
| `max_adds` | int | 1 | ≥ 1. Becomes `max_entry_position_adjustment`. |
| `min_age_hours` | float | 1.0 | ≥ 0. Hours since the trade's last entry fill. An add is refused when the age is below this, and always when it is 0 (entry candle). |
| `under_water_only` | bool | False | When True, add only if `current_profit < 0` at the add candle's open. |

- Merge: dicts merge key by key along the MRO (parent first). A subclass declaring `None`
  explicitly turns the switch off.
- Validation at class creation: an unknown key, a wrong type, or a value out of range raises
  `EntryFrameworkError`.
- Derived class attributes when ON: `position_adjustment_enable = True` and
  `max_entry_position_adjustment = max_adds`. When OFF the base values (`False`, `-1`) are left
  untouched.
- Hashing: `framework_sha` includes the resolved dict only when ON (`OPTIONAL_ATTRIBUTE_MEMBERS`).
  It is model-key IGNORED.

## AddDecision (per call of `adjust_trade_position`)

Inputs: `trade`, `current_time`, `current_rate` (candle open), `current_profit`, `min_stake`,
`max_stake` (already wallet-bounded by freqtrade).

The rules are checked in order. The first failure returns `None` and increments the named counter.

1. switch on (`off`);
2. long trade, with no open order (`not_long` / `open_order`);
3. `trade.nr_of_successful_entries - 1 < max_adds` (`max_adds`);
4. the last analysed row's date equals `current_time − timeframe` (`row_misaligned`);
5. `enter_long == 1` on that row (`no_signal`);
6. age since `trade.date_last_filled_utc` is > 0 and ≥ `min_age_hours` (`too_young`);
7. `under_water_only` ⇒ `current_profit < 0` (`in_profit`);
8. signal-candle quote volume ≥ `MIN_QUOTE_VOLUME` (`thin`);
9. `stake = min(proposed_stake, fillable_stake, max_stake)` ≥ `min_stake` (`below_min`).

Output: the `stake`, with order tag `"hp_add"`.

The counters (`adds` plus one per refusal) live in the strategy instance. Freqtrade runs in a
`run_arm` subprocess, so they leave it through a sidecar file:
- `hp_counters_<StrategyClass>.json` in the backtest results directory
  (`self.config["user_data_dir"]/backtest_results`);
- it is rewritten at process exit (`atexit`, registered on the first `adjust_trade_position` call);
- it is written only when the switch is on.

`regime/run_arm.summarize` copies it into the arm record as `hp_counters`, and deletes the file so
the next fold cannot read a stale one. Fidelity and judge read `hp_counters` from the record. They
are the liveness and diagnosis signal for "why no add".

## Trade clock

`trade_clock_start(trade) -> datetime`, defined in `Framework/BaseStrategy`:
- returns `trade.date_last_filled_utc` if `position_adjustment_enable` is on;
- otherwise returns `trade.open_date_utc`.

Every age-based exit and stop rule reads it (research R4 table).

## Arm record additions (`regime/run_arm.summarize`)

| field | when present | meaning |
|---|---|---|
| `n_adds` | > 0 | total filled entry orders beyond the first, over the fold |
| `n_trades_with_adds` | > 0 | trades with ≥ 2 filled entry orders |
| `hp_counters` | switch on | the AddDecision counters from the sidecar file (adds and each refusal) |

`trades_hash` is unchanged in definition.

## Pooled trade addition (`evaluation/pool.py`)

`n_entries`: the number of filled entry orders, or `None` if the archive lacks it. It is additive;
no consumer requires it.

## Pre-registration (`research/held_pair/preregistration.json`)

It contains:
- `exempt_records`: the smoke run (`hp_smoke` f00) and the fidelity run's records (`hp_aro_dca`
  f00–f07), allowed to predate the commit
  and labelled EXPLORATORY;
- `setting`: the declared HELD_PAIR_ADD, with the screen selection it came from;
- `grace`: 24 h;
- `books`: per book the arm tag, class, control, folds, and window roles;
- `rule`: `judge_one` for NNNC, the 2022 rule for SimpleTrigger, verdict maps, and minimum control
  trades;
- `reporting`: exposure, drawdown and the leverage control;
- `screen`: the early-half predictions and the fidelity result.

The judge refuses to run if the file is uncommitted or dirty, or if any DCA arm record that is not
listed in `exempt_records` predates its commit.

## Book verdict (`research/held_pair/verdicts.json`)

Per book:
- `decision` ∈ {ADOPTED, NOT ADOPTED, INCONCLUSIVE};
- the judge_one / judge_2022 payload;
- `n_adds` per fold;
- `exposure` {avg_stake, peak_stake} for arm and control;
- `maxdd` for arm and control;
- `leverage_control_pp`;
- `decomposition` {same_entry_pp, forgone_n, forgone_pair_held_n, forgone_pp, new_n, new_pp};
- `beats_leverage` (bool);
- `evidence` label.
