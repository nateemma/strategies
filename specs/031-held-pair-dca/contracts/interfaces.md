# Interfaces: Held-Pair Position Add (DCA)

## Strategy declaration

```python
class NNNC_HP_ARO_DCA(NNNC_A_Raw_Open):
    HELD_PAIR_ADD = {}            # ON with defaults: max_adds 1, min_age_hours 1.0, under_water_only False
```

- OFF is `HELD_PAIR_ADD = None` (the inherited default).
- Invalid declarations raise `EntryFrameworkError` at import.

## `EntryFrameworkMixin.adjust_trade_position(trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs) -> float | None | tuple`

- It follows freqtrade's callback signature.
- It returns `(stake, "hp_add")` for an add, and `None` otherwise.
- It never returns a negative value (no partial exits).
- It follows the AddDecision rules in data-model.md.

## `BaseStrategy.trade_clock_start(trade) -> datetime`

- It is generic: it reads only `self.position_adjustment_enable` and trade fields.
- It is the single source for trade age in exit and stop rules. The exit-parameter guard test is
  unaffected, since it names no exit parameter.

## `BaseStrategy._fillable_stake(pair) -> tuple[float, float]`

- It returns `(quote_volume, quote_volume / QUOTE_VOLUME_HEADROOM_MULT)` for the last analysed
  row.
- It is shared by `custom_stake_amount` (value unchanged) and the add.

## Research CLIs

```text
PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.build_arms        # classes, .json, plans
PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.fidelity hp_aro_dca --folds f00-f07
PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.exposure <arm> <control>
PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.judge             # -> verdicts.json
```

Exit codes:
- `0` ok;
- `2` refused (pre-registration missing/dirty, record older than the commit, or fidelity below the
  gate);
- `1` error.
