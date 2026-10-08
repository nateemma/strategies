# Contract: utils/ExitPolicy.py — the single exit primitive

**FR-001 through FR-005.** Pure functions over numbers and arrays. MUST NOT import
freqtrade, MUST NOT take a `Trade`, MUST NOT read a dataprovider.

## Surface

```
atr_stop_fraction(atr_pct, rvol, cfg) -> float
    The stop level relative to entry, negative. The arithmetic that is currently
    duplicated three ways:
        stop = -cfg.atr_stoploss_multiplier * atr_pct
        stop = max(min(stop, cfg.atr_stoploss_cap), cfg.atr_stoploss_floor)
        if cfg.use_volume_confirmation_stoploss and rvol > 1:
            stop = min(stop / sqrt(rvol), -0.02)
    Returns cfg.static_stoploss when atr_pct <= 0 or ATR adaptation is disabled.

effective_stop_fraction(age_hours, atr_pct, rvol, cfg) -> float
    atr_stop_fraction, except inside the grace window where it is
    cfg.stoploss_grace_level.

simulate_forward(high, low, close, atr_pct, rvol, start_index, cfg) -> OutcomeRecord
    Walks forward from start_index applying effective_stop_fraction bar by bar until a
    closing rule fires. Returns profit (net of 2x cfg.fee_per_side), hold_bars and
    close_reason.
```

## Rule precedence within one bar

Fixed and non-negotiable, because OHLC alone cannot order intra-bar events:

1. `stop` — if `low` breaches the effective stop
2. `take_profit` — if `high` reaches it
3. `hold_cap` — if `max_hold_days` is reached
4. `end_of_data`

The stop wins a same-bar tie. This is the conservative reading and matches how the live stop
is evaluated; the alternative would let the table claim profits the live path would not have
collected, which is the phantom-return failure Principle V exists to prevent.

## Callers after the refactor

| caller | keeps | delegates |
|---|---|---|
| `BaseStrategy.custom_stoploss` | freqtrade's return convention (`1.0` = no change; the `(open_rate * (1+normal) / current_rate) - 1` rescale after grace); sourcing `atr_pct_roll` and volume from the dataprovider | `atr_stop_fraction`, `effective_stop_fraction` |
| `BaseStrategy._resolve_normal_stop` | nothing — becomes a one-line delegation | `atr_stop_fraction` |
| `NNPredictStrategy._stop_aware_labels` | its target-construction logic | `effective_stop_fraction` |
| `research/outcomes/table.py` | nothing | `simulate_forward` |

## Guarantees

- **SC-001**: live stop decisions identical on 100% of bars, proven by a characterisation
  fixture generated BEFORE the refactor (research R4 — the ordering matters; a fixture
  generated afterwards proves nothing).
- **SC-002**: the arithmetic exists here and nowhere else, proven by the allowlist test
  (research R3).
- Grace and non-grace branches, and both `after_fill` values, are covered. The grace branch
  returns `1.0` on the non-`after_fill` call, and losing that asymmetry would widen every
  live stop silently.
