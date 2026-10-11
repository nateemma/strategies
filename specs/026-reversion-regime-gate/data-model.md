# Data Model: Regime Gate on the Reversion Family (026)

## RegimeInstrument

| field | values | rule |
|---|---|---|
| `id` | `breadth` / `volreg` / `persist` / `btctrend` | Fixed set (R1) |
| series | date-indexed float, 15m | Causal: the value at t uses bars ≤ t (R5). NaN during warm-up or with < 6 pairs. |

## GateRule (chosen on f00–f07, R2)

| field | values |
|---|---|
| `instrument` | an instrument id |
| `threshold` | an early-entry percentile value (p33 / p50 / p67) |
| `off_when` | `"ge"` (off if value ≥ threshold) or `"lt"` |
| `on_share_early` | the share of early reversion trades in "on" (must be ≥ 0.50) |

NaN → "on" (reversion allowed).

## FeasibilityRecord (per instrument; US1)

| field | meaning |
|---|---|
| `rule` | the chosen GateRule, or None if no option met on-share ≥ 0.50 |
| `pnl[state][half]` | reversion pp contribution: state ∈ {on, off}, half ∈ {early, late}, for `st_fam_combo` (primary) and `st_gbb` (cross-check) |
| `trades[state][half]` | trade counts |
| `off_bar_share[half]` | the share of 15m bars in "off" |
| `off_runs[half]` | the number of contiguous "off" runs |
| `undefined_bars` | warm-up and too-few-pairs bars (counted "on") |
| `verdict` | PASS iff `pnl.off.early < 0` AND `pnl.off.late < 0` AND `pnl.on.early > 0` AND `on_share_early ≥ 0.5` (primary arm) |
| `cross_check_agrees` | the same verdict on `st_gbb` (reported, not a veto) |

## GatedArm config (US2; SimpleTrigger class attribute)

`REVERSION_REGIME_GATE = None | {"instrument": id, "threshold": float, "off_when": "ge"|"lt"}`.

Gated arms and the always-on control set `startup_candle_count = 3600` (analyze H1). Validation
refuses a gate with `startup_candle_count < 3600`: the instrument would be undefined for part of
each fold, and so partly INERT.

- `None` reproduces `st_fam_combo`.
- Validation: a known id, a finite threshold, `off_when` in the allowed set, and an enabled
  reversion trigger present (otherwise the gate is INERT by construction).

## Run-record additions (sink; 025 C3 format)

`entry_condition_counts.regime_gate_refused` (`pass` = reversion rows dropped, `total` = rows) and
`entry_condition_counts.regime_gate_off_bars`.
