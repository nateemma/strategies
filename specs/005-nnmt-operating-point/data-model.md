# Data Model — 005 NNMT Operating Point

Five entities. No database; everything is JSON arm records, backtest archives
and written findings.

## 1. ControlConfiguration

NNMT's pre-inheritance settings, pinned so the feature has a fixed reference.

| field | value | note |
|---|---|---|
| label gain / loss threshold | 0.007 | pre-inheritance; class now says 0.020 |
| take-profit | 0.032 | pre-inheritance; class now says 0.022 |
| entry guards | rvol 0.25, guard 0.3, adx 38, atr 0.006, bb_width 0.06, close_norm −0.1 | NNMT's own swept values, unchanged |
| confidence threshold | 0.5 | NNMT's own override, unchanged |
| exit relative-volume gate | 2.0 | unchanged in value; newly tunable |
| max hold | 21 days | vs NNNC's 10; long-standing |

**Validation**: the control is only valid if the model on disk was trained
against its label. If not, one retrain is required before any comparison.

## 2. HeadroomMeasurement

The three-way comparison that gates the feature.

| field | notes |
|---|---|
| `no_model_return` / `no_model_trades` | model's entry opinion bypassed |
| `real_model_return` / `real_model_trades` | the control |
| `perfect_model_return` / `perfect_model_trades` | oracle; lookahead |
| `headroom` | perfect − real, on the acceptance window |
| `captured` | real − no-model |

**Validation**: all three arms must share the same configuration and the same
model weights where applicable; the oracle and bypass arms must alias the
control's model rather than training their own.

## 3. GuardContribution

One guard's individual effect, from a leave-one-out.

| field | notes |
|---|---|
| `guard` | which threshold was neutralised |
| `delta_return`, `delta_trades` | per window |
| `inert` | true when neither trades nor return change at all |

**Validation**: the sum of individual deltas is reported alongside the joint
effect, so interaction is visible rather than assumed.

## 4. InheritedChange

One of the two configuration changes NNMT received from the NNNC work.

| field | notes |
|---|---|
| `parameter` | label threshold, or take-profit |
| `old_value` / `new_value` | 0.007→0.020, or 0.032→0.022 |
| `couples_to` | label additionally drives the profit head at PROFIT_HORIZON (R2) |
| `verdict` | keep old / adopt new / adopt a third value |
| `seeds` | at least two, with the spread |

**Validation**: each change is measured ALONE against the control. The label arm
pins the profit-head threshold so it stays single-variable.

## 5. Arm

One evaluated configuration, as `regime/arms/<tag>_<window>.json`.

| field | notes |
|---|---|
| `tag`, `window`, `strategy` | |
| `n_trades`, `trades_hash`, `exit_config_hash` | liveness and identity |
| `wallet_profit_pct`, `max_drawdown_pct`, `profit_factor`, `win_rate` | |
| `returncode` | must be 0 or the record is not evidence |
| `is_lookahead` | true for oracle arms; forces the ceiling labelling |
| `seed` | which MLX_SEED produced the model |

**Validation**: an arm whose trades are identical to its control is
INCONCLUSIVE, not refuted — the change did not reach the trades.

## Relationships

    ControlConfiguration 1───* Arm
    Arm                  1───* TradeRecord         (via its backtest archive)
    HeadroomMeasurement  1───3 Arm                 (no-model, real, perfect)
    GuardContribution    1───1 Arm                 (one leave-one-out arm each)
    InheritedChange      1───* Arm                 (>= 2 seeds, own operating point)
