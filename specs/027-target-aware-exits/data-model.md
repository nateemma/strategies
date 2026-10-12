# Data Model: Target-Aware Exits (027)

## DeclaredTargets

The reference a coupling derives from. Declared on the strategy (FR-007).

| field | type | value for both families | rule |
|---|---|---|---|
| `gain` | float, gross price fraction | 0.02 | > 0 |
| `horizon_bars` | int | 48 | ≥ 1; honoured in bars, never rounded to days (FR-008) |

Class attribute: `EXIT_TARGETS = {"gain": 0.02, "horizon_bars": 48}`. `None` (the default) means no
coupling may be switched on: a switch without targets raises at class creation.

## ExitCoupling

Per-family switch values, merged into `ExitParams`. All `None` = off, today's exit exactly.

| field | ExitParams field | type / domain | coupling |
|---|---|---|---|
| `stop_min_depth` | same | float in [−0.10, 0) | C1 |
| `take_profit_atr_mult` | same | float > 0; resolved TP clamped to [0.01, 0.05] net | C2 (reversion only) |
| `horizon_action` | same | "close" \| "breakeven" | C3 |
| `grace_hours_override` | same | float > 0 (arm-wide: grace is shared) | C4 |
| `lock_after_target` | same | λ in [0, 1] | C5 |

Plus `target_gain` and `target_horizon_bars`, copied from DeclaredTargets when any switch is on.

Validation (in `ExitParams.__post_init__`): domains as above; `stop_min_depth`
≥ the documented hard bound −0.10. Every switch needs `target_gain`/`target_horizon_bars` except
C1 and C4.

## WinnersPathProfile (US1 output; early-fixed)

Per family, from f00–f07 entries only:
- MAE-before-target percentiles p50/p80/p90/p95;
- time-to-target percentiles;
- the chosen grid values (p for C1, action for C3, λ for C5) and `k` for C2.

Stored in `feasibility.json`. The arm classes copy the chosen values as literals, with a comment
citing the file.

## FeasibilityRecord (US1)

Per family × half:
- n trades, n label-positive entries, n ambiguous;
- (a) the stopped-too-early shares (within the horizon, within the max hold);
- (b) the killed-winner share and the MAE distribution vs the stop;
- (c) time-to-target; past-horizon trades' eventual vs close-at-horizon P&L;
- (d) touched-but-not-exited give-back;
- replay deltas for C1/C3/C4/C5 at each grid value.

Per coupling × family: support rule text, values, SUPPORTED / NOT SUPPORTED / N/A.

## CouplingArm (US3)

`SimpleTrigger_<Ref>_C<n>` classes, tag `st_cpl_<ref>_c<n>`, a plan copied from the reference arm.

Run record additions:
- `exit_coupling`: the resolved values;
- `coupling_liveness`: counts per half of trades whose exit the coupling changed:
  - C1 stop exits at a level deeper than the ATR stop would have been;
  - C2 take-profit exits at a level ≠ 2.2%;
  - C3 `horizon_close` exits or break-even stop exits;
  - C4 stop exits inside the extended window;
  - C5 lock exits.
