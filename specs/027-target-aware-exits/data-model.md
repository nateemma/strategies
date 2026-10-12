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
| `grace_level_override` | same | float in [−0.27, 0) (arm-wide) | C4 depth |
| `lock_after_target` | same | λ in [0, 1] | C5 |

Plus `target_gain` and `target_horizon_bars`, copied from DeclaredTargets when any switch is on.

Validation (in `ExitParams.__post_init__`): domains as above; `stop_min_depth`
≥ the documented hard bound −0.10. Only C3 and C5 need `target_gain` / `target_horizon_bars`; a
C3 or C5 switch without them is refused.

## WinnersPathProfile (US1 output; early-fixed)

Per family, from f00–f07 entries only:
- MAE-before-target percentiles p50/p80/p90/p95;
- time-to-target percentiles;
- the chosen grid values (p for C1, action for C3, depth for C4, λ for C5) and `k` for C2
  (rule: the early median).
- the null-replay fidelity per family (research R2): the share of trades whose recorded exit
  reason and bar the replay reproduces.

Stored in `feasibility.json`. The arm classes copy the chosen values as literals, with a comment
citing the file.

## FeasibilityRecord (US1)

Per family × half:
- n trades, n label-positive entries, n ambiguous;
- (a) the stopped-too-early shares (within the horizon, within the max hold);
- (b) the killed-winner share and the MAE distribution vs the stop;
- (c) time-to-target; past-horizon trades' eventual vs close-at-horizon P&L;
- (d) touched-but-not-exited give-back;
- replay deltas for C1–C5 at each grid value (C2 at its rule value).

Per coupling × family: support rule text, values, SUPPORTED / NOT SUPPORTED / N/A.

## CouplingArm (US3)

`SimpleTrigger_<Ref>_C<n>` classes, tag `st_cpl_<ref>_c<n>`, a plan copied from the reference arm.

Run record addition: `exit_coupling` (the resolved values).

Liveness (research R5b), from pooled sets against the reference arm, per half:
- trades paired on (fold, pair, open_date) with a different exit (date, rate or reason);
- trades unique to either side;
- folds with a different `trades_hash`;
- `horizon_close` exits (C3).

Zero changed trades in a half means INERT.
