# Data Model: Trigger-Family Exit Presets (024)

## TriggerFamily (enum)

`reversion` | `momentum` | `unassigned`

## Trigger (registry entry; 2 fields added)

| field | type | rule |
|---|---|---|
| `family` | TriggerFamily or None | Required for every trigger registered by `ensure_builtins()` (test-enforced, SC-006). None is allowed only for ad-hoc construction in tests. |
| `family_reason` | str | One line on what the condition tests (weakness vs strength vs broad filter). Required when `family` is set. |

The initial classification is fixed by spec FR-001.

## FamilyExitSet

A dict of `ExitParams` field overrides, keyed by family, held per preset in
`SimpleTrigger/presets.py`.

| rule | |
|---|---|
| Allowed keys | `take_profit`, `unclog_12h_hours`, `unclog_12h_min_profit`, `unclog_1d_hours`, `max_hold_days`, `atr_stoploss_multiplier`, `atr_stoploss_cap`, `atr_stoploss_floor`, `use_atr_adaptive_stoploss`, `use_volume_confirmation_stoploss`, `static_stoploss` |
| Forbidden keys | grace (`stoploss_grace_*`), ROI, every entry-side setting. Shared by definition (FR-002). Validation refuses them. |
| Resolution | `dataclasses.replace(base_params, **set[family])`. `base_params` is the strategy's default `_exit_params()`, so the preset and any `SELL_OVERRIDES` still apply underneath. |
| Validation | Every key must be an `ExitParams` field. `ExitParams.__post_init__` invariants hold after replacement (for example floor ≤ cap < 0). Unknown family → error at class creation. |

Initial values under the new preset `ARM_D_FAMILY` (= ARM_D_OPEN + these sets; FR-003):

| family | overrides |
|---|---|
| reversion | `{}` (ARM_D_OPEN exactly, so it reproduces `st_gbb`) |
| momentum | `{"take_profit": 10.0, "unclog_12h_hours": 24.0}` (reproduces `st_bgo_mx`, research R5) |
| unassigned | `{}` |

## TradeFamilyTag

`enter_tag` string `"<family>:<trigger>"`, set on the signal row and copied by freqtrade onto
the trade. The family is parsed from the prefix. A missing or unknown prefix uses the shared
exits and increments a per-run counter, `untagged_exits`, which goes into the run record (FR-005).

## CombinedArmConfig (class attributes on a SimpleTrigger subclass)

| attribute | default | rule |
|---|---|---|
| `TRIGGER_NAME` | None | Legacy single-trigger form; equivalent to `TRIGGERS=(name,)`. |
| `TRIGGERS` | None | Tuple of registered trigger names. Exactly one of `TRIGGER_NAME` / `TRIGGERS` may be set. |
| `DISABLED_TRIGGERS` | `()` | Subset of `TRIGGERS`. A disabled trigger is not evaluated at all. Disabling every trigger is an error. |
| `FAMILY_PRIORITY` | `("reversion","momentum","unassigned")` | A permutation of the three families. On an overlap bar the highest-priority family wins the tag. |
| `FAMILY_EXITS` | from the preset | Per-family `FamilyExitSet`. A subclass override MERGES per family, never replaces wholesale (constitution). Empty for every family under `ARM_D_OPEN` / `LEGACY_TRIGGERONLY`, which makes those presets byte-identical to 023. Populated only by `ARM_D_FAMILY`. |

## Fingerprint additions (optional members, research R8)

Present only when the strategy opts in:
- `TRIGGERS`;
- `DISABLED_TRIGGERS`;
- `FAMILY_PRIORITY`;
- `FAMILY_EXITS_RESOLVED`, the full `ExitParams` dict per family;
- `TRIGGER_FAMILIES`, the {trigger: family} map for the arm's triggers.

## ContentionRecord (output of `evaluation/contention.py`)

Per family, and per half (early f00–f07, late f08–f14):

| field | meaning |
|---|---|
| `single_trades`, `combined_trades` | that family's trade count in each arm |
| `lost` | count of single-family entries absent from the combined arm |
| `lost_by_cause` | `pair_held_by_other_family` / `pair_held_same_family` / `slots_full` / `divergence` |
| `gained` | combined-arm entries of that family absent from its single-family arm |
| `lost_pnl_pct`, `gained_pnl_pct` | summed profit_ratio of those trades in their own arm |
