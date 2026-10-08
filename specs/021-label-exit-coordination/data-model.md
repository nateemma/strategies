# Data Model: Label-Exit Coordination (Phase 1)

## ExitConfig

The parameters the exit policy reads. Immutable; its hash keys the outcome table.

| field | type | source |
|---|---|---|
| `atr_stoploss_multiplier` | float | `BaseStrategy` (2.5) |
| `atr_stoploss_cap` | float | `BaseStrategy` (−0.02, tightest allowed) |
| `atr_stoploss_floor` | float | `BaseStrategy` (−0.04, loosest allowed) |
| `use_atr_adaptive_stoploss` | bool | `BaseNNStrategy` (True for the NN families) |
| `use_volume_confirmation_stoploss` | bool | `BaseStrategy` |
| `static_stoploss` | float | `self.stoploss` fallback when ATR is unavailable |
| `stoploss_grace_hours` | float | strategy params (6.5 at the Open operating point) |
| `stoploss_grace_level` | float | strategy params (−0.27 at the Open operating point) |
| `take_profit` | float | `cexit_take_profit` (0.022 at the Open operating point) |
| `max_hold_days` | int | `cexit_max_days` (10 at the Open operating point) |
| `fee_per_side` | float | the fee freqtrade applies in a backtest (FR-006a) |

**Validation**: `atr_stoploss_floor <= atr_stoploss_cap < 0`; `take_profit > 0`;
`max_hold_days >= 1`; `fee_per_side >= 0`. `config_hash` is the first 16 hex characters of a
SHA-256 over the canonical JSON of every field above.

## OutcomeRecord

One row per bar. The result of a hypothetical long entry at that bar's close, closed by the
`ExitConfig`'s policy.

| field | type | notes |
|---|---|---|
| `date` | timestamp | the entry bar |
| `pair` | str | |
| `__outcome_profit` | float | realised fraction, NET of `fee_per_side` on both sides (FR-006a) |
| `__outcome_hold_bars` | int | bars from entry to close |
| `__outcome_close_reason` | enum | `stop` / `take_profit` / `hold_cap` / `end_of_data` |
| `__outcome_positive` | bool | `__outcome_profit > 0`; the shared positive-outcome condition |
| `__outcome_fillable` | bool | quote volume supports the production stake (FR-009a) |
| `__outcome_fillable_fraction` | float | fraction of the intended stake that could fill |
| `__outcome_gapfill_share` | float | share of the hold window that is gap-fill bars (FR-020) |

**The `__outcome_` prefix is load-bearing**, not cosmetic: it is how FR-009's leakage guard
works. See `contracts/outcome-table.md`.

**State transitions**: a record moves from unresolved to one of the four close reasons exactly
once, at the first bar where a closing rule fires. Rule precedence within a bar must be fixed
and documented, because a bar whose low breaches the stop and whose high reaches the
take-profit is ambiguous from OHLC alone — the stop takes precedence, which is the
conservative reading and matches how the live stop is evaluated.

## OutcomeTable

The set of `OutcomeRecord`s for one `(dataset, ExitConfig)`.

| field | type |
|---|---|
| `records` | OutcomeRecord[] |
| `config_hash` | str |
| `exit_config` | ExitConfig (full, in the sidecar) |
| `dataset_id` | str — pairs, timeframe, timerange |
| `built_at` | timestamp |

**Invariant**: loading a table whose `config_hash` differs from the requested `ExitConfig`'s
hash is an error, never a silent reuse (FR-007).

## Trigger

A boolean condition over bars, carrying no outcome test (FR-010).

| field | type | notes |
|---|---|---|
| `name` | str | registry key |
| `evaluate` | callable | `(dataframe) -> bool series` |
| `required_columns` | str[] | asserted present before evaluation |
| `source` | str | the `LabelMethod` it was derived from, or `native` |

**The empty trigger** (`name="empty"`, every bar True) is a required member: it is the null
the verdict rule compares against (FR-014a).

**Constraint**: no trigger may reference a `__outcome_` column, and none may be pair-specific
(Constitution X).

## Label

`trigger AND __outcome_positive` (FR-011). Carries no outcome definition of its own.

## TriggerScore

One leaderboard row. The two columns are deliberately separate and must not be combined
(FR-016).

| field | type | notes |
|---|---|---|
| `trigger` | str | |
| `base_rate` | float | share of bars the trigger fires on |
| `economic_value` | float | mean realised outcome over triggered bars, no model |
| `economic_value_ci` | (float, float) | 95%, block bootstrap (research R9) |
| `economic_value_fillable` | float | the same over fillable triggered bars only (FR-009b) |
| `learnability_mcc` | float | the one fixed probe (FR-015) |
| `learnability_se` | float | seed/window spread |
| `gapfill_share` | float | FR-020, SC-009 |
| `verdict` | enum | `selection_value` / `decoration` / `indeterminate` (FR-014b) |

**Verdict derivation** — the rule is total, and is applied against the `empty` trigger:

- `selection_value` — `economic_value` exceeds empty's by more than the CI on the difference
- `decoration` — it does not, while `learnability_mcc` exceeds empty's by more than the SE on
  that difference
- `indeterminate` — neither holds
