# Phase 1 Data Model: Pooled Walk-Forward Evaluation Harness

Six entities. All records are JSON on disk so a run is inspectable and reproducible (FR-022).

## 1. Fold

One train/test split and its execution state.

| field | type | rule |
|---|---|---|
| `fold_id` | str | `f00`..`fNN`, stable and ordered by test start |
| `train_start` / `train_end` | date | anchored: `train_start` is fixed at the data start |
| `test_start` / `test_end` | date | **MUST NOT** intersect [train_start, train_end] (FR-002) |
| `strategy_class` | str | generated identity, e.g. `NNNC_WF_f07` (FR-003) |
| `model_path` | str | fold-unique; two folds sharing one is a hard error |
| `seed` | int | MLX_SEED used |
| `status` | enum | `planned` / `trained` / `evaluated` / `failed` / `zero_trade` |
| `model_artifact_verified` | bool | from an artefact existence check, never an exit code (FR-004) |
| `arm_tag` | str | the tag its `run_arm.py` record was written under |
| `n_trades` | int | 0 is legitimate and means zero-trade, never zero-return (FR-012) |

**State transitions**: `planned → trained → evaluated`; any step may go to `failed`. `evaluated`
with `n_trades == 0` becomes `zero_trade`, which still counts as a completed fold but is excluded
from expectancy statistics.

## 2. Fold Plan

The ordered set of folds, validated as a whole before any compute is spent (FR-001).

| field | type | rule |
|---|---|---|
| `folds` | Fold[] | non-empty, ordered, `test_start` strictly increasing |
| `anchored` | bool | true for this feature |
| `min_train_days` | int | 730 (24 months) |
| `test_days` | int | 90 |
| `universe` | str[] | pair list; the incumbent 11 for phases B-D |
| `validated` | bool | **false blocks execution entirely** |
| `validation_errors` | str[] | each names the offending fold |

Validation rules, all blocking: no train/test overlap on any fold; `strategy_class` unique across
folds; `model_path` unique across folds; every `test_end` within available data.

## 3. Pooled Trade Set

The union of per-fold trades, fold identity retained (FR-006).

| field | type | note |
|---|---|---|
| `fold_id` | str | retained so any statistic can be recomputed per fold or per era (FR-021) |
| `pair` | str | with `open_date`, the pairing key `compare.py` uses |
| `open_date` / `close_date` | datetime | |
| `profit_abs` | float | the bootstrap quantity |
| `profit_ratio` | float | |
| `exit_reason` | str | carried for forensics, not used in any verdict |
| `pair_group` | enum | `incumbent` / `added` — populated only when the universe is expanded (FR-014) |

Invariant: a trade appears in exactly one fold. Overlapping test ranges would break this and are
already forbidden by the plan validator.

## 4. Power Report

The feature's headline output (FR-007, FR-008, FR-009).

| field | type | note |
|---|---|---|
| `n_trades_pooled` | int | against 99 today (SC-002 ≥ 350) |
| `ci_low` / `ci_high` | float | bootstrap 95% CI on the pooled return statistic |
| `half_width_pp` | float | **the achieved floor** (SC-001 ≤ 4.0) |
| `baseline_half_width_pp` | float | 7.633, fixed |
| `improvement_ratio` | float | baseline / achieved |
| `min_resolvable_effect_pp` | float | stated explicitly so no reader over-reads the instrument (SC-004) |
| `predicted_half_width_pp` | float | the 1/sqrt(n) prediction |
| `sqrt_n_shortfall_pp` | float | achieved − predicted; positive means pooling underdelivered |
| `inter_fold_correlation` | float | the explanation when it does (FR-009, SC-007) |
| `per_fold` | object[] | per-fold trade count and return (FR-010) |
| `folds_complete` / `folds_planned` | int | unequal ⇒ the report is marked **PARTIAL** (FR-020) |
| `seed_identical_folds` | str[] | folds where two seeds gave identical trades (FR-011) |

## 5. Pair Group Report

Only when the universe is expanded (FR-014, SC-006).

| field | type | note |
|---|---|---|
| `group` | enum | `incumbent` / `added` |
| `n_trades` | int | the samples the group contributes |
| `expectancy_per_trade` | float | the question — do added pairs hold expectancy? |
| `verdict` | enum | `valid_power_source` / `invalid_dilutes_expectancy` |

## 6. Run Manifest

Makes a run reproducible and auditable (FR-017, FR-022).

| field | type | note |
|---|---|---|
| `run_id` | str | timestamp-based |
| `plan` | Fold Plan | the exact plan executed |
| `git_commit` | str | strategy-repo HEAD at run time |
| `bootstrap_seed` | int | 0, matching `compare.py` |
| `elapsed_minutes` | float | actual cost against the stated budget (FR-017) |
| `status` | enum | `complete` / `partial` |
