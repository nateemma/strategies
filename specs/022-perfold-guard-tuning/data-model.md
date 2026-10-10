# Data Model: Per-Fold Guard Tuning

## GuardConfig

One point in the search space — the object being tuned.

| field | type | constraint |
|---|---|---|
| `entry_atr_pct` | float | one of 4 enumerated levels; the heaviest lifter (−7.33pp) |
| `entry_adx_threshold` | float | one of 4 enumerated levels (−5.95pp) |
| `entry_rvol_threshold` | float | one of 4 enumerated levels (−3.42pp) |
| `entry_enable_guards` | bool | **MUST be `True`** (FR-008) — also gates `volume > 0` |
| `config_id` | str | stable id over the three tuned values, so a grid point is citable |

**Not in this model, deliberately**: `entry_guard_threshold` (proven inert — excluded by FR-011,
and used only as the FR-013 negative control), `entry_bb_width_threshold` and
`entry_close_norm_threshold` (lightest contributors, −0.17pp and −1.01pp). No exit field appears
here at all; the exit is not tunable in this feature (FR-009).

**Validation**: the enumerated levels are fixed before the search runs and committed, so the
candidate set is finite and inspectable (FR-005). A configuration whose values fall outside the
enumerated levels is rejected rather than scored.

## GridScore

One `(fold, config, window)` scoring. Produced by `pick_guards.py`.

| field | type | notes |
|---|---|---|
| `fold_id` | str | `f00`..`f14` |
| `config_id` | str | → GuardConfig |
| `window` | enum | `train` \| `test` — `train` drives selection; `test` is for the gate only |
| `proxy_score` | float | **A PROXY, NEVER A RETURN.** Pooled mean outcome over admitted, concurrency-selected bars, from the outcome table. Used to SELECT configurations and nothing else; FR-018 restricts every P&L claim to `evaluation/` arm records. Named `proxy_score` rather than `score` because feature 021's central error was a table-derived figure being read as a realisable return |
| `n_admitted` | int | bars the configuration admits |
| `n_taken` | int | after concurrency selection |
| `exit_config_hash` | str | must equal the arms' hash, else the score describes a different exit |
| `below_floor` | bool | `n_taken < 25` → the configuration is rejected (FR-012) |

**State transition**: a GridScore with `below_floor` true may never become the selected
configuration for its fold. If every configuration in a fold is below the floor, that fold is
**unpowered** (FR-017) and contributes no return observation.

## TrackabilityResult

The Principle VII gate's output. One per feature run; it is the thing FR-006 reads.

| field | type | notes |
|---|---|---|
| `chosen_by_fold` | map | `fold_id` → `config_id` selected on that fold's train window |
| `n_distinct_choices` | int | how often the winner changes — **variation, not yet trackability** |
| `best_constant` | str | `config_id` maximising summed test score, chosen in hindsight |
| `regret_adaptive` | float[15] | per fold |
| `regret_constant` | float[15] | per fold |
| `mean_regret_delta` | float | `mean(regret_adaptive) − mean(regret_constant)`; **negative = adaptive wins** |
| `paired_interval` | (float, float) | on the per-fold regret difference |
| `verdict` | enum | `trackable` \| `not_trackable` |

**Gate semantics**: `verdict == trackable` requires `mean_regret_delta < 0` **and** the paired
interval excluding zero. Anything else is `not_trackable`, and FR-006 then forbids arm C.

## Arm

One configuration run across all 15 folds. Three instances already exist and are reused.

| field | type | notes |
|---|---|---|
| `arm_id` | enum | `A` \| `B` \| `C` \| `D` \| `E` |
| `tag` | str | `wf_gbbo` (A), `wf_gbbg` (B), `wf_gbbgrid` (C), `wf_aro` (D) |
| `per_fold_wallet` | float[15] | `wallet_profit_pct` from each arm record |
| `per_fold_trades` | int[15] | a fold with 0 is excluded, never averaged as 0.0pp |
| `pooled_wallet` | float | sum over powered folds |
| `folds_positive` | int | out of the powered folds |
| `trades_hash` | str[15] | per fold; drives the FR-013 inertness proof |
| `exit_config_hash` | str | **MUST be identical across A, B, C, D, E** (FR-009, SC-004) |
| `guards_written` | map | `fold_id` → the GuardConfig actually written, from the generated params |

**Invariant**: `exit_config_hash` equality across all arms is what makes the comparisons
exit-for-exit. A mismatch invalidates every comparison and is a hard failure, not a warning.

## ProvenanceVerdict

US1's output, and FR-001 requires it to exist before any arm runs.

| field | type | notes |
|---|---|---|
| `parameter_set` | str | the pre-2026-09-06 tuned guard values |
| `tuning_window` | str \| null | date range, or `null` meaning **unrecoverable** |
| `overlaps_folds` | enum | `yes` \| `no` \| `unknown` |
| `evidence` | str[] | commit ids, file paths, any `.fthypt` records |
| `arm_b_status` | enum | `trustworthy` \| `in_sample` \| `untrustworthy` |

**Rule**: `tuning_window == null` ⇒ `overlaps_folds = unknown` ⇒ `arm_b_status =
untrustworthy`. Arm B's +18.76pp must carry this status wherever it is quoted (FR-002), and
C vs B loses its interpretation while C vs A keeps its own.
