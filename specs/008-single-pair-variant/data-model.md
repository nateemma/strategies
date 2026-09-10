# Phase 1 Data Model: Single-Pair (LINK) Variant

## 1. Variant

| field | rule |
|---|---|
| `strategy_class` | `NNNC_MLX_LINK`, subclass of `NNNC_MLX`, empty body |
| `model_dir` | `saved_data/NNNC_MLX_LINK/` — distinct from the control's, enforced by the class name |
| `params_file` | full copy of `NNNC_MLX.json`; `strategy_name` rewritten |
| `config` | `config/config_link.json`, whitelist exactly `["LINK/USDT"]` |
| `artifact_verified` | bool, from an existence check — never an exit code |
| `ungated_warning_seen` | bool; **true voids the run** |

## 2. Pair-Restricted Trade Set

Either arm's trades on one window, filtered to LINK.

| field | note |
|---|---|
| `pair` | always `LINK/USDT` |
| `open_date` | with `pair`, the identity key for set comparison |
| `profit_abs` / `profit_ratio` | expectancy and bootstrap inputs |
| `exit_reason` | carried for forensics; **stripped** — tags have a trailing space |

## 3. Selection Comparison (US2)

| field | note |
|---|---|
| `shared` / `identical` | present in both; identical means same outcome too |
| `variant_only` / `control_only` | counts plus each trade's outcome |
| `verdict` | `same_selection` / `differs` |

`same_selection` is a POSITIVE finding, not a null (FR-010).

## 4. Expectancy Comparison (US3)

| field | note |
|---|---|
| `n`, `mean_pct`, `sd_pct`, `ci_low`, `ci_high` | per arm, per window |
| `overlap` | bool; true ⇒ indistinguishable, with intervals quoted |
| `win_rate` | **suppressed below n=20** (FR-021) |

## 5. Window Result

| field | note |
|---|---|
| `window`, `in_sample` | W1 is the only acceptance window |
| `n_trades_variant` / `n_trades_control` | 0 ⇒ zero-trade, never zero return |
| `delta_pct`, `ci`, `noise_floor` | every delta carries floor and interval |
| `resolved` | false when \|delta\| < floor |
