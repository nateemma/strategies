# Contract: Output Artifact Schemas

**Feature**: 001-market-phase-entry-gate | **Date**: 2026-09-02

Durable outputs written under `regime/`. Schemas are contracts: a downstream phase reads the previous phase's artifact rather than recomputing, so column names and semantics are fixed.

---

## `instrument_catalogue.csv`

The register of every phase instrument. Written by Phase A, amended by Phase B.

| Column | Type | Notes |
|---|---|---|
| `name` | string | Unique |
| `level` | string | `asset` \| `market` \| `btc` |
| `states` | string | Pipe-separated vocabulary, always including `UNKNOWN` |
| `window_bars` | int | Fixed lookback |
| `params` | string | JSON-encoded estimator parameters |
| `is_baseline` | bool | Exactly one row is `True` — the reproduced exp4 taxonomy |
| `status` | string | `PENDING` \| `SURVIVING` \| `ELIMINATED` |
| `status_reason` | string | Empty unless `ELIMINATED`; must contain a measured number |

---

## `lookahead_validation.csv`

The Phase A gate. One row per instrument per cut point.

| Column | Type | Notes |
|---|---|---|
| `instrument` | string | |
| `cut_date` | date | Truncation point |
| `n_bars_compared` | int | Overlapping region size |
| `n_differing` | int | **Must be 0 to proceed** |
| `first_diff_date` | date, nullable | Populated when `n_differing > 0`, for debugging |

**Contract**: any row with `n_differing > 0` blocks that instrument from Phase B. The `first_diff_date` column exists because the common causes — a centred window, a backward `ffill`, a cross-sectional operation over a panel containing future rows — are each identifiable from where the divergence starts.

---

## `actionability.csv`

Phase B characterisation. One row per instrument per era per state.

| Column | Type | Notes |
|---|---|---|
| `instrument` | string | |
| `era` | string | `W1` \| `W2` \| `W3` |
| `state` | string | |
| `share` | float | Fraction of bars; sums to 1.0 within an instrument × era |
| `median_dwell_bars` | int | Consecutive-run length |
| `median_dwell_hours` | float | Convenience — `median_dwell_bars × 0.25` at 15m |
| `n_bars` | int | Sample size |
| `elimination_vote` | string | `KEEP` \| `DROP_DWELL` \| `DROP_UNSTABLE` |

**Contract**: `median_dwell_hours` is compared against the 25th percentile of the **host's own** measured trade duration. Rare states — below a stated minimum share — carry their `n_bars` and are excluded from verdict-driving comparisons.

---

## `risk_normalized.csv`

The Phase B re-analysis of the prior taxonomy's per-state risk.

| Column | Type | Notes |
|---|---|---|
| `state` | string | The five exp4 states |
| `pct_time` | float | Reproduced from the original for comparison |
| `maxdd_raw_pct` | float | The original, unnormalized figure |
| `maxdd_per_unit_time` | float | Normalized for time-in-state |
| `n_bars` | int | |
| `verdict` | string | `CONFIRMED` \| `REFUTED` |

**Contract**: the original figures showed a roughly 4× spread (28.1% in the best state versus 118.3% in the worst) while the worst state also held 1.55× the bars. This artifact exists to determine whether the spread survives normalization. A `REFUTED` verdict removes the primary motivation for treating phase as a risk-modulation signal and must be stated plainly rather than buried.

---

## `trackability.csv`

The Phase C evidence table. One row per instrument per window.

| Column | Type | Notes |
|---|---|---|
| `instrument` | string | Must be `SURVIVING` |
| `window` | string | `W1` \| `W2` \| `W3` |
| `horizon` | string | Decision-block length — `14d` \| `30d` \| `60d` |
| `n_blocks_qualifying` | int | Blocks scored |
| `n_blocks_excluded` | int | Blocks dropped for too few trades — reported, never scored as ties |
| `phase_choice_error` | float | Mean regret of the phase rule |
| `error_always_aggressive` | float | Mean regret of always choosing aggressive |
| `error_always_conservative` | float | Mean regret of always choosing conservative |
| `constant_error` | float | **`min` of the two above** — the bar |
| `status_quo_error` | float | Mean regret of the tuned configuration — **reference only, not the bar** |
| `beats_constant` | bool | `phase_choice_error < constant_error` |
| `margin` | float | `constant_error − phase_choice_error`; must exceed the paired-seed noise band |
| `n_trades` | int | Realized trades backing the row |
| `source` | string | Path to the trade export — **never** a price series |

**Contract**

- `source` is recorded so the prohibition in FR-013 is auditable after the fact.
- A row with `n_trades = 0` cannot support a `PASS`.
- `constant_error` is the **better fixed choice**, not `status_quo_error`. Both columns exist precisely so the distinction cannot be lost: a rule that beats the status quo while losing to the better fixed choice has demonstrated threshold tuning, not trackability, and must not be read as a PASS (research R12, FR-018a).
- `n_blocks_excluded` is reported so that a verdict resting on a handful of qualifying blocks is visible rather than implied.

---

## `VERDICT.md`

The single go/no-go, human-readable.

Required content: the verdict line (`PASS <instrument> <level> H=<n>` or `FAIL`), the full `trackability.csv` rendered as a table, the eliminated-instrument list with reasons, and — on FAIL — a short statement of what was measured and why it does not support a gate.

**Contract**: on FAIL this file is the feature's deliverable. It is committed, and it must contain enough detail that the negative result is reproducible and nobody re-proposes the idea without re-running it. This mirrors the docstring convention already used by the tool that killed the dynamic lookback.

---

## `arms/<tag>_<window>.json`

One per mechanism arm. Conforms to the `MechanismArm` entity.

```json
{
  "mechanism": "M3",
  "tag": "m3_phase_thr",
  "window": "W1",
  "timerange": "20250901-20260831",
  "strategy": "NNNC_MLX_PhaseThr",
  "baseline_tag": "baseline",
  "model_pass_count": 0,
  "chain_pass_count": 0,
  "n_trades": 0,
  "profit_pct": 0.0,
  "risk_adjusted": 0.0,
  "max_drawdown_pct": 0.0,
  "liveness": "LIVE",
  "seed_paired": false,
  "exit_config_hash": "",
  "volume_filter_active": true
}
```

**Contract**

- `liveness` is `INERT` when `chain_pass_count` equals the baseline arm's for the same window. An `INERT` arm's `profit_pct` and `risk_adjusted` are withheld from every comparison table — reporting them as a null result is the specific failure Principle IV forbids.
- `exit_config_hash` must match the baseline's for the same window, proving exit invariance (SC-010).
- `volume_filter_active` must be `true`; an arm with it `false` is void.
- `timerange` must be a literal date range. A today-relative range makes the arm non-comparable and is void.
- `seed_paired` must be `true` before accepting any delta under one percentage point.
