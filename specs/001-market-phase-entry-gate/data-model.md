# Phase 1 Data Model: Market-Phase Framework and Phase-Adaptive Entry Gate

**Feature**: 001-market-phase-entry-gate | **Date**: 2026-09-02

Five entities, derived from the spec's Key Entities section. Validation rules trace to functional requirements.

---

## 1. PhaseInstrument

A named, causal recipe mapping price and volume history to a discrete state per bar.

| Field | Type | Notes |
|---|---|---|
| `name` | string | Unique key, e.g. `btc_exp4`, `asset_vr24`, `market_breadth` |
| `level` | enum | `asset` \| `market` \| `btc` (FR-001) |
| `states` | ordered list of string | The state vocabulary, always including `UNKNOWN` |
| `window_bars` | int | Lookback used by the estimator; fixed, never adaptive (Principle VII) |
| `params` | dict | Estimator parameters, e.g. `{q: [2,4,8,24]}` |
| `is_baseline` | bool | True only for the reproduced exp4 taxonomy (FR-004) |
| `status` | enum | `PENDING` \| `SURVIVING` \| `ELIMINATED` |
| `status_reason` | string | Required when `ELIMINATED`; must quote a measured value (FR-010, FR-011) |

**Validation**

- `name` unique across the catalogue.
- Exactly one instrument has `is_baseline = True`.
- `UNKNOWN ∈ states` for every instrument.
- `window_bars` must exceed the largest estimator horizon by at least 20× (R1).
- An instrument may not reach `SURVIVING` until it has passed lookahead validation (FR-007).
- `status_reason` non-empty whenever `status = ELIMINATED`; a bare "unstable" without a figure is invalid (FR-011).
- No instrument definition may reference a specific trading pair (FR-025, Principle X).

**State transitions**

```
PENDING ──lookahead fail──> ELIMINATED (reason: lookahead)
PENDING ──dwell < threshold──> ELIMINATED (reason: dwell, quoted)
PENDING ──unstable across eras──> ELIMINATED (reason: instability, quoted)
PENDING ──all checks pass──> SURVIVING
```

`ELIMINATED` is terminal. An instrument is never revived without re-running Phase A and B.

---

## 2. PhaseLabelSeries

The per-bar state assignment produced by one instrument for one asset.

| Field | Type | Notes |
|---|---|---|
| `instrument` | string | FK to `PhaseInstrument.name` |
| `pair` | string | The asset; `MARKET` for market-level, `BTC` for BTC-level |
| `date` | timestamp, tz-aware UTC | Bar timestamp; matches the OHLCV `date` column exactly |
| `state` | string | Member of the instrument's `states` |
| `coverage_n` | int, nullable | Market level only — qualifying pairs at this bar (R4) |

**Validation**

- Computed causally: the value at `date` uses only bars at or before `date` (FR-002).
- Identical under truncation: labels over `[start, T]` computed from history truncated at `T` equal labels computed from full history, at three cut points (FR-007, R9).
- Market level emits `UNKNOWN` when `coverage_n < 20` or fewer than 20 pairs have ≥200 bars of prior history (R4).
- BTC level emits its defined neutral state, not an exception, when BTC data is missing or stale (edge case; matches the existing fail-open merge behaviour).
- `date` is tz-aware UTC — the existing merge pattern joins on this column and a naive timestamp silently produces an all-NaN merge.
- `UNKNOWN` bars are excluded from every per-state statistic, never folded into an adjacent state.

---

## 3. ActionabilityReport

Per instrument, per era: the characterisation that drives elimination.

| Field | Type | Notes |
|---|---|---|
| `instrument` | string | FK |
| `era` | enum | `W1` \| `W2` \| `W3` (R3) |
| `state_share` | dict state→float | Fraction of bars per state; sums to 1.0 |
| `median_dwell_bars` | dict state→int | Consecutive-run length per state |
| `transition_matrix` | dict (state,state)→float | Row-stochastic |
| `n_bars` | int | Sample size backing the row |
| `elimination_vote` | enum | `KEEP` \| `DROP_DWELL` \| `DROP_UNSTABLE` |

**Validation**

- All three characteristics reported for all three eras; no instrument left uncharacterised (SC-002).
- `state_share` sums to 1.0 within tolerance; each transition-matrix row sums to 1.0.
- A state holding less than **5%** of the window's bars is reported with its sample count and excluded from verdict-driving comparisons (edge case: rare states).
- `DROP_DWELL` requires `median_dwell_bars` below the 25th percentile of measured host trade duration — a figure that must come from the host strategy, not a borrowed one (R6).
- `DROP_UNSTABLE` requires a quoted difference between the W1 and W3 rows, meeting either threshold in FR-011: largest absolute state-share change exceeding 15 percentage points, or the most-likely next state differing for at least half the states. W1-versus-W2 alone is insufficient: those windows are three months apart and share a regime (Principle II, R3).

---

## 4. TrackabilityVerdict

The feature's single go/no-go. Exactly one instance exists.

| Field | Type | Notes |
|---|---|---|
| `verdict` | enum | `PASS` \| `FAIL` (FR-016) |
| `winning_instrument` | string, nullable | Required on `PASS` |
| `winning_level` | enum, nullable | Required on `PASS` |
| `horizon_bars` | int, nullable | The forward horizon `H`; required on `PASS` |
| `comparisons` | list of `WindowComparison` | Evidence; one row per instrument × window |
| `decided_on` | date | |

**`WindowComparison`**

| Field | Type | Notes |
|---|---|---|
| `instrument` | string | |
| `window` | enum | `W1` \| `W2` \| `W3` |
| `horizon` | enum | Decision-block length — `14d` \| `30d` \| `60d` (research R12) |
| `phase_choice_error` | float | Mean regret of the phase rule over qualifying blocks |
| `error_always_aggressive` | float | Mean regret of the always-aggressive fixed choice |
| `error_always_conservative` | float | Mean regret of the always-conservative fixed choice |
| `constant_error` | float | `min` of the two fixed choices — **the bar** |
| `status_quo_error` | float | Mean regret of the tuned configuration — reference only |
| `beats_constant` | bool | `phase_choice_error < constant_error` |
| `margin` | float | `constant_error − phase_choice_error` |
| `n_blocks_qualifying` | int | Blocks scored |
| `n_blocks_excluded` | int | Blocks dropped for too few trades |
| `n_trades` | int | Realized trades backing the row |

**Validation**

- Every comparison derives from the host's realized trades; passive market, basket, or buy-and-hold returns are prohibited inputs (FR-013). This is the specific error that invalidated the prior BTC gate.
- Block winners are decided on risk-adjusted return, so a configuration raising return and drawdown proportionally does not register as a win (FR-018).
- `constant_error` MUST be the better of the two fixed choices, never `status_quo_error` (FR-018a). A rule that beats the status quo while losing to the better fixed choice has demonstrated threshold tuning, not trackability.
- Error is mean regret over qualifying blocks; a block where either arm has fewer than `N` trades is excluded and counted in `n_blocks_excluded`, never scored as a tie (FR-018b). `N` is derived at task T013 from the host's measured trade rate.
- `PASS` requires `beats_constant` in at least two windows, of which at least one pair must be temporally distant — W1 plus W2 alone cannot produce a `PASS` (FR-015).
- One window passing and another failing yields `FAIL`, never "mixed" (edge case).
- `winning_instrument` must have `status = SURVIVING`.
- On `FAIL`, no `MechanismArm` may be created (FR-017).
- A row with `n_trades = 0` is inconclusive and cannot support a `PASS` (FR-032).

---

## 5. MechanismArm

One phase-adaptive entry configuration measured against a common baseline. Exists only after a `PASS`.

| Field | Type | Notes |
|---|---|---|
| `mechanism` | enum | `M3` \| `M1` \| `M2` \| `M4` |
| `window` | enum | `W1` \| `W2` \| `W3` |
| `timerange` | string | Literal `YYYYMMDD-YYYYMMDD`; never today-relative (R8) |
| `model_pass_count` | int | Bars where the model condition passed |
| `chain_pass_count` | int | Bars where the full entry chain passed — the liveness measure |
| `n_trades` | int | |
| `profit_pct` | float | |
| `risk_adjusted` | float | Calmar |
| `max_drawdown_pct` | float | |
| `liveness` | enum | `LIVE` \| `INERT` |
| `seed_paired` | bool | Whether paired repeated runs were performed |

**Validation**

- `liveness = INERT` whenever `chain_pass_count` equals the baseline's. An `INERT` arm's trading figures are discarded and MUST NOT be recorded as a null result (FR-031, Principle IV).
- Exit configuration byte-identical to baseline (FR-024, SC-010).
- Run under active volume/liquidity constraints (FR-028).
- Accepted only when `risk_adjusted` improves without reversing sign in W3 (FR-027a, SC-009).
- Any accepted delta under one percentage point requires `seed_paired = True` (FR-030).
- `n_trades = 0` ⇒ inconclusive, not zero-return (FR-032).
- M4 additionally requires scalers regenerated before comparison (FR-022) and a phase-feature correlation check against the existing feature set beforehand (Principle IX).
- All arms for a given mechanism share one host strategy (FR-033).

---

## Entity relationships

```
PhaseInstrument 1──n PhaseLabelSeries      (one series per pair per instrument)
PhaseInstrument 1──3 ActionabilityReport   (one per era)
PhaseInstrument 0..1──1 TrackabilityVerdict (the winner, on PASS only)
TrackabilityVerdict 1──n WindowComparison  (evidence)
TrackabilityVerdict 1──n MechanismArm      (created only when verdict = PASS)
```

The `TrackabilityVerdict → MechanismArm` edge is the stop gate. `FAIL` means that edge has zero members, permanently, and the feature terminates with the negative result recorded.
