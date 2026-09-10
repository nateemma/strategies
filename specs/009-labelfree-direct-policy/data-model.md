# Phase 1 Data Model: Label-free Direct Policy Optimisation

**Feature**: 009-labelfree-direct-policy · **Date**: 2026-09-10

Entities are drawn from spec "Key Entities" and constrained by the Functional Requirements they
serve. Field types are logical, not storage-specific.

---

## 1. TrialRecord

One evaluated configuration. **Written before the run starts** (FR-019, SC-015); the result is
attached afterwards. Append-only JSONL, one object per line.

| Field | Type | Notes |
|---|---|---|
| `trial_id` | string | Stable unique id; also the filename stem for saved artefacts |
| `written_at` | timestamp | When the *pre-run* record was appended |
| `objective` | enum | `utility` \| `robust_sharpe` (FR-014) |
| `turnover_control` | enum | `band` \| `penalty` (FR-013) |
| `gamma` | float | Training cost scale, one of {0.5, 1.0, 1.5} (FR-004) |
| `lambda_risk` | float | Risk-aversion (utility) or turnover penalty weight |
| `architecture` | object | `{kind: lstm\|mamba, hidden: int, seq_len: int}` |
| `band_width` | float \| null | Null when `turnover_control = penalty` |
| `seed` | int | `MLX_SEED` value (R9) |
| `fold_id` | string | From the existing fold plan |
| `spread_estimator` | object | `{kind: abdi_ranaldo\|corwin_schultz, floor_bp: float}` (FR-004a) |
| `liquidity_cap` | object | `{fraction: 0.10, soft_min_tau: float}` (FR-005) |
| `leaky` | bool | True only for the deliberate-lookahead sanity run (FR-010) |
| `result` | object \| null | **Null at write time.** Populated post-run |

**Validation rules**
- `result` MUST be null when the record is first appended (SC-015). A record whose first written
  form already carries a result is an audit failure.
- `trial_id` MUST be unique across the file.
- Every configuration that touches any fold MUST appear, including abandoned and failed runs — the
  count used for DSR/PBO equals the number of records (FR-024, SC-015).
- The pinned trend-rule baseline contributes **no** records (FR-021a).

**State transitions**: `pre-written (result=null)` → `completed (result populated)` or
`abandoned (result = {status: abandoned, reason})`. Abandoned records are never deleted.

---

## 2. PolicyConfiguration

The reproducible description of a trained policy — sufficient for a fresh process to reproduce
identical trades (FR-026, SC-008).

| Field | Type | Notes |
|---|---|---|
| `trial_id` | string | Links to TrialRecord |
| `feature_list` | list[string] | The include list actually used |
| `scaler_ref` | string | Identity of the scaler set from `saved_data/` |
| `seq_len` | int | 16, matching the production classifier |
| `architecture` | object | As above |
| `objective`, `turnover_control`, `gamma`, `lambda_risk` | — | As above |
| `seed` | int | |
| `fold_id` | string | One policy per fold (FR-015) |
| `burn_in_bars` | int | 96 (R7) |
| `weights_ref` | string | Path to saved policy weights |

**Validation rules**
- `feature_list` MUST contain only pair-agnostic features — no raw price, raw volume, or any
  quantity proportional to price (FR-002).
- Reloading this configuration in a separate process MUST yield byte-identical trades (SC-008).
- If `feature_list` ever differs from the production include list, scalers MUST be regenerated
  before any comparison (constitution workflow gate 7).

---

## 3. SimulatedRun

Simulator output for one policy over one range (FR-007).

| Field | Type | Notes |
|---|---|---|
| `net_log_return` | series[bar] | Per-bar, net of fees/spread/slippage |
| `turnover` | series[bar] | \|Δw_exec\| per bar |
| `position` | series[bar, pair] | Realised exposure after the liquidity cap |
| `requested_delta` | series[bar, pair] | Pre-cap Δw, so cap-binding is measurable |
| `unfilled_carry` | series[bar, pair] | Remainder carried forward (FR-005) |
| `trades` | list[Trade] | Shaped like the backtest archive |

**Trade** — MUST match the archive shape so `regime/wf/pool.py` and `compare_pooled.py` work
unchanged: `pair`, `open_date`, `close_date`, `profit_abs` (plus `exit_reason` to support SC-018).

**Validation rules**
- A decision at bar `t` MUST use only bars `≤ t`; the fill MUST occur at `t+1` (FR-003).
- During calibration the fill MUST be the following bar's **open**, not a VWAP proxy (FR-003).
- `position` MUST lie in [0, 1] per pair (long-only spot).
- Concurrent open positions MUST NOT exceed `max_open_trades` (10); stake and
  `tradable_balance_ratio` (0.99) as configured (FR-006).
- `exit_reason` MUST distinguish band-driven exits from deployment-stack exits, so SC-018's share
  can be computed.

---

## 4. Fold

One anchored train/test split from the existing plan (FR-015). Read-only for this feature.

| Field | Type | Notes |
|---|---|---|
| `fold_id` | string | |
| `train_start`, `train_end` | date | ≥730 days |
| `test_start`, `test_end` | date | 90 days |
| `purge`, `embargo` | duration | Applied when training uses any forward-looking quantity (FR-018) |
| `pair_universe` | list[string] | Pairs with sufficient history **for this fold** |

**Validation rules**
- `pair_universe` MUST be recorded per fold and per window, and the control MUST be evaluated on
  the identical universe within that window (FR-015a, SC-004).
- Results MUST NOT be compared across windows whose universes differ (FR-015a).

---

## 5. CalibrationReport

Evidence that the simulator matches the arbiter (FR-008, SC-001). Blocking.

| Field | Type | Notes |
|---|---|---|
| `window` | string | W1 (`20250901`–`20260831`) |
| `frozen_signal_ref` | string | The arbiter run whose realised trades are replayed |
| `n_trades_arbiter`, `n_trades_sim` | int | MUST be equal |
| `total_return_arbiter`, `total_return_sim` | float | MUST agree within 0.10pp |
| `max_per_trade_diff_bp` | float | MUST be ≤ 1.0 |
| `worst_trades` | list | The largest disagreements, for diagnosis |
| `component_attribution` | object | Which of fee / fill timing / stake sizing explains the residual (R11, Principle VI) |
| `attempt_number` | int | Against the budget of 5 (SC-012) |
| `verdict` | enum | `pass` \| `fail` |

---

## 6. ResultReport

The study's standard per-brief write-up (protocol §6, FR-033).

Required content: hypothesis; config path and seed list; trials logged (n); pooled walk-forward
return per year with CI, folds positive, trades/yr, mean hold time; the four-baseline table; cost
stress (1.5×, 2× fees, 5% cap) with the fee level where the edge crosses zero; PBO and DSR;
simulator-vs-arbiter reconciliation (SC-018); per-window pair-universe sizes (FR-015a); verdict
(`accept` \| `not resolved` \| `kill`); what changes for the next brief.

**Validation rules**
- A delta below 3.35pp/yr MUST read "not resolved", never "no effect" (FR-025, SC-009).
- Written even when the verdict is kill or not-resolved (SC-009, US5).
- MUST link the feature directory and both `/speckit-analyze` reports (FR-032, SC-017).
