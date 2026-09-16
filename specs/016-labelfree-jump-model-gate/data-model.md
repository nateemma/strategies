# Phase 1 Data Model: labelfree-jump-model-gate

Entities, their fields, and the invariants that must hold. Invariants marked **(test)** have a
named test in `research/labelfree/tests/`.

---

## FeaturePanel

Causal inputs to the jump model, per pair and for BTC.

| field | type | notes |
|---|---|---|
| `pair` | str | or the literal `BTC` for the market-wide block |
| `timestamp` | datetime64[ns, UTC] | 15-minute closed bars |
| `rvol_realised` | float | trailing realised volatility |
| `rvol_downside` | float | trailing semi-volatility |
| `sign_persistence` | float | trailing return-sign persistence |

**Invariants**

- Every value at `t` uses only bars at or before `t`. **(test: test_jump_causality)**
- Standardisation statistics come from the expanding window **up to the current refit point
  only**, never the full series. **(test: test_jump_causality)**
- No NaN reaches the model; warm-up bars are dropped, never forward-filled from the future.

---

## RegimeFit

One fit, produced at a refit point.

| field | type | notes |
|---|---|---|
| `refit_at` | datetime64 | the refit point |
| `train_end` | datetime64 | strictly `< refit_at` |
| `centroids` | ndarray (K, d) | `μ` |
| `weights` | ndarray (d,) | `w`, on the simplex under an L1 budget |
| `scaler_mean`, `scaler_scale` | ndarray (d,) | fitted on the expanding window only |
| `lam` | float | frozen; identical across every fit in the run |
| `seed` | int | k-means initialisation seed; **the fit is stochastic in this** |
| `k` | int | 2 (primary) |
| `state_order` | tuple[int, ...] | state indices sorted by fitted realised-vol centroid |

**Invariants**

- `train_end < refit_at`; no bar at or after `refit_at` enters the fit. **(test)**
- `lam` and `k` are identical across every `RegimeFit` in a run — a per-fold or per-refit value
  is a defect, not a tuning choice. **(test)**
- `state_order` is recomputed at every refit, so "the high-vol state" always names the same
  economic thing even when the fitter permutes its labels. **(test: test_jump_model)**
- `fit()` is deterministic for a fixed `seed` and **not** deterministic across seeds — see
  `SeedEnsemble`. **(test: test_jump_model)**

---

## SeedEnsemble

The ≥5-seed run required by `00-evaluation-protocol.md` §4 (FR-022). Exists because the backtests
are deterministic while the fit that drives them is not.

| field | type | notes |
|---|---|---|
| `seeds` | tuple[int, ...] | at least 5 |
| `state_agreement` | float | pairwise state-for-state agreement across seeds |
| `block_rate_iqm` | float | interquartile mean of the per-fold block rate |
| `block_rate_ci` | tuple[float, float] | stratified bootstrap across seeds |
| `across_seed_spread` | float | spread of the block rate across seeds |
| `gated_vs_ungated_effect` | float | the effect the feature is trying to measure |

**Invariants**

- `len(seeds) >= 5`; the reported figure is the IQM, never a single seed and never the best one.
- `state_agreement >= 0.95`, else the regime structure is an initialisation artifact and the
  feature stops. **(test)**
- `across_seed_spread < gated_vs_ungated_effect`. If seed noise exceeds the effect, the gate is
  initialisation noise and the result is a **KILL**, not a finding (SC-017). **(test)**

---

## StateSequence

The artefact the backtest consumes. **This** is the object the causality test targets.

| field | type | notes |
|---|---|---|
| `pair` | str | |
| `timestamp` | datetime64 | |
| `state` | int8 | in `[0, k)`, already volatility-ordered |
| `mode` | enum | `filtered` \| `smoothed` |
| `refit_id` | int | which `RegimeFit` produced it |

**Invariants**

- Produced from the `SeedEnsemble`, not from one fit.
- `mode == filtered` for every deployed run. A `smoothed` sequence may only ever reach the
  leak-reference arm. **(test: test_jump_causality)**
- Truncating input history at any cut point leaves `filtered` states over the overlapping region
  **bit-identical**. **(test: test_jump_causality)**
- The same truncation applied to `smoothed` states **must change them** — a smoothed sequence that
  passes the causality test means the traceback is not running and the leak reference is not a
  leak reference. **(test: test_jump_causality)**

---

## GateDecision

Per candidate entry, what the gate did.

| field | type | notes |
|---|---|---|
| `pair`, `timestamp` | | |
| `state` | int8 | |
| `blocked` | bool | true when the state is the high-vol state |
| `rvol_would_block` | bool | whether the incumbent's rvol ≥ 2.0 guard already rejects this bar |

**Invariants**

- `blocked` depends only on `state`; no other condition may enter the gate. **(test: test_jump_gate)**
- Blocking counts are reported **per fold**, never only pooled. **(test: test_jump_gate)**

---

## GateActivity

The Principle IV liveness record, computed before any performance number is read.

| field | type | notes |
|---|---|---|
| `fold_id` | str | |
| `n_candidates` | int | entries the ungated incumbent would take |
| `n_blocked` | int | |
| `block_rate` | float | |
| `novel_block_rate` | float | share of blocks **not** already rejected by rvol (Principle IX) |
| `seed_block_rate_ci` | tuple[float, float] | across-seed interval on `block_rate` (FR-022) |
| `verdict` | enum | `active` \| `inert` \| `total` |

**Invariants**

- `n_blocked == 0` → `inert`. No return or drawdown comparison may be computed from an inert
  gate. **(test: test_jump_gate)**
- `n_blocked == n_candidates` → `total`. Refused by `admissibility.py`; abstention is not risk
  reduction. **(test: test_jump_gate)**
- A gate `active` in fewer than 10 of 15 folds cannot support a pooled drawdown claim (SC-020).
- `novel_block_rate < 0.20` → `redundant_with_rvol`: inert because rvol got there first, reported
  as a defect under Principle IV (FR-024, SC-019). **(test: test_jump_gate)**

---

## GatedArm

One evaluated arm.

| field | type | notes |
|---|---|---|
| `arm` | enum | `ungated` \| `gated_online` \| `gated_smoothed` |
| `fold_id` | str | |
| `strategy_class` | str | generated fold class |
| `model_path` | str | **points at the ungated fold's weights** |
| `trades` | list[Trade] | from the backtest archive |
| `return_pp` | float | additive fixed-stake |

**Invariants**

- `model_path` resolves to an existing `*.safetensors` under the **ungated** fold's directory, and
  the gated run creates **no new** `saved_data/` directory. **(test)** Without this the class
  trains inside its own backtest, silently, with returncode 0.
- `ungated` is read from the existing `evaluation/pooled_nnnc.json` and is never re-run.
- A zero-trade arm is `inconclusive`, never a zero return.

---

## DrawdownStat

| field | type | notes |
|---|---|---|
| `fold_id` | str | |
| `arm` | enum | |
| `max_dd_pp` | float | peak-to-trough of the additive cumulative-pp curve |
| `time_underwater` | float | fraction of the fold span below the running max |
| `ci_low`, `ci_high` | float | per-fold stationary block bootstrap |
| `block_len_trades` | int | **in trades, not bars** |

**Invariants**

- The equity curve is **additive** in percentage points (`profit_abs × scale_pp_per_abs`), never
  compounded — compounding treats concurrent positions as sequential full-wallet bets.
  **(test: test_jump_drawdown)**
- `block_len_trades` is chosen in trade units and recorded; the bar-panel constants
  `BLOCK_LENGTHS = (96, …, 288)` are 15-minute-bar lengths and MUST NOT be inherited.
  **(test: test_jump_drawdown)**
- CIs are per fold, because SC-011 counts folds whose CI excludes zero.

---

## LeakReference

| field | type | notes |
|---|---|---|
| `online_result` | GatedArm[] | the honest arm |
| `smoothed_result` | GatedArm[] | the deliberate leak |
| `gap_pp` | float | **the published size of the avoided leak** |
| `verdict` | enum | `leak_measured` \| `suspect_online_leaks` |

**Invariants**

- Produced **first**, before the online arm's numbers are read.
- `gap_pp < 7.633` (the pinned noise floor) → `suspect_online_leaks`, and the feature **stops**
  (FR-009 / SC-014). It is not a reassuring result.

---

## CrossCheck

| field | type | notes |
|---|---|---|
| `agreement_rate` | float | in-repo vs `jumpmodels 0.1.1` state sequences |
| `package_mode_audited` | bool | confirmed **filtered**, not smoothed |
| `interpreter` | str | must be `freqtrade/.venv/bin/python` |

**Invariants**

- `agreement_rate >= 0.95`, else the feature stops (FR-016).
- `package_mode_audited` must be true before the agreement rate means anything. A package running
  a Viterbi traceback would agree with our *smoothed* reference and disagree with our honest one,
  which reads as a bug in our code.
- Material disagreement **stops** the feature; it is never averaged or reported as a range.
