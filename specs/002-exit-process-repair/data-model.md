# Phase 1 Data Model: Exit-Process Characterisation and Repair

**Feature**: 002-exit-process-repair | **Date**: 2026-09-02

Four entities, from the spec's Key Entities section. Validation rules trace to
functional requirements.

---

## 1. BaselineConfig

The exit configuration under which the 167 characterised trades were produced.
**Pinned in code, never read from the strategy JSON** (FR-028b).

| Field | Value | Notes |
|---|---|---|
| `grace_hours` | 5.6 | The one live grace dimension |
| `grace_level` | −0.27 | **INERT** — recorded for completeness only |
| `take_profit` | 0.024 | Live |
| `max_days` | 10 | Live |
| `roi` | `{"0": 0.03}` | Live; 22% of exits land here exactly |
| `stoploss` | −0.05 | The post-grace anchored level |
| `exit_signal_enabled` | False | Model contributes nothing to exits |
| `archives` | 3 paths | The recorded backtest results per window |

**Validation**

- MUST be a literal in source, not read from `NNNC/NNNC_MLX.json`. That file
  already drifted once mid-feature and no longer holds these values.
- Each archive path MUST resolve to an existing result, and the trade counts
  MUST be 51 / 96 / 20 for W1 / W2 / W3 (167 total). A mismatch means the
  baseline is not what characterisation assumed — halt rather than proceed.
- `grace_level` MUST be marked inert wherever it is displayed, so no reader
  mistakes it for a tunable.

---

## 2. StoppedTrade

One recorded trade that ended at the stop.

| Field | Type | Notes |
|---|---|---|
| `pair`, `window` | string | |
| `open_date`, `close_date` | tz-aware UTC | |
| `open_rate`, `close_rate` | float | |
| `profit_ratio` | float | Realized outcome |
| `age_h` | float | Median 1.00h; p10 = 0.00h |
| `initial_stop_pct` | float | −5.00% for every baseline trade |
| `final_stop_pct` | float | Relative to entry; **positive on 50.6%** of stopped trades |
| `stop_above_exit` | bool | True for 18.1% |
| `ended_in_profit` | bool | True for 37.4% — the population split |

**Validation**

- `ended_in_profit` MUST partition every report (FR-004, SC-003). The profitable
  and losing stops are distinct phenomena and may want opposite remedies.
- `initial_stop_pct` MUST equal −5.00% for baseline trades. Any other value means
  the grace level became reachable, which would invalidate research R1.
- Trades are identified by `(pair, open_date)`, the same key feature 001 used.

---

## 3. CounterfactualOutcome

One stopped trade replayed over one holding extension.

| Field | Type | Notes |
|---|---|---|
| `trade_key` | (pair, open_date) | FK |
| `extension_h` | float | ~1 / 3 / 6 / 12 / 24 |
| `best_pct`, `worst_pct` | float | Max favourable / adverse excursion after the exit |
| `would_reach_take_profit` | bool | |
| `would_reach_roi` | bool | The +3.00% level |
| `sufficient_history` | bool | False near a window's end |

**Validation**

- A trade with `sufficient_history = False` MUST be excluded from that extension
  and counted, never treated as "did not recover" (FR-008, edge case).
- Replay MUST use only recorded trades and price history — no model load, no
  training (FR-007).
- Excursions MUST be measured from the actual exit price, so the counterfactual
  answers "should this trade have been held?" rather than a different question.

---

## 4. CandidateIntervention

A proposed change to exit behaviour.

| Field | Type | Notes |
|---|---|---|
| `name` | string | |
| `motivating_measurement` | string | **Required** (FR-009) — a candidate without one is not listed |
| `kind` | enum | `adjust_existing` \| `enable_dormant` (FR-012) |
| `requires_retrain` | bool | MUST be False (FR-010, FR-025) |
| `screen_delta` | float, nullable | Change vs baseline under the counterfactual screen |
| `screen_verdict` | enum | `PASS` \| `REJECTED` |
| `arms` | list | Populated only for screen survivors |
| `excluded_reason` | string, nullable | For candidates research already killed |

**Validation**

- `requires_retrain = True` is disqualifying; such a candidate is not listed.
- A candidate research already eliminated MUST appear with `excluded_reason`
  rather than being omitted (FR-011). Four are pre-populated as excluded:
  grace level (inert), the `return 1.0` correction (refuted, −9.24pp),
  `trailing_*` (inert), the hyperopt exit config (rejected on a W3 sign reversal).
- `screen_verdict = REJECTED` MUST prevent any arm being created (FR-016).
- `kind = enable_dormant` MUST be flagged as higher risk than `adjust_existing`.

---

## 5. EvaluationArm

One candidate measured over one window. Created only for screen survivors.

| Field | Type | Notes |
|---|---|---|
| `candidate`, `window` | string | |
| `timerange` | string | Literal `YYYYMMDD-YYYYMMDD` |
| `n_trades` | int | |
| `exit_mix` | dict | Count and total P&L per exit reason |
| `wallet_profit_pct`, `max_drawdown_pct` | float | |
| `profit_factor`, `calmar`, `sortino` | float | |
| `trades_hash` | string | Inertness check |
| `liveness` | enum | `LIVE` \| `INERT` |
| `entry_config_hash` | string | MUST match baseline (FR-024) |

**Validation**

- `liveness = INERT` when `trades_hash` equals the baseline's for that window.
  An INERT arm's figures are withheld from every comparison (FR-023).
- `entry_config_hash` MUST match baseline — entry is frozen by feature 001.
- Accepted only if risk-adjusted return improves without reversing sign in W3
  (FR-021, SC-009).
- `n_trades = 0` or very few ⇒ inconclusive, never a null (FR-022).
- `exit_mix` is reported so category-shifting is visible: converting stops into
  equally-bad time-based exits improves the mix without improving the outcome.

---

## Relationships

```
BaselineConfig 1──n StoppedTrade          (83 of 167 baseline trades)
StoppedTrade   1──n CounterfactualOutcome (one per holding extension)
CandidateIntervention 0..n EvaluationArm  (created ONLY on screen PASS)
```

The `CandidateIntervention → EvaluationArm` edge is the cost gate: a candidate
rejected at the screen never consumes a backtest.
