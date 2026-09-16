# Phase 1 Data Model: signal-context-timing

Entities, fields and invariants. Invariants marked **(test)** have a named test.

---

## Signal

One entry **event** — a contiguous run of `enter_long` bars collapsed to its first bar (R2).

| field | type | notes |
|---|---|---|
| `fold_id`, `pair` | str | |
| `timestamp` | datetime64[ns, UTC] | first bar of the run |
| `run_bars` | int | length of the run; >1 means the signal persisted |
| `admitted` | bool | did freqtrade open a trade from it |
| `masked_reason` | enum | `none` \| `pair_has_open_trade` \| `max_open_trades` |

**Invariants**

- Every **recorded trade's** entry bar appears as a Signal. Measured f08: **70/71**; the miss is
  the test range's first bar. A fidelity rate below 0.95 voids everything downstream. **(test)**
- Runs are collapsed: two adjacent signal bars are ONE Signal. **(test)**
- `admitted` is derived from the trade archive, never assumed from the gate.

---

## SignalContext

Causal facts known at signal time (FR-004).

| field | type | notes |
|---|---|---|
| `position_open` | bool | the pair already holds a trade → this is a **repeat** signal |
| `bars_since_prev_signal` | int | on the same pair; null for the first ever |
| `prev_signal_won` | bool \| null | outcome of the previous **admitted** signal on the pair |
| `drawdown_from_high` | float | pair's close vs its trailing high |
| `burst_position` | int | index within a cluster of signals on the pair |

**Invariants**

- Every field uses bars at or before `timestamp`. Demonstrated by truncation, not asserted.
  **(test: test_signalctx_context)**
- `prev_signal_won` uses only signals whose trade **closed** before `timestamp`; a trade still
  open at signal time has no outcome yet and MUST be null. **(test)** This is the most likely
  place for a leak to enter, because the obvious implementation joins on the previous signal
  regardless of whether its result was knowable.
- Exactly **four** hypotheses are tested (R5); the set is fixed before any test runs.

---

## TradePath

Per-bar unrealised P&L for an open trade, reconstructed because the archive omits it (R4).

| field | type | notes |
|---|---|---|
| `trade_id`, `pair`, `fold_id` | | |
| `bar` | datetime64 | within `[open_date, close_date]` |
| `holding_age_bars` | int | |
| `unrealised_pp` | float | additive percentage points against entry |
| `mark` | enum | `close` (default) or `low` for stop-proximity questions |

**Invariants**

- Additive in pp, never compounded across concurrent trades (brief 013's recorded inversion).
  **(test: test_signalctx_paths)**
- `unrealised_pp` at `close_date` reconciles with the archive's recorded `profit_ratio` to
  within **`RECONCILE_TOL = 0.002`** (about twice a round-trip fee). **(test)** — the cheap
  check that the reconstruction is right at all.
- A claim about stop proximity MUST use `low` and MUST say so.

---

## ConfoundResult

US1's output — the gate everything else waits on.

| field | type | notes |
|---|---|---|
| `n_repeat`, `n_control` | int | |
| `strata` | list | decile bins of `unrealised_pp` at signal time |
| `cmh_statistic`, `cmh_p` | float | primary, stratified |
| `logit_coef`, `logit_ci`, `logit_p` | float | secondary |
| `effect_winrate_pp` | float | win-rate difference, in **win-rate** points — reported, never gated |
| `effect_return_pp_yr` | float | `Δ mean profit_ratio × 9.0 × trades/yr` — the **only** quantity the 7.633 floor applies to |
| `verdict` | enum | `survives` \| `proxy_for_drawdown` |
| `exploratory` | bool | persisted here so `sweep.py` and `report.py` read it from disk rather than inheriting an in-memory flag |

**Invariants**

- Controls are positions **alive at the same holding age** (FR-002). **(test)**
- `verdict = proxy_for_drawdown` stops the repeat-signal arm (FR-003) and flips US3 to
  EXPLORATORY (FR-023).
- A significant result whose **`effect_return_pp_yr`** is inside 7.633 is reported **not
  resolved**, not a finding (SC-015, Principle III).
- `effect_winrate_pp` MUST NOT be compared to 7.633. The two are different currencies, and the
  motivating 14.8-point win-rate effect would clear that floor while saying nothing about
  return. **(test: a guard that raises if the win-rate figure reaches the gate)**

---

## MaskedOutcome

What a masked signal would have done, observed only under `--eps` (R6).

| field | type | notes |
|---|---|---|
| `signal` | Signal | must have `masked_reason != none` |
| `profit_ratio` | float | from the stacked run |
| `label` | const | `"not reproducible in dry/live"` — **mandatory on every row** |

**Invariants**

- No aggregate from this entity may be reported as an edge, a delta, or a candidate result
  (FR-009). **(test: a guard that raises if a MaskedOutcome reaches the report's P&L path)**
- Comparisons acknowledge that stacking gives independent stops, which a deployed change would
  not (FR-010).

---

## ExitRuleArm

US2/US5 — an evaluated arm.

| field | type | notes |
|---|---|---|
| `arm` | enum | `control` \| `exit_on_repeat` \| `tighten_on_repeat` |
| `fold_id` | str | |
| `model_path` | str | the **ungated** fold's weights |
| `n_triggered` | int | trades the rule acted on, per fold |
| `return_pp`, `max_dd_pp` | float | both mandatory (FR-021) |
| `overlap_with_existing_stop` | float | share of triggers where the existing stack fired anyway within 8 bars; **≥ 0.70 is inert** |
| `exploratory` | bool | set when T023 killed the repeat-signal arm; stamped on every downstream row |

**Invariants**

- `model_path` resolves to existing `*.safetensors` under the ungated fold's directory, and no
  new `saved_data/` directory is created. **(test)**
- Entry behaviour is **byte-identical** to the control (FR-020). **(test)**
- `n_triggered == 0` → inert; no performance figure may be computed (SC-020).
- An arm reported as affecting nothing MUST be byte-identical to its control — the null-case
  check that catches a broken liveness probe (R9). **(test)**
- `overlap_with_existing_stop >= 0.70` → inertness under Principle IV, not a result (SC-019).
- The stop-tightening factor is **0.5, frozen**; an outright exit is a labelled sensitivity that
  cannot produce the verdict (FR-019a).
