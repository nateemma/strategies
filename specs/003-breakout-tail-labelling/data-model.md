# Phase 1 Data Model — Tail-Targeted Breakout Labelling

Five entities. Validation rules trace to the FR that requires them.

---

## 1. Candidate

A labelling configuration to be screened, admitted or excluded.

| Field | Type | Notes |
|---|---|---|
| `id` | str | short stable key, e.g. `L1_h48_t02` |
| `horizon` | int | label lookahead in 15m bars |
| `min_gain` | float | forward-gain threshold (buy); mirrored for sell |
| `basis` | enum | `guard_metric` (method 24) or `bb_position` (method 26) |
| `mae_bound` | float or None | optional adverse-excursion constraint (FR-006) |
| `requires_hold_change` | bool | derived: time-to-peak exceeds the forced-exit window |
| `motivating_measurement` | str | REQUIRED, non-empty (FR-008) |
| `excluded_reason` | str | non-empty ⇒ never trained |

**Validation**
- `motivating_measurement` MUST be non-empty (FR-008).
- A candidate MUST NOT be trained while `excluded_reason` is set (FR-002).
- At least one admitted candidate MUST have `requires_hold_change == False`
  (FR-004) so the label effect can be isolated.
- At least one admitted candidate MUST have `mae_bound` set (FR-006) and at least
  one MUST use the alternative `basis` (FR-007).
- `min_gain` interpretation is basis-independent, but a `bb_position` threshold
  MUST be expressed on the **[−1, 1]** scale, not [0, 1].

---

## 2. ScreeningRecord

The pre-training characterisation that admits or excludes a Candidate.

| Field | Type | Notes |
|---|---|---|
| `candidate_id` | str | FK → Candidate |
| `density_pct` | float | labelled bars as % of all bars |
| `median_gain_pct` | float | median labelled forward gain |
| `clears_stop_pct` | float | share of labelled bars reaching the operative stop magnitude |
| `reward_risk` | float | `median_gain / stop_magnitude` |
| `time_to_peak_h` | float | median hours from entry to the labelled peak |
| `stopped_first_pct` | float | share whose adverse excursion breaches the stop before the peak |
| `n_pairs` | int | pairs screened |

**Validation**
- All six measures MUST be present before the candidate is admitted (FR-001).
- `density_pct` below the all-Hold floor ⇒ set `excluded_reason` (FR-002).
- `time_to_peak_h` exceeding the forced-exit window ⇒ set
  `requires_hold_change = True`, which routes the candidate to the package phase.
- Produced without loading or training a model (FR-003).

---

## 3. Arm

A trained model plus its evaluation.

| Field | Type | Notes |
|---|---|---|
| `tag` | str | arm-record key |
| `candidate_id` | str | FK → Candidate |
| `strategy_class` | str | throwaway subclass on the host |
| `has_params_file` | bool | MUST be True (FR-016) |
| `exit_config_hash` | str | compared to control |
| `hold_window_changed` | bool | Phase 4 only |
| `results` | Window[3] | one per pinned window |
| `verdict` | enum | `improved` / `unchanged` / `worsened` / `inconclusive_absorbed` |

**Validation**
- `has_params_file` MUST be True before any result is read (FR-016).
- `exit_config_hash` MUST equal the control's, EXCEPT where the hold window is
  deliberately under test.
- `verdict` MUST be `inconclusive_absorbed` when W1 `trades_hash` equals the
  control's, regardless of the returns (FR-015). This outranks every other
  classification.
- `verdict` MUST NOT be derived from any classifier metric (FR-014).

---

## 4. WindowResult

One window's outcome for an Arm.

| Field | Type | Notes |
|---|---|---|
| `window` | enum | `W1` / `W2` / `W3` |
| `in_sample` | bool | True for W2 and W3 under the pinned training window |
| `n_trades` | int | reported always (FR-017) |
| `wallet_profit_pct` | float | realised return |
| `trades_hash` | str | liveness comparison |
| `max_drawdown_pct`, `profit_factor`, `win_rate` | float | context, never acceptance |

**Validation**
- Acceptance MUST read `W1` only; `in_sample` rows are reported and excluded
  (FR-013).
- A W1 delta below the power threshold MUST be reported as indistinguishable
  rather than as an improvement (FR-017, R9).

---

## 5. Control

The reference configuration every Arm is compared against.

| Field | Type | Notes |
|---|---|---|
| `tag` | str | re-baselined control |
| `params_file` | path | NEW — pins the effective configuration (R8) |
| `exit_config_hash` | str | the value all non-hold-window arms must match |
| `results` | Window[3] | control returns and hashes |
| `reproduces_parked_hashes` | bool | MUST be True |

**Validation**
- `reproduces_parked_hashes` MUST be True before any Arm is compared. If pinning
  the configuration changes behaviour, the discrepancy is resolved first (R8).
- The control MUST carry a params file, so that arms and control differ only in
  the parameter under test (FR-016).

---

## Relationships

```text
Candidate 1───1 ScreeningRecord        (admit / exclude)
Candidate 1───0..n Arm                 (only if admitted)
Arm       1───3 WindowResult
Arm       n───1 Control                (comparison target)
```

## State transitions — Candidate

```text
proposed ──screened──> admitted ──trained──> evaluated ──> {improved | unchanged |
                                                            worsened | inconclusive_absorbed}
         └─screened──> excluded  (density floor, or irreconcilable hold-window conflict)
```

`inconclusive_absorbed` is terminal for the arm but NOT a refutation of the
candidate — it says the gate prevented the test, not that the idea failed.
