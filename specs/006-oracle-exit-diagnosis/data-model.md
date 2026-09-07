# Data Model — 006 Oracle-Instrumented Exit Diagnosis

Six entities. No database; everything is JSON arm records, backtest archives and
written findings.

## 1. InstrumentCompliance

Whether the perfect-entry arm is actually doing what it claims. Gates the feature.

| field | notes |
|---|---|
| `entries_checked` | trades in the perfect-entry arm |
| `entries_meeting_label` | how many satisfy the label on the strategy's own data |
| `compliance_rate` | currently **84.1%** measured against raw candles |
| `reference_bar` | entry / signal / entry−2 — entry is the best alignment (84.1 / 81.4 / 64.1) |
| `cause` | named cause, or NULL |
| `residual_bound` | the error every downstream figure inherits if no cause is found |

**Validation**: the feature does not proceed past Phase 1 until `cause` is
populated or `residual_bound` is stated. A shortfall is never set aside.

## 2. PayoffGeometry

The reward:risk arithmetic per family. Cheapest decisive measurement (R3).

| field | NNMT | NNNC |
|---|---|---|
| `label_target` | 2.0% | 2.0% |
| `harvest_point` (take-profit) | 2.2% | 2.4% |
| `stop_band` | −2% … −4% (mult 2.5) | −4% … −6% (mult 4.0) |
| `roi` | 3% | 5% |
| `reward_risk` | **1:1 … 1:2** | 1:2 … 1:3 |
| `stop_breach_rate` | recompute per family with the CORRECT band |

**Validation**: the previously reported 6.6% breach rate used NNNC's band on NNMT
data and must be recomputed. A negative-expectancy geometry is a complete
explanation of the paradox and may end the feature at Phase 2.

## 3. ExitOpportunityCost

The bound. What perfect entries earn when the exit waits for the promised move.

| field | notes |
|---|---|
| `current_exit_return` | perfect entries under the shipped exit |
| `patient_exit_return` | perfect entries held to the label horizon |
| `opportunity_cost` | the difference — what the exit leaves on the table |
| `resolution` | paired-by-trade 2sd for the same comparison |

**Validation**: if `opportunity_cost` is not materially larger than `resolution`,
the exit is not the constraint and Phases 3–5 are re-scoped (FR-005). The patient
arm is a BOUND and is never promotable.

## 4. MechanismContribution

One exit mechanism's individual effect, from re-enabling it alone.

| field | notes |
|---|---|
| `mechanism` | ATR stop / ROI / take-profit / unclog / max-hold |
| `delta_return`, `delta_exit_mix` | per window |
| `inert` | true when neither trades nor return change at all |
| `sums_to_joint` | whether individual effects add to the joint bound |

**Validation**: individual effects are compared against the joint effect so
interaction is visible. On the entry side the equivalent measurement was 4.20×
super-additive, so addition must not be assumed here either.

## 5. Arm

One evaluated configuration, as `regime/arms/<tag>_<window>.json`.

| field | notes |
|---|---|
| `tag`, `window`, `strategy` | |
| `n_trades`, `trades_hash`, `exit_config_hash` | liveness and identity |
| `wallet_profit_pct`, `max_drawdown_pct`, `profit_factor`, `win_rate` | |
| `returncode` | must be 0 or the record is not evidence |
| `net_gross_ratio` | **required for every window** (R7) — trade count alone is not enough |
| `is_lookahead` | true for perfect-entry arms; forces the ceiling labelling |
| `pairing` | `by_trade` when entries are shared, `by_membership` when they are not |

**Validation**: identical trades to its control is INCONCLUSIVE, never refuted.
Zero trades is inconclusive, never a zero return.

## 6. PromotionCandidate

Anything that survives the instrument and is put to the real model.

| field | notes |
|---|---|
| `change` | the single mechanism varied |
| `instrument_effect` | effect under perfect entries — never sufficient |
| `real_model_effect` | effect on the real model, two seeds, paired, W1 |
| `verdict` | promote / instrument-only / reject |
| `capacity_curve` | required before promotion (FR-013) |

**Validation**: `verdict` is decided by `real_model_effect` alone. A patient exit
is rewarded on an instrument where every entry is guaranteed to reach its target
and punished on a real model whose entries often do not — this is the expected
failure mode (R8), not an anomaly.

## Relationships

    InstrumentCompliance  1───1 (gates) the entire feature
    PayoffGeometry        1───2 family                (measured before any arm)
    ExitOpportunityCost   1───2 Arm                   (current vs patient)
    MechanismContribution 1───1 Arm                   (one re-enable arm each)
    PromotionCandidate    1───* Arm                   (instrument + real model x seeds)
