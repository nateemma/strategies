# Specification Quality Checklist: Market-Phase Framework and Phase-Adaptive Entry Gate

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-02
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Validation Log

**Iteration 1 (2026-09-02)**

Issues found and fixed before this pass:

1. *Implementation leakage* — the first draft named source files, function names, and parameter identifiers (`entry_guard_threshold`, `BtcRegimeGate.py`, `rolling_optimal_lookback.py`) inside functional requirements. Rewritten to describe capabilities: "the dip-forcing entry guard thresholds", "the prior BTC taxonomy and its existing gate scaffolding". Concrete code references now live only in the planning phase, not the spec.
2. *Unmeasurable success criteria* — "the framework is useful" style outcomes replaced with countable ones (zero label differences, three eras characterized, every instrument carrying a status with a quoted reason).
3. *Missing negative-outcome path* — SC-006 added so that a FAIL trackability verdict is explicitly recorded as a successful feature outcome, not a failure to deliver. Without this the spec would have implied the gate must ship.
4. *Inert-parameter blind spot* — FR-031 and Edge Case "a mechanism silently does nothing" added, because this repository has five recorded cases of parameters that were confirmed loaded yet changed nothing, each of which risked being read as a genuine null result.
5. *Missing coverage rule* — market-breadth measures are undefined in early history when few pairs were listed. Edge case plus an UNKNOWN state added.

**Iteration 2 (2026-09-02) — clarifications resolved**

Both outstanding questions were answered by the user and folded in:

1. *Q1, what the phase label predicts* (was blocking FR-018) — resolved as the forward relative performance of two configurations of the same reversion leg, aggressive versus conservative at entry, scored against always choosing one fixed configuration, with the winner decided on risk-adjusted return. FR-018 rewritten; User Story 2 narrative, Independent Test, and acceptance scenarios updated to the two-configuration framing; new acceptance scenario added so that proportional exposure changes cannot register as wins.
2. *Q2, host strategy* — resolved as the non-GAN classifier base. Captured as a Host strategy assumption, plus FR-033 requiring a single shared host across the trackability measurement and every mechanism arm, so no finding is confounded by a host change mid-feature.

Open Questions section replaced with Resolved Questions, recording both decisions and the reasoning.

**Iteration 3 (2026-09-02) — constitution compliance**

The project constitution was ratified at v1.0.0 after this spec was drafted, so the spec was re-checked against all ten principles:

| Principle | Status | Where |
|---|---|---|
| I. Evaluate in the deployment regime | **GAP → FIXED** | Spec required distant-era windows but never stated which window carries the headline verdict. Added FR-027a. |
| II. Persistence across distant eras | PASS | FR-015, FR-027; edge case "the trackability verdict is marginal" |
| III. Learnability is not edge | PASS | FR-029; US3 scenario 2; US4 scenario 2 |
| IV. Verify parameters are actually live | PASS | FR-031; US3 scenario 1; edge case "a mechanism silently does nothing" |
| V. Lookahead-clean is not realistic | PASS | FR-028 (execution realism), FR-007 (end-to-end lookahead, not framework analysis) |
| VI. Diagnose before tuning | PASS | Structural — US1 characterization and US2 verdict both precede any mechanism work |
| VII. Validate cheaply; trackability before adaptivity | PASS | US2 is exactly this gate; FR-014, FR-017 |
| VIII. One parameter space at a time | N/A | No parameter search in scope; mechanisms are measured as discrete arms |
| IX. Orthogonality earns a factor its place | PARTIAL | Applies only to M4, where phase features join an existing feature set. Not raised to an FR; flag at `/speckit-plan` if M4 is reached. |
| X. Structural fixes, not per-pair overrides | PASS | FR-025; Out of Scope |

All 16 checklist items now pass.

## Notes

- Spec is ready for `/speckit-plan`. `/speckit-clarify` is not required — no clarification markers remain.
- One deferred item for the planning phase: Principle IX (orthogonality) applies to mechanism M4, where phase features would be added to an existing model input set. If M4 is reached, check phase-feature correlation against the existing features before evaluating performance.
- Validated against constitution v1.0.0 (ratified 2026-09-02).
