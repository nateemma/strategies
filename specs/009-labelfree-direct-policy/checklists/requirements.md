# Specification Quality Checklist: Label-free Direct Policy Optimisation

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-10
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

## Validation Notes

### Iteration 1 (2026-09-10) — 3 markers open

Three numeric thresholds on pass/fail gates had no defensible default: the calibration
tolerance (SC-001), the lookahead absurdity bar (SC-002), and the calibration budget (SC-012).

### Iteration 2 (2026-09-10) — RESOLVED, all items pass

| Marker | Resolution | Knock-on change |
|---|---|---|
| Calibration tolerance (SC-001) | Per-trade profit within 1 bp **and** total return within 0.10pp **and** identical trade counts — all three | FR-008 restated to require the per-trade figure be reported beside the total, so offsetting errors cannot pass on the total alone. FR-003 now pins the arbiter's exact fill convention (following bar's open, not a VWAP proxy) *during calibration*, since a per-trade comparison is otherwise meaningless; any training-time proxy must be reported as a deviation. |
| Absurdity bar (SC-002) | ≥ 5× the control's deployment-window return (≈ +50%), above the +26.5% oracle ceiling | FR-010 restated with the same bar and an explicit consequence: below it, the learning path is defective and no variant sweep may run. |
| Calibration budget (SC-012) | 5 fix-and-retest attempts, no wall-clock bound | Attempt defined as one calibration run against the arbiter, so the stop rule is countable. |

**Deliberate deviations, recorded rather than silently fixed:**

- *Technology-agnostic success criteria*: SC-lines retain domain metrics (pooled walk-forward
  return, fee multiples, PBO) because the protocol requires its §5 gates verbatim or stricter.
  These are outcome measures, not implementation details — no language, framework or module is
  named in any SC-line. Tool and path names are confined to Assumptions and Functional
  Requirements.
- *Success criteria are stricter than the template's "user-facing" framing* because the user
  here is a researcher and the deliverable is an admissible finding, not an end-user feature.

**Constitution compliance** (`.specify/memory/constitution.md` v1.0.0) — checked, no violations:

| Principle | Where honoured |
|---|---|
| I. Evaluate in the deployment regime | Assumptions: deployment window is the sole single-window acceptance window |
| II. Persistence across distant eras (NON-NEGOTIABLE) | SC-004 requires positivity in each of the three windows, one of which is the distant era |
| III. Learnability is not edge | No model-fit metric appears in any SC-line; all gates are cost-aware P&L |
| IV. Verify parameters are actually live | SC-002 and US2; a non-responsive learning path is a defect, not a null |
| V. Lookahead-clean is not realistic | FR-004, FR-005: costs and the fill cap sit inside the objective; SC-007 cost stress |
| VI. Diagnose before tuning | US1/US2 ordered before any variant sweep |
| VII. Validate cheaply before building | Calibration and sanity ceiling gate the expensive grid |
| VIII. One parameter space at a time | US3 screens on one fold before the pooled run |
| IX. Orthogonality | Not applicable — single policy, not a multi-factor book |
| X. Structural fixes, not per-pair overrides | Non-Goals bars per-pair code; FR-002 requires pair-agnostic features |
| Repo boundary / staging | FR-031, Assumptions |
| Workflow gates 1–7 | FR-015 (pinned ranges), SC-004 (distant era), FR-005 (realistic execution), SC-002 (liveness), FR-016 (seed robustness), edge cases (trade count), Assumptions (scalers) |

### Iteration 3 (2026-09-10) — re-validated after `/speckit-clarify`

5 clarifications integrated; all 16 items still pass, no regressions. Two open items from the
specify report were closed by measurement rather than by asking:

- **Fold count** — `regime/wf/pooled_nnnc.json` carries exactly 15 per-fold entries and 378 pooled
  trades, so SC-004's "10 of 15" is correct as written.
- **W2/W3 binding** — protocol §5 gate 2 governs over §2's "reported only" phrasing, per the
  study's rule that window questions are answered by the protocol rather than re-litigated. SC-004
  keeps the gate verbatim; FR-015a now records the universe consequence.

Scope added by the clarifications (relevant to `/speckit-plan` sizing): FR-004a (spread estimator),
FR-008a (calibration validates accounting only), FR-014a/b (band-only training + mandatory
sim-vs-arbiter reconciliation), FR-015a (single-window universe), FR-021a (frozen trend baseline),
SC-018 (reconciliation gate). Requirements went 33 → 39, success criteria 17 → 18.

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`
- The three markers are posed as questions in the session; answering them clears the only
  failing checklist item
