# Specification Quality Checklist: Tail-Targeted Breakout Labelling

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-04
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

## Notes

Validation performed 2026-09-04. Two iterations.

**Iteration 1 findings, all fixed:**

1. *Implementation detail leaked into requirements.* Draft FRs named concrete
   parameters (`HORIZON`, `MIN_BUY_GAIN_THRESHOLD`, `unclog_12h`,
   `atr_stoploss_multiplier`), specific values (H=96, 0.03) and file paths.
   Rewritten to state the property required — "a configuration that requires no
   exit-side change", "the best-economics configuration identified by screening"
   — leaving the settings to planning. The motivating figures remain in Overview
   as evidence, which is where measured context belongs.
2. *Success criteria were technology-flavoured.* SC drafts referenced
   `trades_hash`, `exit_config_hash` and `aug_risk` by name. Restated as
   observable outcomes: "the trade-difference check recorded", "configuration
   fingerprint verified", "above the density floor".
3. *Two user stories shared priority P1 without justification.* Stories 1 and 2
   are both P1 because the headline candidate is unusable without the hold-window
   change — changing the label alone would produce a false negative. The "Why
   this priority" text now says so explicitly rather than leaving equal
   priorities unexplained.

**Deliberate choices worth recording:**

- **No [NEEDS CLARIFICATION] markers.** Every open question had a defensible
  default from the preceding study: the operative stop is the adaptive band (the
  static value was confirmed inert), the density floor comes from the existing
  convention, and the training window is set so the deployment-regime window
  stays out-of-sample. All are recorded in Assumptions.
- **FR-022 makes a negative result a deliverable.** The feature's own evidence
  says six comparable interventions changed nothing, so "the label was not the
  problem" is a likely and legitimate outcome. Without this the feature would be
  under pressure to manufacture a positive.
- **FR-015 and SC-003 exist because "no effect" is ambiguous here.** In the
  preceding study, changes repeatedly produced byte-identical trades. The spec
  therefore distinguishes *refuted* from *absorbed by the gate*.
- **FR-014 forbids metric-based acceptance.** Three separate metrics have each
  failed to predict return in this codebase, once pointing the wrong way by 43
  percentage points.
