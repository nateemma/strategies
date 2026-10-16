# Specification Quality Checklist: Realistic P&L for NNNC

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-16
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

- **Repo names are kept on purpose.** As in specs 032–035, the spec names repo artefacts (arm tags,
  the momentum bot's config file, the control class). They identify which existing system is meant,
  and are not implementation choices. No languages, libraries or code structure are prescribed.
- **Resolved (2026-10-16):** Q1 = B (US4 in scope, run last, deployment information only);
  Q2 = C (at least 4 weeks AND every pair at ≥ 5 orders; pairs still short at 12 weeks are pooled
  as thin); Q3 = A (control only, one bot).
- **Defaults taken without a marker (see Assumptions):**
  - the live configuration is the NNNC control;
  - the retrain cadence is 90 days;
  - fees are 0.10% per side;
  - snapshots keep the top 20 levels;
  - the budget set is 5/10/20/40 bps.
