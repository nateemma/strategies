# Specification Quality Checklist: Oracle-Instrumented Exit Diagnosis

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-07
**Feature**: [spec.md](../spec.md)

## Content Quality

- [X] No implementation details (languages, frameworks, APIs)
- [X] Focused on user value and business needs
- [X] Written for non-technical stakeholders
- [X] All mandatory sections completed

## Requirement Completeness

- [X] No [NEEDS CLARIFICATION] markers remain
- [X] Requirements are testable and unambiguous
- [X] Success criteria are measurable
- [X] Success criteria are technology-agnostic (no implementation details)
- [X] All acceptance scenarios are defined
- [X] Edge cases are identified
- [X] Scope is clearly bounded
- [X] Dependencies and assumptions identified

## Feature Readiness

- [X] All functional requirements have clear acceptance criteria
- [X] User scenarios cover primary flows
- [X] Feature meets measurable outcomes defined in Success Criteria
- [X] No implementation details leak into specification

## Validation notes

**Naming.** Parameter and class names are deliberately avoided in favour of the
behaviour they control — "profit-taking level", "stop band", "label horizon",
"perfect-entry arm". The one place implementation surfaces is FR-021 and FR-023
(parameter files, shared framework code), which are process constraints this
repository has had to encode as requirements because violating them has silently
corrupted prior results.

**Coverage, checked in both directions.**

| user story | requirements | success criterion |
|---|---|---|
| US1 instrument integrity | FR-001..003 | SC-001 |
| US2 the bound | FR-004..005 | SC-002 |
| US3 attribution | FR-006..008 | SC-003 |
| US4 target ordering | FR-009..010 | SC-004 |
| US5 real-model confirmation | FR-011..013 | SC-005 |
| US6 conclusion | FR-026 | SC-009 |
| cross-cutting evidence standards | FR-014..025 | SC-006, SC-007, SC-008 |

26 FRs (FR-001..FR-026, contiguous), 9 SCs, 6 user stories, 6 edge cases.
Every FR traces to a story or to the cross-cutting block; every story has at
least one success criterion.

**Two structural notes for planning.**

1. **US5 is P1 despite being fifth.** It is the only step that can license a
   promotion, and the standing finding is that a perfect-entry instrument cannot
   settle exit questions on its own. It must not be dropped if the feature is cut
   short.
2. **US1 gates everything.** The instrument is 84.1% compliant with its own
   label and the cause is unknown. No exit conclusion is trustworthy until that
   is explained or bounded, so US1 precedes US2 even though US2 is the headline
   measurement.
