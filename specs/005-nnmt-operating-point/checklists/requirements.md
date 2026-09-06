# Specification Quality Checklist: NNMT Operating-Point Programme

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-06
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

## Notes

Validated 2026-09-06, 16/16 pass. 29 FRs, 10 SCs, 7 user stories, 7 edge cases,
zero clarification markers, no numbering gaps.

Three points recorded deliberately rather than as defects:

1. **US1 is a kill switch, not just a first step.** FR-004 requires the study to
   re-scope rather than execute the remaining work if NNMT's headroom proves
   negligible. That is the outcome on NNNC's prior configuration, so it is the
   likely case, not an edge case.

2. **US2 is a defect fix, not an experiment**, and is P1 for that reason: the
   label inconsistency was introduced by work done for another family and should
   not queue behind a research question.

3. **The spec is written without naming files, parameters or tools.** The
   underlying facts are concrete (specific thresholds, specific measurements),
   but the requirements are phrased in terms of what must be established rather
   than which knob to turn, so the plan retains freedom over method.

Constitution alignment: Principle II (persistence) is carried by FR-018's
window treatment and by the edge case recording that NNMT's third window is
near-flat. Principle III (learnability is not edge) is enforced mechanically by
FR-019. Principle IV (verify parameters are live) by FR-010's inert-guard
requirement and FR-023's model-path rule. Principle VII (validate cheaply) is
the feature's spine — FR-003 requires the decisive measurement to need no
training.

**Upstream dependency resolved 2026-09-06**: the label threshold that NNMT
inherits was contested during specification. It has since been settled at 0.020
(`regime/LABEL_EXIT_ALIGNMENT.md` — 0.024/0.026 win the acceptance window 4 of 4
but lose summed 3 of 4 and cost 20-45% of trade volume). NNMT therefore inherits
a settled value, and FR-005's two-way comparison is 0.007 against 0.020.

**Correction applied 2026-09-06 before planning**: the spec initially claimed
THREE untested inherited changes. Verified against NNMT's actual effective
configuration, there are TWO — the label threshold (0.007 → 0.020) and the
profit-taking level (0.032 → 0.022). The exit-side relative-volume gate was
parameterised but its default equals the value that was previously hardcoded, so
NNMT's behaviour there is unchanged; every entry guard and the confidence
threshold are likewise unchanged because NNMT overrides them with its own swept
values. Recorded because the erroneous claim would have led the plan to
construct a baseline arm for a change that never happened.
