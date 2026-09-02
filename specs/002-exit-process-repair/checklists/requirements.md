# Specification Quality Checklist: Exit-Process Characterisation and Repair

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

1. *Implementation leakage* — the draft named source files, line numbers, and parameter identifiers (`stoploss_grace_hours`, `cexit_take_profit`, `enable_exit_signal`, `BaseStrategy.py:794`) throughout the requirements. Rewritten to describe behaviour: "the early-window boundary", "the take-profit level", "currently-dormant behaviour". The concrete identifiers belong in the plan, where they are needed, not in the spec.
2. *Ordering hazard in the screen* — FR-014 added after noticing that a screen which cannot reproduce the current rule's recorded outcomes would silently mis-score every candidate. Validating the screen against the known baseline must precede scoring anything with it.
3. *Missing the two-population risk* — the 37.4% of stops that exit in profit and the 62.6% that exit at a loss may want opposite remedies (widening helps one, harms the other). Averaging them into a single recommendation would hide that. FR-004, SC-003 and an edge case added.
4. *Category-shifting as false improvement* — an intervention that converts stops into equally-bad time-based exits would look like success if judged on exit-reason mix. Edge case added; judgement is on total outcome.
5. *No negative-outcome path* — SC-010 added so that "the exit is already well-calibrated and nothing improves it" is a successful completion, not a failure to deliver. Prior work already records the exit as calibrated under a different configuration, so this is a live possibility.
6. *Inert-parameter blind spot* — the exit side specifically holds several settings that look tunable and do nothing. FR-023, FR-027 and an edge case added; an unchanged outcome is a defect, never a finding.
7. *Screen-versus-reality gap unstated* — the screen cannot model position limits or capital freed by an earlier exit. Rather than hide this, the disagreement between screen and full evaluation is now itself a reported quantity (edge case + Assumptions).

**Iteration 2 (2026-09-02) — clarification resolved**

Question 1 answered: the baseline is the configuration that produced the 167 characterised trades, so characterisation needs no re-run, and the recent parameter-search output is screened as a candidate rather than adopted.

Folded in as FR-028 plus two additions the answer exposed:

- **FR-028a** — the search output is explicitly *not* privileged by recency; it is screened on the same footing as every other candidate, which also yields a direct read on whether it improved the exit or merely moved it.
- **FR-028b** — the baseline must be captured explicitly at the start rather than read from configuration at each step. The baseline values live in the strategy's parameter file, which overrides the class declarations, and the class fallback **differs on six of eight exit settings**. A step reading the class instead of the file would silently measure a different baseline — the same multi-layer drift that produced three conflicting values for one entry parameter in feature 001.

All 16 checklist items now pass.

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`.
- Validated against constitution v1.0.0. This feature leans hardest on Principle IV (verify parameters are actually live) and Principle VII (validate cheaply before building) — the counterfactual screen in User Story 3 is a direct application of the latter.
