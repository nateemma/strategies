# Specification Quality Checklist: Label-Exit Coordination

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-08
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

Two deliberate deviations from the usual "no implementation details" rule, both of which I
judged necessary rather than leaked:

1. **Named existing code.** FR-004 names `_stop_aware_labels` and the Why-This-Is-a-Redesign
   section names `custom_stoploss` / `_resolve_normal_stop`. The single-source-of-truth
   requirement is meaningless without identifying the specific duplication it exists to
   remove, and that duplication is a concrete present-tense fact about this repository, not
   a design choice being pre-empted.

2. **Quantitative evidence table.** The spec opens with measured results rather than a
   narrative. This feature's premise is that four cheaper explanations have been eliminated;
   a reader who cannot see those numbers cannot judge whether the redesign is warranted, and
   would likely propose one of the eliminated patches.

Both are recorded here so a reviewer can object to them explicitly.

The spec deliberately does NOT prescribe how the outcome table is stored, how triggers are
registered, or how economic value is computed — those are planning decisions.

Resolved by `/speckit-clarify` (session 2026-10-08), four questions:

1. **Liquidity** — outcome records carry a fillable flag and fillable fraction, and every
   economic value is reported both unfiltered and fillable-only (FR-009a/b, SC-010). This
   closes the risk that a trigger earns its value from bars that could not have been traded,
   which this repository has already been bitten by once.
2. **Costs** — a positive outcome is net of the same fees freqtrade applies in a backtest,
   with no slippage model, so the table and the walk-forward harness measure the same
   quantity (FR-006a).
3. **The verdict rule** — SC-007 was the item flagged here as open. Each trigger is now judged
   against the EMPTY trigger, with three total verdicts (selection value / decoration /
   indeterminate) and both boundaries set by the measurement uncertainty on the difference
   rather than a threshold invented in advance (FR-014a/b).
4. **Learnability** — one fixed cheap probe for every trigger, so the column is comparable,
   and explicitly diagnostic rather than a promotion criterion (FR-015, FR-015a).

Nothing remains open that would change the design. Ready for `/speckit-plan`.
