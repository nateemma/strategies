# Specification Quality Checklist: labelfree-meta-labelling

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-12
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

## Validation notes (2026-09-12)

Two iterations were needed.

**Iteration 1 — implementation detail leaked into the spec.** FR lines named LightGBM,
`confirm_trade_entry`, `research/labelfree/metalabel/` and `trials.jsonl` directly. All were
rewritten to state the capability ("a small gradient-boosting model rather than a neural
network" appears only as an Assumption; "the strategy subclass consults the secondary model at
entry" replaces the hook name). The starter prompt's paths are preserved in the Input block,
where they belong, and will re-enter at plan time.

**Iteration 2 — success criteria were not measurable as written.** "Protocol §5 verbatim"
is not a testable SC-line. Each gate was expanded into a numbered SC with its own threshold,
and two study-wide figures were corrected against the archive rather than inherited:

- **The floor.** The spec initially quoted 3.35 pp/yr. That was measured on 378 trades over
  3.68 years; the evaluation span is 287 trades over 2.45 years, so the figure does not
  transfer. SC-001 now requires the floor be *recomputed* for the span, and an Assumption
  records the expected magnitude (~5–6 pp/yr) as an estimate the measurement governs.
- **Folds positive.** Protocol §5 gate 2 says "10 of 15 folds". The evaluation span has 10
  folds, so the gate was rescaled to 7 of 10 — the same proportion, not a relaxation.

**One figure in the brief does not match the archive.** Brief 03 states ZEC-like pairs are
"≈70% of P&L", citing an earlier study. On this archive's evaluation span ZEC is 16.0% of
trades and 27.9% of P&L. The spec uses the measured figure and flags the discrepancy; SC-010
is unaffected, since it tests whether the gain *disappears* without ZEC rather than assuming
a share.

**Two requirements exist because of defects found in features 009 and 010**, not from the
brief: FR-004 (the leakage audit must detect a planted leak — an audit that cannot fail
certifies nothing) and FR-012/SC-014 (every swept parameter proven live before compute — four
parameters in this study were configured, logged, believed active, and reaching nothing).
