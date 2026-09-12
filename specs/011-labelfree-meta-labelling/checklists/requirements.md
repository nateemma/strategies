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


## Re-validation after /speckit-clarify (2026-09-12)

Four questions asked and integrated. Checklist re-checked: **16/16 still passing**, no
regressions. Sections touched: Clarifications (new), Requirements (FR-005a/b/c), Success
Criteria (SC-002 rewritten, SC-017/SC-018 added), Assumptions.

Coverage summary:

| Category | Status |
|---|---|
| Functional scope & behaviour | Resolved — feature set now pinned (re-inferred primary confidence in, pair features out) |
| Domain & data model | Resolved — re-inference reproduction gate defined |
| Non-functional quality | Clear — floor recomputation, seeds, trial accounting already specified |
| Integration & dependencies | Resolved — dependency on the surviving per-fold models made explicit |
| Edge cases & failure handling | Clear — thin folds, empty folds, unseen pairs, null configuration all covered |
| Constraints & tradeoffs | Resolved — the W3 conflict surfaced and settled |
| Terminology | Clear |
| Completion signals | Clear |

**The most consequential answer was Q2.** W1/W2/W3 as written was not satisfiable: W3 (2023)
lies entirely inside the warm-up folds, so no nested model can be trained to evaluate it. Left
unclarified, implementation would have hit this after the feature pipeline was built and either
silently dropped the gate or quietly relaxed it. It is now an explicit anti-causal probe, which
keeps Constitution Principle II's distant-era requirement (W3 vs W1 are ~2.5 years apart)
satisfiable without letting a probe figure be quoted as an out-of-sample return.

**Q3 removed a self-fulfilling result.** The pair-recent-hit-rate feature is causal and would
have passed the leakage audit, but it hands the model a per-pair quality score — making
SC-010's "pair selection, not meta-labelling" kill near-certain to fire, after the compute was
spent. Excluding pair features makes SC-010 a genuine test.
