# Specification Quality Checklist: Per-Fold Guard Tuning

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-09
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

## Constitution Alignment (project-specific, added for this feature)

- [x] **II. Persistence across distant eras** — FR-016 requires early/late halves, SC-007
      requires both reported.
- [x] **III. Learnability is not edge** — FR-018 bars classifier metrics as an acceptance input.
- [x] **IV. Verify parameters are actually live** — FR-013 requires each tuned parameter proven
      to change the trade list, from the hyperopt record and a trade-list hash, not the console.
- [x] **VII. Test trackability before building adaptivity (NON-NEGOTIABLE)** — US2 is a P1 gate,
      FR-004/005/006 require trailing-best vs a CONSTANT with regret against real candidates,
      and FR-006 forbids arm C if the gate fails. **This was absent from the source outline and
      is the single largest change this spec makes to it.**
- [x] **VIII. One parameter space at a time** — FR-010 and FR-011 restrict the search to a
      reduced guard space only.
- [x] **XI. No data leakage** — FR-007 restricts tuning to the training window with one test
      evaluation per fold.

## Notes

### Validation findings, resolved during authoring

1. **The source outline violated Principle VII.** It proposed 15 per-fold hyperopts (arm C) with
   no prior test of whether the guard optimum is *predictable* rather than merely *variable*.
   The constitution makes that gate mandatory and specific, and records that four adaptive
   schemes have already lost to static values here. Resolved by adding US2 as a P1 gate ahead of
   arm C, with FR-006 making the gate blocking.
2. **"Reduce the search space" was an instruction without a failure mode.** Resolved by an edge
   case and FR-011: if no reduction can be justified, the feature terminates early and that is a
   legitimate outcome.
3. **Success criteria initially leaned on pp/yr thresholds inside SC items.** Moved the
   thresholds into the pre-registered decision rule, leaving the SC items as statements about
   *what gets measured and recorded* — verifiable without knowing the result.
4. **Zero-trade folds.** Arm B had one. FR-017 and an edge case now prevent such a fold from
   contributing a 0.0pp observation, which would bias any pooled mean toward zero.

### Deliberate deviation from the template

A **Pre-registered decision rule** subsection was added under Success Criteria, and an
**Out of Scope** section with the evidence for each exclusion. Both exist because this is a
measurement feature whose chief risk is re-reading the result after seeing it (FR-019).
