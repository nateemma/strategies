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

### Re-validated after /speckit-clarify (2026-10-09)

Two clarifications were asked and four resolved from repository evidence. All checklist items
remain passing, and three became *more* strongly satisfied rather than newly so:

- **Requirements are testable and unambiguous** — FR-011 and FR-012 were previously deferred to
  planning ("the thresholds with existing evidence of binding", "a pre-registered minimum").
  Both now carry concrete values fixed in the spec: the grid is `atr_pct` x `adx` x `rvol` at
  4 levels, and the floor is 25 trades/fold derived from the harness's own 378-trade /
  3.350 pp/yr power record.
- **Dependencies and assumptions identified** — arm E's dependency is now resolved as a fact
  rather than an assumption: all 15 incumbent model artefacts exist on disk, so no training
  occurs in this feature and the one-trainer-at-a-time constraint does not bind.
- **VIII. One parameter space at a time** — now a *documented deviation* rather than a silent
  one. A joint 3-D coarse grid bends the letter of the principle; the reason is recorded
  (super-additivity makes sequential tuning non-composing) and the mitigation is coarseness.

One premise of the source outline was **corrected** by this pass: it assumed `rvol` was the
dominant guard. The ablation shows `atr_pct` (-7.33pp) and `adx` (-5.95pp) are the heaviest
lifters and `rvol` (-3.42pp) is third; `rvol`'s dominance is of gate *rejections*, not of
return. `guard_metric` is excluded entirely as proven inert.

### Re-validated after /speckit-analyze (2026-10-10)

The analysis found **1 CRITICAL, 4 HIGH, 5 MEDIUM, 1 LOW**. Six were fixed; the rest are
recorded below as outstanding with their rationale.

**Fixed**

- **D1 (CRITICAL) — the blocking gate was not enforceable.** FR-006 says arm C must not run on a
  failed gate, but enforcement was a prose note in tasks.md that nothing prevented an operator or
  agent from reading past. Now T009/T010: `run_folds.py` refuses any plan carrying a `guards` key
  unless the trackability verdict reads `trackable`, with a test covering all three states
  (missing, `not_trackable`, `trackable`). This feature already applied the
  "a guard that cannot fail is worthless" standard to its own negative control; it now applies it
  to its headline gate.
- **C1 (HIGH) — FR-018 had zero task coverage**, and **F1 (HIGH) — the scoring field was named
  `score` while the instrument was a proxy.** Both fixed by T019: the field is `proxy_score`
  everywhere, with a test that no reporting path labels it as a return. Feature 021's central
  error was exactly this conflation.
- **A1/A2/A3 (HIGH) — the pre-registered decision rule was not total.** It had no branch for an
  arm C *worse* than A, allowed two branches to hold simultaneously, and did not gate on the
  powered-fold count. Now four ordered, mutually exclusive branches with Branch 0 (POWER,
  `indeterminate` below 10 powered folds) evaluated first. **This mattered to fix now**: FR-019
  freezes the rule once arm C runs, so an incomplete rule would have hardened into an unfixable
  one.
- **F2/F3 (MEDIUM, fixed opportunistically)** — plan.md's `Framework/` comment stated the
  *opposite* of its reason ("Nothing here is runtime a strategy inherits") and conflicted with
  T008's test-allowlist edit. Now "NO RUNTIME CHANGE — precisely BECAUSE this is runtime code a
  live dry-run inherits".

**Outstanding, accepted**

- **C2 (MEDIUM)** FR-010 has no task asserting the search never widens beyond the three axes.
  Mitigated in practice by T002's committed grid and T019's field test; worth folding into T016's
  test file during implementation.
- **C3 (MEDIUM)** FR-020's "no model is fitted" rests on T038's instruction rather than a check.
  An artefact-mtime assertion would close it; low risk because arm E only runs in the contingent
  branch.
- **F4 (MEDIUM)** five names persist for one entity (guard configuration / `GuardConfig` / grid
  point / candidate / `config_id`). Harmless to execution, worth canonicalising if the artefacts
  are revised.
- **L1 (LOW)** "super-additive" vs "super-additivity" wording. Cosmetic.

**Coverage after fixes**: 28/28 requirements have at least one task (was 26/28). 40 tasks, IDs
sequential, all 29 prose cross-references verified to resolve to their intended targets after
renumbering.
