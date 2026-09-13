# Specification Quality Checklist: labelfree-rl-pooled-env

**Purpose**: Validate specification completeness and quality before planning
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
- [x] Success criteria are technology-agnostic
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

**The brief's own acceptance gate was unenforceable.** It requires beating "brief 1's best
pooled result", but brief 1 was NOT RESOLVED and its report states "No configuration was
carried to Phase 6" — the referent does not exist. Resolved by splitting the gate: SC-009 is a
screening-fold head-to-head against brief 1's differentiable policy (the brief's actual
question), and SC-001 remains the separate acceptance bar (deployability). Those can have
different answers, and conflating them would have made both unanswerable.

**The brief specifies the state it criticises FreqAI for.** It rejects FreqAI's `add_state_info`
because "a position-aware agent can only be evaluated in dry-run — incompatible with the
protocol", then specifies position, bars-in-trade and unrealised P&L in its own observation.
Resolved by evaluating through per-candidate hooks that see the trade. The head-to-head is
unaffected because it runs in the simulator, where both policies see the same environment.

**SC-019 was added, not asked for.** Screening on f07 and then counting f07 in the pooled
verdict is selection contamination — the class of defect that voided 60 runs in feature 009
(screening ranked on the training range). The screening fold's role is now pre-registered: it
selects only between the two action parameterisations, and its contribution to the verdict is
reported separately.

**Three requirements come from defects found in features 009–011, not from the brief**:
FR-006/SC-011 (the environment must be able to *exhibit* reward hacking — a control that cannot
fail certifies nothing, after the planted-leak and liveness controls both first passed
vacuously); FR-019/SC-015 (every swept parameter proven live, after four parameters in this
study were configured, logged, believed active and reaching nothing); and FR-017/SC-013
(a below-floor policy reported as inconclusive with its return withheld, since "learn not to
trade" is this environment's degenerate optimum).

**Prior evidence recorded in Context, deliberately.** Briefs 01–03 all failed, and brief 03
established that the confidence→outcome map is non-stationary across 90-day windows — which
applies to any policy trained on one fold and acting in the next. This is written into the spec
so the gates are set with it in view. It is not an argument against running the feature: the
brief's narrow question, whether sampling beats smoothing on path-dependent decisions, is
untested by 01–03.
