# Specification Quality Checklist: Label Warm-Start with P&L Fine-Tune

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-11
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

## Validation Notes

### Iteration 1 (2026-09-11) — 3 markers open

All three arose because brief 02 was written *before* feature 009 produced its result, so the
brief could not have known what brief 01 would deliver.

### Iteration 2 (2026-09-11) — RESOLVED, all 16 items pass

| Marker | Resolution | Knock-on |
|---|---|---|
| Arm A architecture | All three arms adopt the classifier's encoder architecture, so the warm-start is a literal weight inheritance and A/B/C differ only in initialisation and objective | **FR-004b added**: the P&L control must be RE-RUN under the new architecture. Comparing against brief 01's recorded numbers would confound the label prior with the architecture change, which is the one thing this feature exists to isolate. **SC-010a** requires both figures be reported so the architecture effect is visible separately |
| Which P&L control | `utility-penalty-g0.5` — mean exposure 0.130, −7.06% held-out; the only brief-01 arm that genuinely trades | Brief 01's top-*ranked* configs are excluded as degenerate (flat on 100% of bars, so any arm that trades beats them trivially). The bar is low and the report must say so rather than presenting a win over it as success |
| Screening fold | **f08**, not the f07 the brief names | f08 carries 71 control trades vs f07's 58, and f07 is the most *profitable* fold (+8.08% vs −0.14%) — screening there is the selection bias feature 009 corrected. f08 also keeps this screen directly comparable to brief 01's, which matters because brief 01 is a control |

**Note on the reply**: Q1 was answered "1" against options labelled A/B/C; read as option A, the
first option, consistent with Q2 and Q3 both being answered "A". Recorded here in case that
reading was wrong — the alternatives (B: lossy weight projection; C: drop Arm A) lead to
materially different work.

**Constitution compliance** (v1.0.0) — checked, no violations:

| Principle | Where honoured |
|---|---|
| I. Deployment regime | Assumptions: protocol windows unchanged |
| II. Distant-era persistence (NON-NEGOTIABLE) | SC-002 requires positivity in all three windows |
| III. Learnability is not edge | No model-fit metric in any SC-line; all gates are cost-aware P&L. FR-006/SC-011's correlation is a *diagnostic*, explicitly not an acceptance criterion |
| IV. Verify parameters are live | FR-009 measures each loss term's realised contribution rather than assuming the blend is balanced |
| V. Lookahead-clean is not realistic | Inherited calibrated simulator; FR-013/FR-014 purge and embargo |
| VI. Diagnose before tuning | US1 scenario 4: an underperforming arm is recorded and the work moves on rather than being tuned |
| VII. Validate cheaply | Screening fold before any pooled run |
| VIII. One parameter space at a time | Arms are separate; only the blend weight is swept, and FR-010 forbids reporting a best value |
| IX. Orthogonality | N/A — single policy |
| X. Structural fixes | Non-Goals bars per-pair code |
| Repo boundary / staging | FR-024 |

**Deliberate carry-forward from 009**: the participation floor, mask-normalised coverage,
date-aligned panels and held-out ranking are all inherited as Assumptions rather than
re-specified, so the two features cannot drift.

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`
- All three markers are posed as questions in the session
