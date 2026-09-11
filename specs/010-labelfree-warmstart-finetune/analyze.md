# Cross-Artefact Analysis Report — 010-labelfree-warmstart-finetune

**Date**: 2026-09-11 · **Stage**: pre-implementation (required by
`research/label-free-study/README.md` step 5 and by FR-025 / SC-016)
**Artefacts**: `spec.md`, `plan.md`, `tasks.md`, with `research.md`, `data-model.md`,
`contracts/`, `quickstart.md` as supporting context.
**Authority**: `.specify/memory/constitution.md` v1.0.0.

## Verdict

**CLEAN — 0 CRITICAL, 0 HIGH, 0 MEDIUM outstanding.** Six findings raised, all remediated.

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| G1 | **CRITICAL** | `baselines.py` and `reproduce.py` were RUN by tasks but no task BUILT them, and neither existed nor appeared in the plan tree. Both were listed in brief 01's plan and never built, because its Phases 6-7 were unreached — so this feature silently inherited the assumption they exist. They carry SC-003 (four mandatory baselines) and SC-006 (reproduction): two acceptance gates resting on absent code | **T032a** and **T032b** added to build them; both added to the plan tree as NEW with their provenance noted; a Complexity Tracking row records the inherited-assumption hazard |
| O1 | MEDIUM | T002 forward-referenced T035, 33 tasks later — a setup task validated against a result that does not yet exist | T002 reworded to record brief 01's figures as a standalone reference point, citing FR-004b |
| C1 | LOW | Seven tasks carried no requirement citation; three were substantive (windowed encoding, the Arm C run, the pooled walk-forward) | FR-004a/research R2 on T013, FR-011/FR-016 on T025, FR-018/SC-001 on T030. The three remaining are scaffolding or process instructions |
| D1 | LOW | FR-025~SC-016 (0.89) and FR-020~SC-015 (0.70) restate each other | No action — intentional mirroring of a requirement and its measurable gate, the same structure as feature 009 |
| T1 | LOW | "label control", "screening fold", "participation floor", "arbiter" are defined in spec.md's glossary but appear zero times in plan.md | No action — the spec's glossary governs; noted so a future plan revision aligns |
| A1 | LOW | US1 narrative says "much smaller learning rate" | No action — FR-005 quantifies it (`encoder_lr * 10 <= head_lr`); narrative prose, not the testable requirement |

## Metrics (post-remediation)

| Metric | Before | After |
|---|---|---|
| Requirements | 45 (28 FR + 17 SC) | 45 |
| Tasks | 44 | **46** (+2 build tasks) |
| Requirement coverage | 100% (but 2 citations pointed at absent code) | **100%, all resolvable** |
| CRITICAL | 1 | **0** |
| Malformed tasks | 0 | 0 |
| `RUN` tasks wrongly marked `[P]` | 0 | 0 |
| Forward references | 1 | **0** |

## Constitution

PASS, no violations.

- **Principle III** is guarded in three places: the exposure/label correlation is marked
  diagnostic-only in spec (SC-011), contract (`warm-start.md`) and task (T016). It is the one
  quantity in this feature that could plausibly drift into deciding a verdict, and it cannot.
- **Principle VI** is structurally enforced: T017 forbids tuning Arm A when it underperforms and
  routes to Phase 4 instead, so a weak arm is recorded rather than fitted to the screen.
- **Principle IV**: FR-009/T019 measure each loss term's realised contribution, so a nominal
  blend that has collapsed to one term is detected rather than assumed balanced.
- Feature 009's **Principle VIII exception does not carry over** — this feature sweeps one
  parameter within one arm, not a factorial across three dimensions.

## Why G1 matters beyond this feature

It is the same latent-dependency failure that surfaced in feature 009, where a task was marked
complete for running an `audit_trials.py` that had never been written. The difference is timing:
009 found it in the post-implementation analysis, after the work; this found it before any code
was written. Left alone it would have failed at T033a — deep in Phase 6, immediately after the
15-classifier training step that is this feature's dominant compute cost.

The generalisable lesson, recorded here rather than in a commit message that will not be read
again: **a plan inherited from an unfinished feature carries its unbuilt modules as invisible
assumptions.** Checking "referenced by tasks" against "exists in code" *and* "named in plan" is
cheap and catches it.

## Next action

Proceed to `/speckit-implement`. Re-run `/speckit-analyze` on completion (T039) and append to
this file, per the study README step 8.
