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

---

# Post-Implementation Analysis — 2026-09-12 (T039)

Read-only. Run after Phase 6's unblocked tasks and Phase 7's audit/report, with Phase 6's
carry-forward runs (T029, T030, T032, T033, T034) still gated on a surviving arm.

## Specification Analysis Report

| ID | Category | Severity | Location(s) | Summary | Recommendation |
|----|----------|----------|-------------|---------|----------------|
| C1 | Constitution / Coverage | **CRITICAL** | `folds.py:41-42`, `arms.py:load_panel`, spec.md:233-236 | **Purge and embargo were never applied.** Every `Fold` carries `purge_bars=0, embargo_bars=0`, and `labels.apply_leakage_control` — which exists, is correct, and states FR-014's rationale in its own docstring — is **never called** by `arms.py`. `load_panel` also discards `LabelTarget.valid`, keeping the target's values but not its validity mask. FR-013 requires the purge for every arm with a forward-looking term (B: label CE; C: forward realised vol). FR-014: "a run missing either MUST be **void** rather than caveated." | Call `apply_leakage_control` in `load_panel`, record the applied values per fold, and re-run arms B and C. Under FR-014 as written the current B and C runs are void. |
| C2 | Coverage | **CRITICAL** | `stats.py:trial_log_deflation`, spec.md:289-291 | **SC-004 cannot be evaluated.** DSR and CSCV need a time axis; the trial log stores only summary scalars, leaving 5 seeds as the only axis. Raw figures (DSR 0.902, PBO 0.000) are artefacts of that — a reliably mediocre arm scores well because its across-seed dispersion is tiny. | Extend the trial-log schema to store per-bar net returns per trial. Until then SC-004 is unevaluated for every brief in this study, not just this one. Already recorded in the report and gated in code. |
| H1 | Inconsistency | HIGH | results report tables, spec.md:324-326 | **Returns reported as medians, not IQM with a bootstrap CI**, contrary to SC-014/FR-016. Measured difference is up to **+0.104 log** on the P&L control (median −1.0894 vs IQM −0.9855), which moves Arm C's headline advantage from +0.9811 to +0.8784. | Re-emit the report's tables through `stats.iqm` + `stats.bootstrap_ci`. No verdict changes, but the quoted numbers are not the ones the spec mandates. |
| H2 | Coverage | HIGH | `losses.py`, spec.md:219 | **FR-009 has no implementation.** The realised contribution of each loss term is never measured, so Arm B's blend weight is unverified in exactly the way `label_threshold` was — a swept parameter whose mechanism nobody confirmed was live. That defect has already occurred once in this feature. | Log per-term loss contributions per epoch and assert the CE term is non-zero when `blend_alpha > 0`. |
| M1 | Underspecification | MEDIUM | spec.md:228-231 | **FR-012's test is confounded.** "Whether directional label content was necessary" was to be answered by comparing the non-directional Arm C against A/B. The turnover finding (Spearman(return, turnover) = −1.000) shows that comparison measures cost, not label content. | FR-012 is not answerable by the runs performed. The turnover-matched control named in the report is the experiment that would answer it. |
| M2 | Inconsistency | MEDIUM | report §"The architecture effect" | FR-004b/SC-010a are **satisfied** — both control figures are reported — but the −0.7707 log regression they exposed on an identical nominal configuration (turnover 194→648) is **not tracked as a defect anywhere**. It is 1.02× Arm C's entire advantage. | Open it as a defect against the shared trainer. Brief 01's and brief 02's numbers are not comparable until it is explained. |
| L1 | Terminology | LOW | report, spec.md, tasks.md | "P&L control", `pnl_control`, and "brief 01's best config" denote three different things (010's re-run; the arm name; brief 01's `utility-penalty-g0.5`). The report's architecture table compares against `utility-band-g1.0`, a fourth. | Name them once in the report preamble. |

## Coverage Summary

| Requirement | Has implementation? | Evidence | Notes |
|---|---|---|---|
| FR-001, FR-002, FR-003 | Yes | `arms.py` reuses brief 01's sim/features/folds | |
| FR-004, FR-004a, FR-005 | Yes | `warmstart.py`, `policy.py` | |
| FR-004b | Yes | report §architecture effect | exposed a −0.77 regression (M2) |
| FR-006 | Yes | `evaluate_arm` → `exposure_label_corr` | |
| FR-007, FR-008 | Yes | `losses.exposure_ce`, liveness proof in `run_screen` | |
| FR-009 | **No** | — | H2 |
| FR-010 | Yes | report states no best weight | |
| FR-011 | Yes | `losses.aux_vol_loss` | |
| FR-012 | Partial | report §what the arms are doing | confounded (M1) |
| FR-013, FR-014 | **No** | `purge_bars=0` everywhere | **C1** |
| FR-015 | Yes | `trials.py`, audit passes | |
| FR-016 | Partial | 5 seeds run; medians reported | H1 |
| FR-017 | Yes | `arms.select_arms` ranks on held-out only | |
| FR-018 | Partial | both controls in report; pooled compare gated | T032 |
| FR-019 | Yes | `stats.activity` wired into `evaluate_arm` | |
| FR-020 | Yes | `arms.screen_degenerate` | |
| FR-021, FR-022, FR-026 | Yes | report verdict | |
| FR-023 | Yes | runs sequential | |
| FR-024 | Yes | staged explicitly | |
| FR-025 | Yes | spec/plan/tasks/analyze present | |
| SC-001, SC-002 | Gated | — | needs T029/T030 |
| SC-003 | Yes | `baselines.py`, both folds | |
| SC-004 | **Unevaluable** | `trial_log_deflation` | **C2** |
| SC-005 | Gated | — | T034 |
| SC-006 | **Yes** | `reproduce.py`, bit-exact | |
| SC-007, SC-009, SC-010, SC-010a | Yes | report | |
| SC-008 | Yes | `stats.overfitting_differential` | fires; caveated |
| SC-011 | Yes | `exposure_label_corr` | |
| SC-012 | **No** | — | **C1** |
| SC-013 | Yes | audit passes, 318 trials | |
| SC-014 | Partial | activity emitted; medians not IQM | H1 |
| SC-015 | Yes | `screen_degenerate` withholds returns | |
| SC-016 | Yes | this report | |

## Metrics

- Functional requirements: 26 — covered 21, partial 3, **uncovered 2**
- Success criteria: 20 — satisfied 12, gated 3, partial 2, **unevaluable/failed 3**
- Tasks: 46 — complete 39, gated 5, remaining 2 (T039 this report, T042 commit)
- Critical issues: **2**
- Trials pre-registered: 318 (audit clean)

## Next Actions

1. **C1 first, and it is not optional.** FR-014 is categorical: arms B and C are void as run.
   The fix is small — call `apply_leakage_control` in `load_panel` and record the values —
   but it requires re-running B and C. The measured exposure is bounded (the training frame
   is truncated at `train_end`, so no data crosses the boundary; the defect is ~48
   mislabelled bars per pair at the boundary, 0.037% of training, plus the absent mandated
   record). That bound is context, **not** grounds to caveat instead of voiding.
2. **C2 is a study-wide schema change**, not a feature-010 fix. Raise it against the
   protocol.
3. H1 is a re-emit, no re-runs.
4. Do **not** start the gated Phase 6 runs. The turnover confound means they would measure
   the same confounded quantity at 15× the compute.

## Verdict

**NOT CLEAN.** Two CRITICAL findings. C1 voids arms B and C under the spec's own rule; C2
makes SC-004 unevaluable for the study. The feature's *reported verdict* (NOT RESOLVED,
direction confounded with turnover) is unaffected by both — C1's exposure is bounded and
C2 was already reported as unevaluated rather than passed.


---

## C1 — RESOLVED 2026-09-12 (post-analyze remediation)

Fix applied: `load_panel` purges the last 48 train bars, embargoes the first 48 test bars,
returns the values actually applied, and `run_screen` refuses to start if either is zero.
Arms B and C re-run as `20260912T153710Z`.

**The re-run is bit-identical to the voided runs on every seed.** Measured cause: the purge
lands inside a region training never visits, and the embargo inside the evaluation burn-in.

| | length | already discarded by |
|---|---|---|
| purge (train tail) | 48 bars | the chunk loop's 850-bar unvisited remainder |
| embargo (test head) | 48 bars | `evaluate_arm`'s 96-bar burn-in |

So no leakage occurred in the original runs — FR-013's protection was emergent from chunk
arithmetic rather than implemented. The distinction matters: it held by accident and would
break silently if `CHUNK`, the panel length or `BURN_IN` changed. C1 downgrades from
"results are void" to "the guarantee was unenforced", and the arm-B/C figures are now
admissible under FR-014.

## H3 — NEW (MEDIUM), found during remediation

`stats.activity` reports ~1 trade and a window-length holding time for any policy whose
exposure never drops below `eps`. All feature-010 arms read 6.2 trades/yr and 1416h while
their turnover spanned 15.7 to 647.9 — a 41× range. FR-019's trades-per-year is therefore
near-vacuous for continuous-exposure policies, and turnover is the load-bearing activity
measure. Documented in the function; both values are emitted so the discrepancy is visible.


---

## M1 — RESOLVED 2026-09-12 (turnover-matched control)

M1 recorded FR-012's test as confounded: "whether directional label content was necessary"
was to be answered by comparing the non-directional Arm C against A/B, and the turnover
finding (Spearman −1.000) showed that comparison measured cost.

A turnover-matched control was pre-registered and run. `band_width` matched Arm C at IQM
turnover 16.3 vs 16.9 and exposure 0.564 vs 0.429. Arm C retains **+0.0594 log
(+27.2 pp/yr)**, CI excluding zero on the pre-registered IQM statistic.

- **FR-012 is answered for Arm C**: the non-directional auxiliary target has value beyond
  cost damping, 92% of its raw advantage having been turnover.
- **FR-012 is NOT answered for Arm B**: exposure 0.125 is unmatchable against the reachable
  band range 0.48–0.56, at any turnover. Recorded as unanswered rather than relaxing the
  match criterion. The label-prior question carries to brief 03.

## H4 — NEW (HIGH), found during the match

`policy.band_hold` is **bimodal across seeds**. At a fixed `band_width` the policy lands in
one of two attractors — moving nearly every bar, or holding the opening position for the
whole window. The matched control's five seeds give turnover 33.3 / 545.0 / 15.4 / 0.0 / 0.3.

Consequences: any turnover setting is a coin-flip between two behaviours rather than a
level; an IQM over 5 seeds is a trimmed average across two distinct regimes, not five draws
of one; and a single-seed bracket over `band_width` reads as a smooth transition when it is
sampling a mixture. An interim claim in this session was wrong for exactly that reason.

This is a property of the shared trainer, not of feature 010, and every remaining brief
inherits it. It belongs with the −0.7707 architecture regression (M2) as a trainer defect to
fix before reuse.
