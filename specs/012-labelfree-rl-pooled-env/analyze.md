# Cross-Artefact Analysis Report — 012-labelfree-rl-pooled-env

**Date**: 2026-09-12 · **Stage**: pre-implementation (required before `/speckit-implement`)
**Artefacts**: spec.md, plan.md, research.md, data-model.md, contracts/rl-cli.md, quickstart.md,
tasks.md · **Mode**: read-only

## Specification Analysis Report

| ID | Category | Severity | Location(s) | Summary | Recommendation |
|----|----------|----------|-------------|---------|----------------|
| C1 | Coverage / Inconsistency | **CRITICAL** | contracts/rl-cli.md §6, tasks.md T039, spec.md SC-009 | **The head-to-head has no comparator artefact.** The contract invokes `headtohead.py --brief1 research/labelfree/sweep_best.json`; that file does not exist, and **brief 1 persists no trained policy anywhere** — `sweep.py`, `train.py` and `arms.py` write JSON results only, never model weights. SC-009 is the gate this spec was restructured around, and it cannot be executed. | Add an explicit task to **re-train brief 1's best configuration and persist it** before the head-to-head, budgeted as its own work item. Either that, or SC-009 must compare against a re-run rather than a saved artefact — but the dependency must be named, not assumed. |
| C2 | Coverage | **CRITICAL** | tasks.md T045, `research/labelfree/baselines.py:166` | **SC-003 is not satisfiable, for the second feature running.** T045 runs "the four mandatory baselines on the evaluation span", but `baselines.py` still accepts `--fold` only. Feature 011 found this same gap (its analyze C2) and wrote T043a to fix it — **that task was cancelled by the SC-009 kill**, so the fix never landed and feature 012 inherits the gap. | Re-introduce the span mode as a task in *this* feature. Also a process signal: a cancelled remediation task leaves a hole that the next feature silently inherits. Cancelled-with-reason should distinguish "no longer needed" from "still needed, just not here". |
| H1 | Underspecification | HIGH | tasks.md T037, spec.md SC-018 | **"Distinguishable from brief 1's policy" has no operational definition.** The screen bar requires the RL policy be non-degenerate *and* distinguishable, but no test is named. Two policies differ in some digit almost surely, so a literal reading always passes and the gate is vacuous. | Define it before the screen runs: e.g. the per-seed return distributions differ under a paired test at a stated level, *or* their exposure paths differ on more than X% of bars. Pre-register the choice — a bar defined after seeing the screen is not a bar. |
| H2 | Inconsistency | HIGH | data-model.md StepResult, research.md R2, tasks.md T005 | **The reward's time index is stated three ways.** R2 and T005 say the reward at step *i* uses `log(open[i+2]) − log(open[i+1])`; StepResult writes `bar_return = log(open_{t+2}) − log(open_{t+1})` with `reward = w_t × bar_return`. Consistent — but the observation is stated as "ending at bar *i*" while `w_t` is indexed *t*, and nothing fixes whether *t* and *i* are the same index. Feature 009 lost five bugs to exactly this kind of axis ambiguity, each producing plausible shapes and wrong numbers. | Fix one index symbol across all artefacts and state the invariant once: *the action taken on observation ending at bar i earns the return between open[i+1] and open[i+2]*. The equivalence gate (T008) will catch an error, but only after it is written. |
| M1 | Underspecification | MEDIUM | tasks.md T011, quickstart S2 | **"Turnover rises measurably" is not a threshold.** The reward-hacking control asserts a rise but names no magnitude, so a 1% rise and a 10× rise both pass. | State the multiple (e.g. ≥2× turnover under the shaped reward) and pre-register it. T012 already requires the assertion be *able* to fail; a threshold is what makes that meaningful. |
| M2 | Coverage | MEDIUM | tasks.md Phase 8, spec.md SC-006 | T050 reproduces the accepted configuration, but `research/labelfree/reproduce.py` is `ArmConfig`/`train_arm`-specific — feature 011 recorded this as M2 and gave the meta-label package its own path. An SB3 policy shares neither. | Name the reproduction path as its own task rather than assuming `reproduce.py` extends. Same finding as 011's M2, same cause. |
| M3 | Inconsistency | MEDIUM | plan.md structure, tasks.md | The plan's structure lists `research/labelfree/rl/evaluate.py`, and T037/T038 assign it the screen gate — but no task *creates* it. T042–T046 use it for the pooled run. | Add an explicit implementation task for `evaluate.py`, or fold its creation into T037. |
| L1 | Terminology | LOW | spec.md, tasks.md | "Screening fold", "screen fold" and "f07" are used interchangeably; SC-019 says "the screening fold", the assumptions pin f07. | Name it once in the spec preamble. |

## Coverage Summary

| Requirement group | Covered | Notes |
|---|---|---|
| FR-001…FR-007 (environment) | Yes | Gate A + Gate B + US1 |
| FR-008…FR-011 (policy/training) | Yes | US2, US4 |
| FR-012…FR-019 (selection/evaluation) | Partial | FR-018 blocked by **C1** — no comparator artefact |
| FR-020…FR-024 (process) | Yes | Phase 9 |
| SC-001…SC-008 (protocol §5) | Tasked | SC-003 blocked by **C2** |
| SC-009…SC-019 (brief-specific) | Tasked | SC-009 blocked by **C1**; SC-018 vacuous as written (**H1**) |

**Unmapped tasks**: none. All 57 trace to at least one FR or SC.

## Constitution Alignment

No violations; 10/10 PASS with four tracked deviations in plan.md Complexity Tracking.

Principle VII is served well here — the staged budget, and the two gates before any training.
The risk to it is **H1**: a screen gate whose bar is undefined does not actually gate anything,
which would convert "validate cheaply first" into "run everything and decide afterwards".

## Metrics

- Functional requirements: **24** — covered 23, **blocked 1** (FR-018 via C1)
- Success criteria: **19** — tasked 19, of which **2 blocked** (SC-003, SC-009) and 1 vacuous (SC-018)
- Tasks: **57** — format-valid 57/57, no duplicate ids, 10 parallel, 9 `[GATED]`
- Research decisions: 8 · Entities: 8 · Contract interfaces: 7 · Quickstart scenarios: 9
- Ambiguity: 0 placeholders, 0 vague adjectives · Duplication: 0
- **Critical issues: 2**

## Next Actions

**Resolve C1 and C2 before `/speckit-implement`.** Both are blocking and both are the same
shape: a task that assumes a capability or artefact which does not exist. C1 is the more
serious — it disables the gate this entire specification was restructured around during
clarification, and it would surface only at the head-to-head, after the expensive screening and
pooled work.

**Resolve H1 and H2 in the same pass.** H1 must be settled before the screen runs or the gate is
decorative. H2 is cheap and prevents the axis class of bug the equivalence gate would otherwise
have to catch.

M1–M3 and L1 can follow; none blocks.

## Verdict

**NOT CLEAN — 2 CRITICAL, 2 HIGH.** Do not proceed to `/speckit-implement`.

Every finding is artefact-level; the feature's structure — two blocking gates, two decision
points before the pooled run, degeneracy handled in reporting — is sound.

**One finding worth escalating beyond this feature.** C2 exists because feature 011 identified
the same gap, wrote the fix as T043a, and then *cancelled* it along with everything else when
SC-009 killed the feature. The cancellation was correct for 011's own scope and wrong for the
study: a remediation task is not made unnecessary by the feature that discovered it being
killed. Cancelled-with-reason should distinguish **"no longer needed"** from **"still needed,
just not here"**, and the latter should be re-homed rather than dropped.

---

# Re-run — 2026-09-12 (post-remediation)

Fresh detection pass on the amended artefacts. All eight findings remediated.

## Remediation applied

| ID | Was | Fix |
|----|-----|-----|
| **C1** | The head-to-head named `sweep_best.json`, which does not exist — brief 1 persists no trained policy at all | New **T038a** re-trains brief 1's best configuration on the screening fold and persists it to `brief1_policy.npz`, budgeted as its own work item; **T038b** requires the saved policy to reload and reproduce its exposure path bit-for-bit; **T038c** records in the trial log that the comparator is a **fresh draw at this feature's seeds, not brief 1's reported figures**, and the contract and quickstart now say so. Quoting it as "brief 1's result" would be false |
| **C2** | `baselines.py` is single-fold; SC-003 unsatisfiable | New **T044a** re-homes the span mode, **T044b** requires a one-fold span to reproduce `--fold` exactly. The task text records *why* it is here: feature 011 wrote this fix as T043a and it was cancelled along with everything else when SC-009 killed that feature |
| **H1** | "Distinguishable from brief 1's policy" had no test, so the screen gate was vacuous | New **T036a** pins it **before the screen runs**: exposure paths must differ on **>5% of decision bars** at every seed |
| **H2** | The reward was indexed `i` in research/tasks and `t` in the data model | One symbol feature-wide — **`i`, the decision bar** — with the invariant stated once in R2: *the action taken on the observation ending at bar i earns the return between `open[i+1]` and `open[i+2]`*. `data-model.md` and T004 reindexed; no `w_t` survives in any live artefact |
| **M1** | "Turnover rises measurably" had no threshold | **≥2×**, pinned before the probe runs |
| **M2** | T050 assumed `reproduce.py` extends to an SB3 policy | Reproduction path **owned by this package**, with 011's identical finding cited |
| **M3** | `evaluate.py` used by six tasks, created by none | New **T035a** implements it |
| **L1** | "screening fold" / "f07" used interchangeably | Named once in the spec; f07 is the only fold the term denotes |

## Fresh Detection Pass

| Pass | Result |
|---|---|
| A. Duplication | 0 |
| B. Ambiguity | 0 placeholders, 0 vague adjectives |
| C. Underspecification | 0 — H1, M1 and M2 were the three, all now pinned |
| D. Constitution alignment | 0 violations; H1's fix restores Principle VII, which a vacuous screen gate would have hollowed out |
| E. Coverage gaps | **0 of 24 FRs, 0 of 19 SCs** |
| F. Inconsistency | 0 — one index symbol feature-wide; no stale comparator reference outside the historical findings table; every module named in tasks is declared in the plan |

## Metrics

- Functional requirements: **24** — covered **24**, blocked **0**
- Success criteria: **19** — covered **19**, blocked **0**, vacuous **0**
- Tasks: **64** (was 57; +7 remediation) — format-valid **64/64**, no duplicate ids, 12 parallel, 11 `[GATED]`
- Ambiguity **0** · Duplication **0**
- **Critical 0 · High 0 · Medium 0 · Low 0**

## Verdict

**CLEAN.** Ready for `/speckit-implement`.

Three things carried forward, none a defect:

1. **The comparator is a re-training, not a recovery.** SC-009 compares against a policy this
   feature trains, because brief 1 saved none. Its numbers will not equal brief 1's reported
   figures and the report must not present them as such.
2. **T051 will report DSR/PBO as `CANNOT BE EVALUATED`** — fourth feature into a trial log that
   stores only summary scalars.
3. **T034 remains the likeliest stopping point.** "Learn not to trade" is this environment's
   degenerate optimum, and a policy-gradient method finds it faster than gradient descent on a
   smooth objective.

## Process finding, beyond this feature

C2 exists because a remediation task was cancelled along with the feature that discovered it.
The cancellation was correct for feature 011's scope and wrong for the study: **a fix is not
made unnecessary by the feature that found it being killed.** `cancelled-with-reason` should
distinguish *"no longer needed"* from *"still needed, just not here"*, and the latter re-homed
rather than dropped. Worth adding to the protocol's close-out checklist so the next kill does
not silently drop its own fixes.

---

# Post-Implementation Analysis — 2026-09-13 (T052)

Run after the pooled walk-forward returned reward IQM −0.1301 and 5/15 folds positive.
47 complete, 9 cancelled with reasons, 8 open (close-out).

## Findings

| ID | Category | Severity | Summary | Recommendation |
|----|----------|----------|---------|----------------|
| Q1 | Spec accuracy | **HIGH** | **SC-009 was unsatisfiable twice over.** Clarification moved it off "brief 1's best *pooled* result" (which does not exist) onto "brief 1's *policy*" — which also does not exist, since brief 1 persists no weights. Re-training produced a comparator at exposure 0.0004, so the gate is *still* unsatisfiable, now for a third reason. | A gate naming another feature's artefact must verify the artefact exists **at specification time**. Two clarification rounds and one implementation round were spent discovering it did not. |
| Q2 | Process | **HIGH** | **Nine gated tasks cancelled** because the arm loses to cash. Correct for this feature — but feature 011's identical cancellation dropped a `baselines.py` fix the study still needed, and this feature inherited that hole. T044a/T044b are cancelled here *again*. | The span-mode fix has now been written and dropped **twice**. Re-home it to the protocol as standalone work, not as a task inside a brief that can be killed. |
| Q3 | Coverage | MEDIUM | **SC-004 unevaluable for the fourth consecutive feature.** 505 trials, 27 configurations, 5 seeds; DSR and CSCV need a time axis, the log stores summary scalars. | Escalated four times now. The schema change is overdue. |
| Q4 | Observation | MEDIUM | **R4 was refuted by measurement**, and the refutation corrects brief 02's diagnosis of its own defect. | Recorded in R4a and in brief 02's forward-looking notes. Worth a line in the protocol: a mechanism's failure mode belongs to the mechanism, not to how its parameters are obtained. |
| Q5 | Observation | LOW | The predicted failure (R8, "learn not to trade") **did not occur**: 1/75 runs below any floor, exposure 0.711, 312 trades/yr. The policy participates and loses. | None. A prediction that did not come true, recorded as such. |

## Gate Verification

| gate | evidence | verdict |
|---|---|---|
| A — env ≡ simulator | 0.00e+00 on all five cases (logic); 7.2e-07 vs a 5e-06 ceiling (precision) | PASS |
| B — the reward can be hacked | 8.46× vs the pinned ≥2.0× bar | PASS |
| liveness — every swept parameter | 4/4 LIVE, each in its active regime | PASS |
| screen — SC-018 | exposure 0.679 vs floor 0.05; paths differ on 60.0% vs >5% | FUND |
| admissibility — the pooled arm | exposure 0.711, 312 trades/yr | non-degenerate |

Every gate passed **before** the kill, which is what separates "this method loses" from "this
harness is broken". Gate B in particular found two environment bugs before it could pass.

## Constitution Alignment

10/10 PASS. Principle VI is again load-bearing: the learning-curve diagnostic ran *before* the
screen and changed both the budget and the seed count, and the two gates ran before any policy
was trained. Principle II is satisfied differently than planned — W1/W2/W3 was cancelled with
the other gated work, but the pooled run spans 15 anchored folds across 3.68 years, and no
acceptance claim is made that would require the distant-era evidence.

## Metrics

- Tasks: 47 complete, 9 cancelled-with-reason, 8 close-out
- Tests: 28 passing
- Trials: 505 pre-registered, audit clean, 0 problems
- Gates passed before the kill: 5 of 5
- Critical: **0** · High: 2 (Q1, Q2 — both process, both escalated) · Medium: 2 · Low: 1

## Verdict

**CLEAN.** No CRITICAL findings. The feature reached a well-evidenced negative with every gate
green beforehand, refuted one of its own predictions by measurement, and corrected a
misdiagnosis in an earlier brief.

The durable outputs outlast the verdict: a pooled trading environment proven equivalent to the
study's simulator and proven able to exhibit reward hacking; the shared admissibility gate; and
the finding that band-hold bimodality is mechanical rather than learned.
