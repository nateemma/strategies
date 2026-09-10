# Cross-Artefact Analysis Report — 009-labelfree-direct-policy

**Date**: 2026-09-10 · **Stage**: pre-implementation (required by
`research/label-free-study/README.md` step 5)
**Artefacts**: `spec.md`, `plan.md`, `tasks.md`, with `research.md`, `data-model.md`,
`contracts/`, `quickstart.md` as supporting context.
**Authority**: `.specify/memory/constitution.md` v1.0.0 (non-negotiable within this scope).

## Verdict

**CLEAN — 0 CRITICAL, 0 HIGH, 0 MEDIUM outstanding.** All 11 findings from the first pass were
remediated and re-verified. Implementation may proceed.

## Findings and resolutions

| ID | Category | Severity (initial) | Resolution | Status |
|----|----------|--------------------|------------|--------|
| D1 | Constitution — Principle VIII | CRITICAL | The 12-cell factorial is a joint search over three dimensions. `plan.md` claimed a silent PASS. Now recorded as an explicit **JUSTIFIED EXCEPTION** naming Principle VIII, with the reason (the objective × turnover-control *interaction* is the object of study — the literature's finding is that the naive objective overfits and only the structured version holds) and mandatory mitigations (pre-registered trials, DSR/PBO over the full count, PBO > 0.2 fails outright) | ✅ Closed |
| D2 | Constitution-adjacent — noise floor | CRITICAL | Screening 12 configurations on one fold sits inside the measured ±7.6pp single-window noise floor. Added **T031a** (pre-register the carry-forward rule *before* the run), **T032a** (bootstrap between-configuration deltas and state whether the top-2 selection resolves), and rewrote **T033** to select by rank only when resolvable, otherwise by the pre-registered rule | ✅ Closed |
| F1 | Inconsistency (factual) | HIGH | The brief's "fold f07, the richest" is wrong — measured from `regime/wf/pooled_nnnc.json`: **f08 = 71 trades, f07 = 58** (f07 ranks 2nd of 15). Screening moved to **f08** throughout. Recorded explicitly that f08 is chosen for trade count, **not** return: f08 returns −0.14% and f07 +8.08%, and screening on the profitable fold would bias selection toward configurations that only work in benign conditions | ✅ Closed |
| C1 | Coverage gap | HIGH | FR-001/FR-002 had zero tasks — nothing ingested or validated features. Added **T008a** (feature ingestion via `DataframePopulator` → `FeatureNormalizer` scalers) and **T008b** (include-list validation rejecting raw price/volume, with a test) | ✅ Closed |
| C2 | Coverage gap | HIGH | SC-010's overfitting kill gate was never computed. Added **T040a** computing the W2−W1 advantage differential against 3.35pp/yr and halting the configuration when exceeded | ✅ Closed |
| A1 | Ambiguity | MEDIUM | SC-002's clean-run bar was "far below that level" — untestable. First quantified as a multiple of the control's W1 return; **superseded during implementation 2026-09-10** when that proved incommensurable (a policy's return on its own fold range cannot be compared to a fixed multiple of a different window's figure — measured, +289% from beta alone on f08's 4-year training range vs −16.9% for a good policy on its 90-day test range). Replaced by a PAIRED criterion: leaky exceeds clean by ≥2.0 log units on the same range and seed, and clean does not beat its **exposure-matched** benchmark by more than the floor. The benchmark had to be exposure-matched too: a 41%-invested long-only policy beats a fully-invested benchmark by 17.2pp in a −34% market purely by holding less, which is beta, not leakage | ✅ Closed (reformulated) |
| C3 | Coverage gap | MEDIUM | FR-014a (training must not model the deployment exit stack) had no verifying task. Added **T021a** asserting the training path exposes no take-profit, adaptive-stop or grace exit | ✅ Closed |
| C4 | Coverage gap | MEDIUM | SC-013's process outcome was unrepresented. T050 now requires the report to record that the simulator passes to brief 2 and the family stops, when the verdict is "not resolved" | ✅ Closed |
| F2 | Terminology drift | MEDIUM | `spec.md` used "production backtester"/"production control" while `plan.md`/`tasks.md` used "arbiter"/"the control", with the equivalence never stated. Added a **Terminology** glossary to `spec.md` pinning arbiter, control and screening fold | ✅ Closed |
| C5 | Coverage | LOW | T032 and T037 carried no requirement citation. Both now cite FR-019/FR-020/FR-030 | ✅ Closed |
| C6 | Coverage | LOW | FR-025 was covered in substance but uncited. Now cited in T050 | ✅ Closed |

## Metrics (post-remediation)

| Metric | Before | After |
|---|---|---|
| Total requirements | 57 (39 FR + 18 SC) | 57 |
| Total tasks | 51 | **57** (+6) |
| Requirement coverage | 87.7% | **100.0%** |
| CRITICAL issues | 2 | **0** |
| Ambiguity count | 3 | **0** |
| Duplication count | 0 | 0 |
| Unresolved placeholders | 0 | 0 |
| Malformed tasks | 0 | 0 |
| RUN tasks wrongly marked `[P]` | 0 | 0 |

## Constitution alignment

All ten principles, the repository constraints and the seven workflow gates were evaluated.

- **Principle VIII** carries a documented exception, which the constitution expressly permits
  ("revised **or** carry an explicit, written justification naming the principle"). It is recorded
  in `plan.md` Complexity Tracking, not taken silently.
- **Principle IX** is N/A — a single policy, not a multi-factor book.
- All others PASS.

## Deliberate deviations from the Spec Kit template (not defects)

1. **User stories are strictly sequential, not independently deliverable.** Each is a gate on the
   *credibility* of the next: an uncalibrated simulator makes every downstream number fiction, and
   an unproven gradient path makes a null sweep uninterpretable. Recorded in `plan.md`.
2. **`[P]` marks file authoring only.** AGENT_GUIDE requires every job to run sequentially, so no
   `RUN` task is ever parallel — verified mechanically.
3. **Letter-suffixed task IDs** (T008a, T032a…) were used for insertions to preserve execution
   order without renumbering 51 cross-referenced tasks.

## Residual risks carried into implementation

These are recorded, not resolved — they are properties of the data, not defects in the artefacts.

1. **The distant window trains on 9 of 10 pairs**, six of them on partial history (345-533 days vs
   729). Only XRP is excluded, for a 911-day delisting gap. **Corrected 2026-09-10 during
   implementation**: this report originally said "4 of 10, ZEC excluded", derived from a start-date
   test that wrongly counts XRP as covered and wrongly drops six pairs with usable partial history.
   ZEC is present with 352 days, so the feared missing-pair confound does not apply; what remains is
   a data-quantity asymmetry to report. Per `research.md` R10 and T040.
2. **The screening fold has 71 trades**, still below the ~100 that produced the ±7.6pp noise floor.
   T032a exists to make the screen's resolvability an explicit, reported measurement rather than an
   assumption.
3. **Byte-identical reproduction on GPU is required but only smoke-tested.** A seeded MLX matmul
   reproduced exactly across processes; a full training loop is a stronger test. T027 checks it
   early, and the instruction is to escalate rather than relax SC-008.

## Next action

Proceed to `/speckit-implement`. Re-run `/speckit-analyze` on completion (T048) and save it over
this file, per the study README step 8.
