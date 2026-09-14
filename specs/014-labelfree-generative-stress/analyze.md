# Analyze reports: 014-labelfree-generative-stress

FR-020 and SC-010 require a clean analyze report **before** implementation and **again after**.
Pass 1 is recorded here; pass 2 is appended on completion.

---

## Pass 1 — pre-implementation (2026-09-14)

**1 CRITICAL, 2 HIGH, 2 MEDIUM, 1 LOW.** All fixed; re-run clean at 36/36 requirements covered.

| id | finding | fix |
|---|---|---|
| **C1** | **The calibration compared two different evaluators.** Synthetic P&L comes from the surrogate; the 378-trade reference is *freqtrade* output. Feature 013 measured that gap at **−5.30 ± 23.46 pp** with residual exit-mix divergence. A failed KS test could mean "the bootstrap is bad" *or* "the surrogate differs from freqtrade", and the design could not tell them apart — while SC-001 is the only gate and a failure is a kill. | Calibrate **surrogate-on-synthetic vs surrogate-on-real-panels**, holding the evaluator constant. The archive is retained as a secondary sanity reference, reported but never gating. |
| **H1** | **The method's known bias pointed at the gate it had to pass.** Independent per-pair resampling systematically *understates joint drawdown* — and `dd_ratio` is a calibration statistic, banded 0.6–1.6, i.e. most permissive in the direction the rationale itself called dangerous. | Fixed at the cause, not the band: **one block index shared across all pairs**, so pairs that crashed together still do. The band was also tightened to 0.7 to match ES. |
| **H2** | FR-004 required capability "determined from its **actual output**" while T008 forbade instantiating any model. Both cannot hold — actual output requires running `generate()`. The pre-spec measurement read source, which is weaker than the spec claimed. | FR-004/SC-003 reworded to "the implementation's **return contract in the running code, cited by `file:line`**" — what the tasks can actually evidence. |
| **M1** | T031 hand-rolled a participation check, when the study **centralised exactly that** because it kept being rewritten per site with each rewrite guarding only one side. | FR-013 and T031 now route through `admissibility.compare` with the incumbent-relative exposure floor feature 013 added. |
| **M2** | The block-length sweep — the feature's only parameter, swept under Principle IV — had no FR or SC backing it. | Added FR-003a and SC-006a. |
| **L1** | "≥1,000 paths" never defined the unit; a reader could take it as per-pair. | data-model.md §1 states a path is one complete resampled panel across all pairs. |

### What this pass is worth recording for

C1 and H1 are the same shape, and it is the shape this study keeps meeting: **a gate that would
have failed for a reason unrelated to what it was measuring.** C1 would have attributed evaluator
error to the generator; H1 would have let a known method bias push the drawdown statistic toward
the permissive edge of its own band. Both were invisible to any wiring check — they are errors in
*what gets compared*, not in whether the code runs. Feature 013 had two of the same class (a
return convention that flipped a sign, and a participation floor that rejected production).

### Constitution alignment (pass 1)

| principle | status |
|---|---|
| **II. Persistence** (NON-NEGOTIABLE) | **N/A, and scrutinised.** This feature ships no candidate and SC-009 forbids promotion on its output; Principle II binds the policies it vetoes (clarification Q1). plan.md records that if a stress result is ever used to *support* rather than veto, the row becomes a violation. |
| **IV. Parameters actually live** | satisfied — the block length is swept, now backed by FR-003a/SC-006a (M2) |
| **V. Lookahead-clean is not realistic** | satisfied — cost and slippage ride inside the resampled blocks |
| I, III, VI, VII, VIII, IX, X | no conflicts |

### Metrics (pass 1, post-fix)

- requirements **36/36** cited by a task
- tasks **46** (44 + T018a, T024a)
- unmapped tasks 0 · duplication 0 · terminology drift 0
- Critical 0 · High 0 · Medium 0 · Low 0
