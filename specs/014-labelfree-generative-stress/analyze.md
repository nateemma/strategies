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

---

## Pass 2 — post-implementation (2026-09-14)

**0 CRITICAL, 0 HIGH, 2 MEDIUM, 1 LOW.** All fixed.

| id | finding | fix |
|---|---|---|
| **P1** | **research.md R2 still asserted a prediction the implementation refuted.** R2 argued SC-001 was "expected to pass" and "biased heavily toward passing". The gate *failed* on its first run (`es_ratio` 1.8385, `dd_ratio` 1.7636) and caught a real bug. Leaving the prediction standing without its outcome misleads the next reader and understates what the gate demonstrated. | R2 annotated with the refutation, the analysis left standing as written — a prediction is only worth recording if it stays legible when it misses. |
| **P2** | **The block-length selection rule lived only in the result, not the design.** "Shortest that passes, not best calibrated" was discovered during implementation — the sweep script had picked L=288 on precisely the wrong rule — and was recorded in `calibration.json` and the result document, but not in `data-model.md`. A re-run would have had to rediscover it. | Recorded in data-model.md §2 with the monotonicity evidence (KS 0.0252 → 0.0119) and the degenerate-limit argument. |
| **P3** | A full five-point sweep leaves ~490 MB of banks on disk. Correctly gitignored, but undocumented. | Noted in quickstart.md Step 5. |

### Verification performed

- **36/36 requirements** cited by a task; all 46 tasks complete
- every module probed against what the spec says it does: calibration holds the evaluator
  constant, the shared block index is implemented as `idx[i] != idx[i-1] + 1`, admissibility is
  routed rather than reimplemented, the verdict field is not named `passed`, and the
  pre-registered tolerances are unchanged at `KS_MAX = 0.10`
- quickstart run end to end (T040); its stated expectations replaced with measured numbers
- **191 tests pass**

### What pass 1's fixes were worth

Both of pass 1's serious findings would have produced a wrong verdict rather than a broken run.
**C1** would have had the feature's only gate comparing surrogate-on-synthetic against
freqtrade-on-real, attributing a known −5.30 ± 23.46 pp evaluator gap to the generator — and
since a calibration failure is a kill, the feature could have died of a confound. **H1** would
have let independent per-pair resampling understate joint drawdown, in the direction the
`dd_ratio` band was most permissive about.

Neither was visible to a wiring check. Both are errors in *what gets compared*.

### Constitution alignment (pass 2)

| principle | status |
|---|---|
| **II. Persistence** (NON-NEGOTIABLE) | **N/A, and it held.** The feature shipped a veto instrument, not a candidate; the stress result is explicitly labelled "not an endorsement" and SC-009 was not violated. |
| **IV. Parameters actually live** | satisfied — `L` swept across five values, all reported, with the selection rule now in the design |
| **V. Lookahead-clean is not realistic** | satisfied — cost and slippage ride inside the resampled blocks |
| I, III, VI, VII, VIII, IX, X | no conflicts |

### Metrics (pass 2, post-fix)

- requirements 36/36 · tasks 46/46 · tests 191 passing
- unmapped tasks 0 · duplication 0 · terminology drift 0
- Critical 0 · High 0 · Medium 0 · Low 0
