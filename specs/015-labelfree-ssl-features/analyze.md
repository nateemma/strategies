# Analyze reports: 015-labelfree-ssl-features

FR-021 and SC-014 require a clean analyze report **before** implementation and **again after**.
Pass 1 is recorded here; pass 2 is appended on completion.

---

## Pass 1 — pre-implementation (2026-09-15)

**0 CRITICAL, 1 HIGH, 2 MEDIUM, 1 LOW.** All fixed; re-run clean at 43/43 requirements covered,
48 tasks.

| id | finding | fix |
|---|---|---|
| **H1** | **A gate nothing plausible could fail.** The pair-agnostic threshold was 0.5. Measured on the production feature set: ratio median **0.026**, max **0.120** on f07 — more than 4× headroom. A decorative PASS is worse than no gate, because it reads as corroboration. | Re-anchored to **0.30** on production's own measured behaviour across four folds. |
| **M1** | research.md R1 opened "The spec **sets** SC-001 against ρ = 0.15" — present tense, when SC-001 had been amended in the same commit. A reader hitting R1 first was told the spec says something it no longer said. | Reworded to past tense, noting the amendment. |
| **M2** | SC-001a required the primary screen be "identified before any is run", but nothing **verified** it. With `engineered×48` recorded in three documents it was true by convention, not by check. | T030 now reads the primary from a constant and **writes it into `screen.json` before any combination is scored**, so the record shows it was not chosen afterwards. |
| **L1** | plan.md's phase table had no row for the pair-agnosticism phase, though tasks.md gives it its own section. | Row added, naming the 0.30 gate. |

### The measurement refined the analyst's own recommendation

H1's report proposed **0.25** (≈2× f07's worst dimension). Measuring three more folds changed it:

| fold | median | p90 | max |
|---|---|---|---|
| f05 | 0.0306 | 0.0563 | 0.0992 |
| f07 | 0.0260 | 0.0732 | 0.1203 |
| f10 | 0.0410 | 0.1425 | **0.2297** |
| f13 | 0.0383 | 0.1580 | 0.1792 |

Production's worst single dimension reaches **0.2297**, so 0.25 would leave only **1.1×**
headroom — tight enough to reject an embedding no worse than the features it would sit beside.
**0.30** gives ~1.3×. The rule is *reject an embedding materially less pair-agnostic than
production*, which is a measurement rather than a constant either way.

### What the two earlier corrections were worth

Both would have been CRITICAL had they survived to implementation, and both came from the user's
questions rather than from this pass:

- **the bar.** SC-001 originally set a constant ρ = 0.15 at horizon 48, where the hand-built
  features themselves reach **0.083**. The feature would have produced a confident KILL against a
  bar its own baseline cannot clear.
- **the horizon justification.** The 6-bar screen was justified by a memory describing a
  **labelling bug fixed on 2026-09-03**. Retained on measured grounds instead: predictability
  decays from ρ 0.17 at H=3 to 0.06 at H=96, so a short horizon is where any signal is most
  visible.

A third correction was self-inflicted and recorded: R1's first draft generalised the ceiling from
**one fold with six pairs**, inside a decision whose entire point was *measure the bar, do not
quote it*.

### Constitution alignment (pass 1)

| principle | status |
|---|---|
| **III. Learnability is not edge** | **satisfied, and it is the feature's core tension** — the screen is explicitly a learnability measure; plan.md records it and T040 carries it into the report. Passing earns a backtest, never an acceptance. |
| **IV. Parameters actually live** | satisfied — the random control is this principle applied to feature width |
| **X. Structural fixes, not per-pair overrides** | satisfied, and strengthened by H1 — FR-012a is that failure in feature form, and the gate now discriminates |
| **II. Persistence** (NON-NEGOTIABLE) | conditional; reached only on a passing screen, then SC-006 binds |
| I, V, VI, VII, VIII, IX | no conflicts |

### Verified feasibility

The three-way split is comfortable: f05's training range is 1,179 days (A ≈ 825 / B ≈ 353), f14's
is 1,989. No fold is too short for the encoder and the ridge to have separate data.

### Metrics (pass 1, post-fix)

- requirements **43/43** cited · tasks **48** · unmapped 0 · duplication 0 · drift 0
- Critical 0 · High 0 · Medium 0 · Low 0

---

## Pass 2 — post-implementation (2026-09-15)

**2 CRITICAL, 1 HIGH, 1 MEDIUM.** All fixed.

| id | finding | fix |
|---|---|---|
| **C1** | **T009 was marked `[X]` and its test was never written.** tasks.md calls it load-bearing — *"the bar IS the gate, and a drifted bar would change every verdict with nothing failing"* — and only `test_ssl_causality.py` existed. **This is the second feature in a row where I marked a decisive test complete without writing it** (feature 013's T013 was the first, and it too guarded the comparison the feature turned on). | `test_ssl_bar.py`, five tests, three of which guard the **alignment** rather than the numbers: entry must be at t+1; perturbing bar *t*'s own price must **not** change its target; perturbing t+1 **must**. |
| **C2** | **`MEASURED_CEILING` still held the leaked generation-2 table** (`{3: 0.1465 … 48: 0.0499}`) against a causal bar of `{3: 0.0352 … 48: 0.0238}` — 2–4× too high. Had C1's test existed it would have failed, and the obvious "fix" is to edit the constant, which reintroduces the leak. | Replaced with causal values, both wrong generations recorded above it, and an explicit instruction: on a mismatch, check the target's alignment **before** touching the constant. |
| **H1** | **All four planning artefacts still asserted refuted claims as fact** — "ρ≈0.15 is a horizon-3-to-5 number" and "predictability decays with horizon". The result document was right; the documents a reader opens first were not. | Correction banner at the top of spec, research, data-model and quickstart, with the original text left standing beneath — a corrected document should show what it corrected. |
| **M1** | Quickstart never run end to end; Step 1 still quoted generation-1 numbers, two corrections stale. | Run; expectations replaced with measured values, and Step 1 flagged as the guide's most expensive step rather than the quick check its position implies. |

### A failure worth keeping

C1's new test failed on first run — correctly. The quickstart had regenerated `baseline.json`
from **5 folds** while the constant came from **10**, and nothing in the file recorded which.
A legitimate subset run was therefore indistinguishable from the bar drifting, and for a constant
that has been wrong twice, "looks like drift" is the one signal that must stay trustworthy.
`baseline.json` now records its fold set and the test skips rather than fails when the sets
differ, so a mismatch means drift and nothing else.

### Constitution alignment (pass 2)

| principle | status |
|---|---|
| **III. Learnability is not edge** | **satisfied, and it is why the feature ended cheaply** — the screen measured learnability, failed, and no backtest was earned |
| **IV. Parameters actually live** | satisfied — the random control sat at ±0.0015 throughout |
| **VI. Diagnose before tuning** | satisfied unusually well — the retention mechanism was measured *before* the screen confirmed it |
| **X. Structural fixes** | satisfied — pair-agnosticism passed at 0.2608 against 0.30, 0/320 dimensions over |
| I, II, V, VII, VIII, IX | no conflicts |

### Metrics (pass 2, post-fix)

- requirements 43/43 · tasks 43 done, **5 `[~]` not-reached** (no backtest earned — the designed
  outcome), 0 open
- tests **200 passing**
- Critical 0 · High 0 · Medium 0 · Low 0
