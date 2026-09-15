# Research: labelfree-ssl-features (brief 07)

Phase 0. Every decision carries the measurement or code reading behind it.

---

## R1 — MEASURED: the bar cannot be a constant 0.15, and pre-declaring it as one was wrong

**This is the most important decision in the document and it corrects the spec.**

The spec **originally set** SC-001 against **ρ = 0.15**, quoted from `us_spot_market_study.md:24`,
with the primary screen at **horizon 48**. *(SC-001 has since been amended to the measured
per-horizon bar described below; this paragraph records what was corrected and why.)* Measured on
our own fold panels before writing this plan:

**Measured across 5 folds × 6 horizons, 10 pairs each** (ridge over the 24-column
`FeatureNormalizer.include_list`, fitted on the first 60% of a fold's test range and scored on
the remainder; f05, f07, f09, f11, f13):

| H (bars) | median ρ_oos | min | max |
|---|---|---|---|
| 3 | **0.1717** | 0.1606 | 0.2346 |
| 6 | 0.1127 | 0.0970 | 0.1601 |
| 12 | 0.0944 | 0.0860 | 0.1096 |
| 24 | 0.0720 | 0.0627 | 0.1212 |
| 48 | **0.0826** | 0.0467 | 0.0978 |
| 96 | 0.0583 | 0.0488 | 0.0889 |

*(Raw data: `research/labelfree/ceiling_by_horizon.json`.)*

**ρ ≈ 0.15 IS NOT A GENERAL CONSTANT. It is a horizon-3-to-5 number.** Predictability decays
roughly monotonically with forward horizon — ~0.17 at 3 bars to ~0.06 at 96 — and 0.15 sits
between H=3 and H=6.

**A correction to this document's own first draft.** It reported 0.1594 at H=6 and called that a
reproduction of the ceiling. That was **one fold (f07) with six pairs on a single split**, and it
was the top of the across-fold range: the H=6 median over five folds with ten pairs is **0.1127**
(range 0.097–0.160). The conclusion below survives — the bar must be measured, not quoted — but
generalising a ceiling from one fold was the same error as quoting one from another study.

**So the pre-declared primary screen was incoherent.** It asked the embedding to beat 0.15 at a
horizon where the hand-built features themselves reach **0.083 (median, 0.047–0.098)**. That is a bar the baseline cannot
clear, and an embedding failing it would be failing for a reason that has nothing to do with the
hypothesis. This study has now hit that shape three times — brief 06's calibration comparing two
different evaluators, its drawdown band pointing at the method's own bias, and now this.

**Decision**: the comparison bar is **the hand-built features' own ρ, measured on the same data,
the same pipeline and the same horizon as the embedding it is compared against**. Not a constant.
Every screen reports `rho_embedding`, `rho_handbuilt`, `rho_control` computed identically, and
the gate is the *difference* between the first two.

SC-001 was amended accordingly in the same commit that added this decision, rather than left as a
note for the analyze pass.

---

## R2 — The screen needs a THREE-way split, not a two-way one

**Decision**: encoder on split A, ridge on split B, score on split C, all within a fold and in
time order.

**Why**: there are two fittable objects, and the obvious design leaks through the second. The
encoder is fitted without labels, so it is tempting to treat its output as given and fit the
ridge on the same range — but the ridge *is* supervised, and a ridge fitted and scored on the
same bars will beat any ceiling. That is the spec's own "projection overfits the screen" edge
case, and it is how a null result becomes a false positive.

```
fold training range          →  A: encoder (no labels)
                             →  B: ridge projection (supervised, labels from B only)
fold test range              →  C: score        <- the only number reported
```

Purge and embargo apply at every boundary, not only at A|B — a label at the end of B reaches
`horizon` bars into C.

---

## R3 — TS2Vec in MLX: what it actually is

**Decision**: implement the encoder in MLX, matching the repository's NN stack.

TS2Vec (Yue et al. 2022) is three components and roughly 300 lines:

1. a **dilated causal CNN** over the input window — causal matters here, it is what makes FR-002
   satisfiable by construction rather than by a later check
2. **random cropping** into two overlapping views per instance
3. a **hierarchical contrastive loss** — temporal (same series, different timestamps) and
   instance-wise (different series, same timestamp), applied at successive pooling levels

**MLX specifics this repo has already paid for:**

- **Every updated array must be in the per-step `mx.eval`.** Omitting one lets the graph grow
  until Metal's ~500K limit trips. Recorded in this repo's memory as a real failure.
- **Seed via `MLX_SEED`**, not a framework-level `train_seed` — the latter was measured inert for
  multi-task models here.

**Why not torch**, since it is installed: it is present only as a transitive dependency. Nine MLX
GAN implementations, `MLXBaseClassifier`/`Predictor`/`AnomalyDetector` and `BaseNNStrategy`
itself are MLX, and MLX 0.32.2 runs on the GPU here. An earlier draft said torch; that was an
artefact of probing for it as a Kronos/Chronos dependency, not a comparison.

---

## R4 — The control must pass through the identical pipeline, ridge included

**Decision**: an equally-wide Gaussian block, fitted and scored by the same ridge on the same
splits.

**Why it has to be identical and not merely similar**: the ridge has capacity proportional to its
input width. A 320-dimensional embedding and a 6-column indicator set are not comparable through
a shared projection, and any difference between them would partly measure width. The control
isolates that: same width as the embedding, same ridge, same splits, no information.

**The gate this creates**: if the control approaches the embedding, the result is capacity rather
than information (FR-012) — and if the control approaches the *hand-built* bar, the screen itself
is broken and nothing it says is usable.

---

## R5 — Pair-agnosticism: the threshold, pre-registered

**Decision**: for each embedding dimension, compute its per-pair mean and standard deviation
across the fold universe, and reject the embedding if

```
max over dimensions of  ( std across pairs of per-pair mean ) / ( mean across pairs of per-pair std )  >  0.30
```

**The threshold is anchored on what production actually achieves, measured across four folds:**

| fold | pairs | median ratio | p90 | max |
|---|---|---|---|---|
| f05 | 10 | 0.0306 | 0.0563 | 0.0992 |
| f07 | 10 | 0.0260 | 0.0732 | 0.1203 |
| f10 | 10 | 0.0410 | 0.1425 | **0.2297** |
| f13 | 10 | 0.0383 | 0.1580 | 0.1792 |

**0.30 gives ~1.3× headroom over production's worst single dimension (0.2297).** An earlier draft
used **0.5**, which the pre-implementation analyze pass found to be **more than 4× production's
median-fold maximum** — a gate nothing plausible could fail, which is worse than no gate because
it reads as corroboration. The analyze pass then proposed 0.25, and measuring a *second* and
*third* fold refined that too: at 0.25 the headroom is only 1.1×, tight enough to reject an
embedding no worse than production.

So the rule is: **reject an embedding materially less pair-agnostic than the features it would
sit beside.** Not an arbitrary constant, and not a number that flatters.

**Why a ratio and not an absolute**: embedding dimensions have arbitrary units, so an absolute
threshold would be meaningless. The ratio asks the question `AGENT_GUIDE.md:545` actually asks —
does this feature mean the same thing on every pair?

**Pre-registered at 0.30 before any encoder is trained.** It is anchored on the *production*
distribution, which is legitimate to look at — production is not the thing under test — and the
reason it is fixed now is the one that governs feature 013's liveness thresholds: a threshold
chosen after seeing the *embedding's* distribution is not a threshold.

**A rejection is a finding, not an inconvenience** (FR-012b). An embedding whose scale depends on
the pair cannot be deployed in this repo however well it screens, and saying so is a result about
the method.

---

## R6 — Causality by deletion, because the architecture's guarantee is not evidence

**Decision**: demonstrate FR-002 by deleting future bars and showing the embedding is unchanged.

The dilated CNN is causal by construction, which is an argument, not a measurement. The test:
compute a bar's embedding from the full series; recompute it with every bar after it removed;
require bit-equality. Feature 011 used exactly this shape (causality-by-deletion plus a planted
leak) and it is the only check that distinguishes "cannot see the future" from "happens not to".

**Add the planted leak too**: inject a copy of the forward return as an extra input channel and
confirm the screen's ρ jumps. A causality test that cannot detect a leak it was handed is not
evidence.

---

## R7 — Four screens, one primary, decided before any is run

The spec's clarifications settled this and it is restated because it governs the task order:
engineered features × horizon 48 is **primary**; the other three combinations are **secondary and
exploratory**.

R1 complicates it, and the complication forced an amendment. The corrected per-pair temporal bar
is **0.0499 at H=48 (min −0.0169)** and **0.1003 at H=6 (min 0.0715)**. At H=48 the hand-built
features are barely distinguishable from noise, so a gate defined as "beat them there" is close
to "beat zero" — weak in the dangerous direction for a brief expecting a null.

**The primary is therefore a conjunction: `engineered` must beat its own bar at H=48 AND at
H=6.** Amended after the bar was measured and before any encoder was trained. It raises the gate;
the record says so explicitly, because changing a pre-registered decision after seeing data must
be visible whichever direction it moves. Since
the bar is now measured per combination (R1), each screen is self-consistent — but the *primary*
remains the one declared in advance, and a secondary that passes is a hypothesis for a follow-up
brief, never an acceptance. Four chances at one threshold is the multiple-testing failure brief 05
measured at PBO 0.7128.

---

## R8 — The expected outcome, stated before the work

Null, and the prior is stronger than the brief states:

- a **learned 1D-CNN on raw OHLCV** scored **ρ 0.04–0.06** in this repo, *worse* than hand-crafted
  (`us_spot_market_study.md:69`)
- an **85-indicator battery**, three times the project's feature count, left ρ **invariant** while
  OOS R² got worse (`:68`)
- the study's own conclusion is that ρ≈0.15 is an **information limit, not a modelling problem**

TS2Vec is another learned representation over the same data. What makes it worth one session is
that a **contrastive self-supervised objective is a different training signal** from a supervised
CNN — it optimises for representation consistency rather than for predicting the label — and the
screen kills it for a session's work rather than a week's.

**What would change the conclusion**: the embedding's ρ exceeding the hand-built bar *at the same
horizon on the same data*, with an interval excluding it, and the random control well below both.
