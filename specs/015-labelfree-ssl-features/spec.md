# Feature Specification: labelfree-ssl-features

**Feature Branch**: `015-labelfree-ssl-features`

**Created**: 2026-09-14

**Status**: Draft — clarified

**Input**: User description: "Feature: labelfree-ssl-features. This is research/label-free-study brief 07 … Implement research/labelfree/ssl/ with per-fold TS2Vec training on the pooled normalised feature stream, a random-Gaussian control block, and the walk-forward Spearman screen against forward return. Only if the screen beats rho=0.15 on nested OOS, add the features to an NNNC_MLX subclass and run the pooled walk-forward." Plus the user scope decision of 2026-09-14: **TS2Vec only**; Kronos-small and Chronos-Bolt out of scope.

## Context

The hypothesis is narrow and stated so it can be killed: **an embedding learned without labels
from the pooled feature stream carries information about forward returns that the hand-built
`DataframePopulator` indicators do not.**

### The prior is stronger than the brief states

The brief expects a null. This repository has already run a close analogue and recorded it:

| prior experiment | result | source |
|---|---|---|
| hand-built indicators | **ρ ≈ 0.15** — the ceiling | `us_spot_market_study.md:24` |
| 85-indicator battery (3× the project's features) + RF/MLP interactions | **no new signal**; ρ invariant, OOS R² *worse* | `us_spot_market_study.md:68` |
| **learned 1D-CNN on raw OHLCV** | **ρ ≈ 0.04–0.06 — worse than hand-crafted** | `us_spot_market_study.md:69` |

That third row is the one that matters. A learned representation over raw OHLCV has already been
tried here and scored **a third of the hand-built ceiling**. TS2Vec is a learned representation
over the same data. The study's own conclusion is that ρ≈0.15 is an **information limit, not a
modelling problem** — "you can't extract signal that isn't in price data".

**This does not make the brief pointless.** It makes it cheap and decisive: a self-supervised
contrastive objective is a genuinely different training signal from a supervised CNN, and the
screen costs one session. A negative closes the door with a number instead of an intuition.

### Why the screen exists and what it gates

The brief's design is a **cheap screen before any backtest**: Spearman ρ of a ridge projection of
the embedding against forward return, walk-forward, versus the hand-built ρ≈0.15. If the
projection does not beat that ceiling on nested out-of-sample data, **stop — no backtest is run**.
That ordering is the feature's main virtue and is preserved exactly.

### What the study has already established

| brief | verdict |
|---|---|
| 01 direct policy optimisation | NOT RESOLVED |
| 02 label warm-start | NOT RESOLVED — 92% of the advantage was turnover |
| 03 meta-labelling | KILL — nested AUC 0.4978 |
| 04 RL in a pooled env | KILL — pooled reward IQM −0.1301 |
| 05 gradient-free policy search | KILL — PBO 0.7128 over 48,015 evaluations |
| 06 generative stress harness | harness ACCEPTED — 38.6% of re-orderings lose |

## Clarifications

### Session 2026-09-14

- Q: Which encoders are in scope? → A: **TS2Vec only.** Kronos-small and Chronos-Bolt are out of
  scope: neither `transformers` nor any pretrained weights are installed, and adding them means
  new dependencies in the production venv plus an external HuggingFace fetch, for a brief whose
  own expected result is null. If the screen passes, they become worth their install cost and are
  revisited with evidence.
- Q: Which array framework? → A: **MLX.** An earlier draft said torch, which was an artefact of
  probing for torch as a Kronos/Chronos dependency and finding it present — not a comparison.
  MLX 0.32.2 is installed and running on GPU, and the repository's entire NN stack is MLX: nine
  MLX GAN implementations, `MLXBaseClassifier`/`Predictor`/`AnomalyDetector`, and
  `BaseNNStrategy` itself. Torch is present only transitively.

- Q: What does the encoder consume — the existing engineered features, or raw OHLCV? → A:
  **Both, reported separately.** They fail for different reasons worth telling apart: raw OHLCV
  is the input the 1D-CNN already failed on (ρ 0.04–0.06), so a null there confirms the
  information ceiling; engineered features cannot add information the indicators lack, so a null
  there says SSL finds no exploitable structure the classifier is missing. The screen is cheap
  enough to run both.
- Q: How does the embedding satisfy the repo's pair-agnostic feature rule
  (`AGENT_GUIDE.md:545`)? → A: **Verify and enforce.** Measure each embedding dimension's
  per-pair distribution and reject the embedding if it varies beyond a stated threshold. This is
  a real gate that can itself kill the embedding, and that would be a finding: a feature whose
  scale depends on the pair cannot be deployed in this repo however well it screens.
- Q: Which forward-return horizon does the screen score against? → A: **Both, reported
  separately** — the study's `DEFAULT_HORIZON = 48` and a short horizon of 6 bars.

  **Corrected 2026-09-15.** The 6-bar choice was originally justified by a memory saying "HORIZON
  is inert above PEAK_WINDOW; the effective horizon is 6 bars". That described a **labelling bug
  that was fixed on 2026-09-03** — `HORIZON` is live and 48 is the label's real forward reach.
  The 6-bar screen is retained on different grounds: measured predictability is strongly
  horizon-dependent (median ρ 0.17 at H=3 falling to 0.06 at H=96), so a short horizon is where
  any signal is most likely to be visible, and screening only at 48 could miss an embedding whose
  information is short-lived.

  **With four screens (2 inputs × 2 horizons) against one threshold, a "best of four" pass would
  be four chances to clear a bar that one should clear.** That is the multiple-testing failure
  this study measured at PBO 0.7128 in brief 05. So **one combination is pre-declared PRIMARY —
  engineered features at horizon 48**, matching the label the rest of the study uses and the
  purge length already assumed. The other three are **secondary and exploratory**: a pass on a
  secondary is a hypothesis for a follow-up brief, never an acceptance.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A screen that kills before any backtest (Priority: P1) 🎯 MVP

A researcher learns, for one session's work and without running a single backtest, whether a
label-free embedding carries forward-return information the hand-built features lack.

**Why this priority**: this is the whole feature. Everything downstream is gated on it, and the
brief's budget assumes it usually ends here.

**Independent Test**: score the embedding's ridge projection against forward return on nested
out-of-sample data and compare with the hand-built ρ ceiling.

**Acceptance Scenarios**:

1. **Given** a trained encoder, **When** its embedding is projected and scored, **Then** the
   result is a number compared against the hand-built ceiling, not an assertion.
2. **Given** a projection that does not beat the ceiling, **When** the screen concludes,
   **Then** **no backtest is run** and the feature is killed at that point.
3. **Given** the screen, **When** it is scored, **Then** the projection is fitted and evaluated
   **nested out-of-sample** — never on the data that trained the encoder or fitted the projection.

---

### User Story 2 - A control that separates information from capacity (Priority: P2)

A researcher compares the embedding against an **equally-sized block of random Gaussian
features** run through the identical pipeline.

**Why this priority**: adding any wide block of inputs can change a model's behaviour through
capacity alone. Without the control, "the embedding helped" and "more inputs helped" are
indistinguishable — and the brief makes a matching control a kill gate.

**Acceptance Scenarios**:

1. **Given** the control, **When** it is built, **Then** it has the **same dimensionality** as
   the embedding and passes through the same projection and scoring.
2. **Given** a screen result, **When** the random control scores comparably, **Then** the result
   is recorded as capacity, not information, and the feature is killed.

---

### User Story 3 - The encoder never sees the future (Priority: P3)

A researcher can verify that the encoder was fitted on a fold's **training range only** and that
a bar's embedding uses only bars at or before that bar.

**Why this priority**: a self-supervised encoder trained across a fold boundary would leak, and
the leak would look like signal — which is the one failure that would make a positive result
worthless.

**Acceptance Scenarios**:

1. **Given** a fold, **When** its encoder is fitted, **Then** only that fold's training range is
   used, with purge and embargo applied.
2. **Given** a bar's embedding, **When** it is computed, **Then** it depends on no bar after it,
   and this is demonstrated by a deletion test rather than asserted.

---

### User Story 4 - Only a passing screen earns a backtest (Priority: P4)

If and only if the screen beats the ceiling, the embedding is added to an `NNNC_MLX` **subclass**
and evaluated on the pooled walk-forward under protocol §5.

**Acceptance Scenarios**:

1. **Given** a passing screen, **When** the consumer is built, **Then** it is a subclass and the
   production class is never modified.
2. **Given** a failing screen, **When** the feature concludes, **Then** the result report is
   written anyway, recording the measured ρ.

### Edge Cases

- **The encoder collapses to a constant.** A degenerate embedding scores ρ≈0 and must be reported
  as encoder failure, not as evidence about the hypothesis.
- **The projection overfits the screen.** A ridge fitted and scored on the same data will beat any
  ceiling; nested evaluation is what prevents it, and the nesting must be checkable.
- **The embedding is pair-specific.** The repo prohibits features whose range varies by pair; an
  embedding with per-pair scale would violate that even if it screened well.
- **The random control also beats the ceiling.** That indicts the screen, not the embedding, and
  must halt the feature rather than promote either.
- **The screen passes narrowly.** A projection at ρ=0.16 against a 0.15 ceiling is inside the
  noise of both; the margin must be reported with an interval, not as a bare comparison.

## Requirements *(mandatory)*

### Functional Requirements

**Encoder**

- **FR-001**: The encoder MUST be trained **without labels**, on a fold's **training range only**.
- **FR-002**: A bar's embedding MUST depend on no bar after it, demonstrated by a deletion test.
- **FR-003**: Purge and embargo MUST be applied at the fold boundary, and the count applied
  recorded.
- **FR-004**: The encoder MUST be implemented in **MLX**, matching the repository's NN stack.
- **FR-005**: Per-step state evaluation MUST include every updated array — MLX's Metal backend
  trips its ~500K limit otherwise, a failure this repo has already recorded.
- **FR-006**: Kronos-small, Chronos-Bolt and any other pretrained model are **out of scope**; no
  new package and no external weight download.

**Screen**

- **FR-007**: The screen MUST score a **ridge projection** of the embedding against forward
  return by Spearman ρ, walk-forward.
- **FR-007a**: The screen MUST run over **both input types** (engineered features, raw OHLCV) and
  **both horizons** (48 bars, 6 bars), reporting each of the four separately.
- **FR-007b**: **Engineered features at horizon 48 is the PRIMARY screen** and the only one that
  can trigger a backtest. The other three are secondary and exploratory; a pass on a secondary is
  recorded as a hypothesis for a follow-up, never as an acceptance. Four screens against one
  threshold would otherwise be four chances to clear a bar meant to be cleared once.
- **FR-008**: The projection MUST be fitted and evaluated **nested out-of-sample**.
- **FR-009**: The screen MUST compare against the hand-built ceiling of **ρ ≈ 0.15**, and report
  the margin with an interval rather than a bare comparison.
- **FR-010**: If the screen does not beat the ceiling, **no backtest MAY be run**.
- **FR-011**: An **equally-sized random Gaussian block** MUST pass through the identical pipeline.
- **FR-012**: If the random control scores comparably, the result MUST be recorded as capacity
  rather than information.

**Pair-agnosticism**

- **FR-012a**: Each embedding dimension's **per-pair distribution MUST be measured**, and the
  embedding rejected if any dimension's between-pair location spread exceeds **0.30×** its
  within-pair scale (`AGENT_GUIDE.md:545` prohibits features whose ranges differ by pair). The
  threshold is anchored on the production feature set's own measured behaviour — worst dimension
  0.2297 across four folds — so it rejects an embedding materially less pair-agnostic than the
  features it would sit beside, rather than being an arbitrary constant.
- **FR-012b**: A rejection under FR-012a MUST be reported as a **finding** — an embedding that
  cannot be deployed in this repo, however well it screens — not silently normalised away.

**Consumer (only on a passing screen)**

- **FR-013**: The consumer MUST be an `NNNC_MLX` **subclass**; the production class is never
  modified.
- **FR-014**: The subclass MUST retrain — nothing is pinned, since the input set changes.
- **FR-015**: Scalers MUST be regenerated after the input set changes, or the validation curve is
  meaningless.

**Reporting**

- **FR-016**: The result report MUST be written whether the screen passes or fails.
- **FR-017**: The measured ρ MUST be reported for the embedding, the random control and the
  hand-built baseline, on the same data.
- **FR-018**: A degenerate encoder MUST be reported as encoder failure, not as evidence about the
  hypothesis.

**Process**

- **FR-019**: All runs MUST execute sequentially.
- **FR-020**: Only files this feature creates or modifies MAY be staged.
- **FR-021**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.

### Key Entities

- **Encoder**: the label-free model producing an embedding from a fold's training range.
- **Embedding**: the per-bar vector under test.
- **Projection**: the ridge map from embedding to a single score, fitted nested.
- **Control**: a random Gaussian block of identical width.
- **Ceiling**: the hand-built feature ρ ≈ 0.15 the projection must beat.

## Success Criteria *(mandatory)*

### The screen (the brief's own gate)

- **SC-001**: The **primary** screen — engineered features, horizon 48 — has its nested-OOS ridge
  projection beat **the hand-built features' own ρ at the same horizon on the same data**, with an
  interval that excludes it. The bar is **measured, never quoted**: ρ≈0.15 is a horizon-3-to-5
  number (median 0.1717 at H=3, 0.0826 at H=48 across five folds), so a constant would be the
  wrong bar at every horizon but one. See research.md R1.
- **SC-001a**: All four input × horizon combinations are reported, and the primary is **written
  into the result record before any combination is scored**, so the record itself shows it was not
  chosen afterwards.
- **SC-002**: The random Gaussian control scores materially below the embedding on the same
  pipeline; if it matches, the result is capacity and the feature is killed.
- **SC-003**: Causality is demonstrated by a deletion test, not asserted.
- **SC-003a**: The embedding's per-pair distribution is measured, and an embedding failing the
  pair-agnostic threshold is reported as such.
- **SC-004**: No backtest is run unless SC-001 and SC-002 both pass.

### Protocol §5 (only if the screen passes)

- **SC-005**: Pooled walk-forward annualised return at least the control plus **3.35 pp/yr**, with
  a bootstrap interval excluding zero.
- **SC-006**: Positive in at least 10 of 15 folds and in each of W1/W2/W3.
- **SC-007**: Beats all four mandatory baselines net of full costs.
- **SC-008**: Overfitting probability at most **0.2**, deflated risk-adjusted return above zero.
- **SC-009**: The return delta remains positive at 1.5× fees.
- **SC-010**: A separate process reproduces the accepted result from its saved configuration.

### Kill gates

- **SC-011**: If the primary screen's projection does not beat the ceiling, the verdict is
  **KILL** and the measured ρ is the finding — regardless of any secondary combination.
- **SC-012**: If the random control matches the embedding, the verdict is **KILL**, recorded as
  capacity rather than information.
- **SC-013**: If the screen passes but the pooled walk-forward does not clear SC-005, the verdict
  is **not resolved**, never "no effect".

### Process gates

- **SC-014**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-015**: The result report is written whatever the outcome, per protocol §6.
- **SC-016**: No pretrained model is downloaded and no new package installed.

## Assumptions

- The hand-built ceiling of ρ ≈ 0.15 is taken from `us_spot_market_study.md:24` and is the
  comparison the brief specifies. It is re-measured on this feature's own data rather than quoted,
  so the comparison is like-for-like.
- Fold geometry, purge and embargo come from `evaluation/foldplan.py` via
  `research/labelfree/folds.py`, unchanged.
- A negative result is the expected deliverable, consistent with briefs 01–05.

## Non-Goals

- Kronos-small, Chronos-Bolt, or any pretrained foundation model.
- Fine-tuning any pretrained model — the brief records that fine-tuning hurt in two studies.
- Installing new packages or downloading external weights.
- The adjacent approaches listed in `briefs/09-excluded-and-why.md`.
- Re-litigating the OHLCV information ceiling itself; this tests one specific way past it.
