# Feature Specification: Single-Pair (LINK) Variant of NNNC_MLX

**Feature Branch**: `008-single-pair-variant`
**Created**: 2026-09-10
**Status**: Draft
**Input**: Test whether the most consistently-contributing pair is worth modelling on its own.

## Context

The question is whether the strategy's edge concentrates in one pair strongly enough that a
single-pair model would do at least as well on that pair — in which case the other ten are
dilution.

### There is no persistent dominant pair — measured 2026-09-10

| pair | W1 %P&L | W2 %P&L | W3 %P&L |
|---|---|---|---|
| ZEC | **44.9%** (36 tr) | 7.0% | 2.8% |
| XRP | 23.6% | **38.3%** | −3.1% |
| NEAR | 5.0% | 6.9% | **27.5%** |
| **LINK** | **11.6%** | **9.5%** | **26.4%** |
| SUI | 4.0% | 21.6% | 3.2% |
| AVAX | — | −2.8% | 21.9% |

Top-3 by window: **W1** ZEC/XRP/LINK · **W2** XRP/SUI/DOT · **W3** NEAR/LINK/AVAX.

**Dominance rotates.** ZEC is 45% of W1 and 2.8% of W3; XRP is 38% of W2 and negative in W3. This
is itself a finding: "the edge is ZEC" is a W1 observation, not a property of the strategy, and a
model trained on last window's dominant pair is fitting a regime.

**LINK is chosen instead** because it is the only pair that contributes materially in all three
windows and never collapses. It is nobody's top pair in W1 or W2, which is precisely what makes it
the honest test of "is a single-pair model viable" rather than "was ZEC special last year".

### LINK's profile

| window | trades | per-trade | win rate | % of window P&L |
|---|---|---|---|---|
| W1 (acceptance) | **8** | +2.270% (sd 1.47) | 100% | 11.6% |
| W2 | 26 | +0.454% (sd 3.67) | 73% | 9.5% |
| W3 | **5** | +2.500% (sd 0.98) | 100% | 26.4% |

LINK history begins 2022-01-14, later than the 2021-01-01 span the 11-pair walk-forward used.

### The power problem defines this feature's shape

W1's measured bootstrap noise floor is **7.633pp on 99 trades**. LINK contributes **8 trades on
W1**. A single-window P&L comparison is not merely underpowered, it is meaningless, and the
comparison tool's 20-differing-trade minimum will refuse to report a CI at all.

**Success is therefore NOT a P&L win.** Two questions remain answerable at this trade count and the
feature is built around them:

- **Which trades** each model selects — a set comparison, well-determined at n=8.
- **Per-trade expectancy** — at W1's n=8 and sd 1.47, the standard error is ≈0.52pp, so an
  expectancy of +2.27% is resolvable as positive even though the *total* return is not.

Total return is not resolvable and must not be used to declare a winner.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A LINK-only model exists and is honestly configured (Priority: P1)

A researcher trains a LINK-only variant sharing the 11-pair model's entry gate, exit stack and
label exactly, differing only in the pairs it sees.

**Why this priority**: Every later question requires the two arms to differ in exactly one thing.
Five strategies were found running completely ungated the day before this spec, invalidating every
comparison made with them.

**Independent Test**: Train it, confirm a model artefact exists, confirm its gate and exit stack
match the control's.

**Acceptance Scenarios**:

1. **Given** a LINK-only universe, **When** the variant trains, **Then** a model artefact exists in
   its own storage location, distinct from the 11-pair model's.
2. **Given** the variant runs, **When** configuration is inspected, **Then** every entry-gate and
   exit-stack value equals the 11-pair model's.
3. **Given** the variant runs, **When** the log is checked, **Then** the ungated-gate warning does
   **not** appear. If it does, the run is void.

### User Story 2 - Which LINK trades each model takes (Priority: P1)

The researcher compares the two models' LINK trade sets as sets — shared, variant-only,
control-only — not by aggregate return.

**Why this priority**: The question that survives n=8. Whether the single-pair model *selects the
same opportunities* is answerable; whether it earns more is not.

**Independent Test**: Restrict both models' trades to LINK and report the three-way split.

**Acceptance Scenarios**:

1. **Given** both models on the same window, **When** LINK trades are paired on identity, **Then**
   shared, variant-only and control-only counts are reported.
2. **Given** identical trade sets, **Then** it is reported as the models selecting the same
   opportunities — a finding, not a null.
3. **Given** they differ, **Then** each differing trade's outcome is reported per side.

### User Story 3 - Per-trade expectancy on LINK (Priority: P1)

The researcher establishes whether single-pair training changes per-trade expectancy on LINK.

**Why this priority**: Better determined at low n than total return, because it is not multiplied
by a small and variable trade count. This is the feature's most likely source of a real answer.

**Independent Test**: Compare mean per-trade return on LINK trades between the two models, with
intervals.

**Acceptance Scenarios**:

1. **Given** both models' LINK trades, **When** compared, **Then** per-trade expectancy is reported
   for each with a bootstrap interval.
2. **Given** overlapping intervals, **Then** the result is reported as indistinguishable, with the
   intervals quoted.

### User Story 4 - Pooled walk-forward evaluation (Priority: P3)

The researcher evaluates the variant across pooled out-of-sample folds.

**Why this priority**: The only instrument with a chance on returns, but expected to remain thin —
LINK's rate implies roughly 48 pooled trades against the 378 the 11-pair run produced. P3 because
US2 and US3 must carry the feature if this cannot resolve.

**Independent Test**: Run pooled folds and report the achieved floor alongside any delta.

**Acceptance Scenarios**:

1. **Given** a fold plan, **When** folds run, **Then** every test range is strictly out-of-sample
   and none predates LINK's listing.
2. **Given** pooled results, **Then** pooled trade count and achieved floor are stated with the
   delta.
3. **Given** a delta smaller than the achieved floor, **Then** it is reported as unresolved.

### Edge Cases

- A window or fold yields zero LINK trades — recorded as zero-trade, never zero return.
- Both models produce byte-identical LINK trades — inconclusive for P&L, a positive finding for US2.
- Training on one pair is degenerate for cross-pair aggregation; the run must be verified to have
  produced a model rather than assumed.
- A fold's training range extends before LINK's listing, so it trains on less history than planned.
- W1's 8 LINK trades are 100% winners; a variant that adds even one loser changes the win rate by
  11 points on no real evidence. Win rate must not be read as a quality signal at this n.
- The answer differs between windows — that disagreement is the result, not a problem to resolve.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The variant MUST be trained and evaluated on a LINK-only pair universe.
- **FR-002**: The variant MUST store its model separately from the 11-pair model, with no
  possibility of either overwriting the other.
- **FR-003**: The variant MUST carry its own complete parameter set — entry gate AND exit stack —
  matching the 11-pair model's exactly.
- **FR-004**: Any run in which the ungated-gate warning appears MUST be discarded, not reported.
- **FR-005**: Training MUST be verified by the existence of a model artefact, not exit status alone.
- **FR-006**: The two models' LINK trade sets MUST be compared as sets — shared, variant-only,
  control-only — on every window evaluated.
- **FR-007**: Per-trade expectancy on LINK MUST be reported for both models with an interval.
- **FR-008**: Every reported delta MUST carry the applicable noise floor and a confidence interval.
- **FR-009**: A delta smaller than the applicable floor MUST be reported as unresolved.
- **FR-010**: Identical trade sets MUST be reported as inconclusive for P&L and as a positive
  selection finding for US2.
- **FR-011**: A zero-trade window or fold MUST be recorded as zero-trade, never as zero return.
- **FR-012**: Results MUST be reported per window, with the pair-dominance rotation stated
  explicitly rather than averaged away.
- **FR-013**: Pooled fold evaluation MUST exclude any fold whose test range predates LINK's listing.
  **CONDITIONAL** — see the scope note below.
- **FR-014**: Pooled fold evaluation MUST report the achieved noise floor on the pooled set.
  **CONDITIONAL** — see the scope note below.
- **FR-015**: Classifier metrics MUST NOT be an input to any verdict; they may be reported as
  context.
- **FR-016**: Where a model is trained, two seeds MUST be used, and identical trades across seeds
  MUST be reported as one measurement rather than two.
- **FR-017**: All work MUST run sequentially.
- **FR-018**: No shared framework code, evaluation harness or comparison tool may be modified.
- **FR-019**: No entry-gate, exit, label or architecture parameter may be changed. The variant
  differs from the control in its pair universe and nothing else.
- **FR-020**: A run whose fold set is incomplete MUST be reported as partial. **CONDITIONAL** —
  see the scope note below.

> **SCOPE NOTE — the conditional cluster.** FR-013, FR-014, FR-020 and SC-006 apply only if pooled
> fold evaluation (US4) actually runs, and US4 is **gated**: it runs only if the two models select
> DIFFERENT LINK trades. If they select the same trades, pooled folds cannot add information, US4
> is skipped and recorded as skipped, and these four are vacuously satisfied. A reader must not
> take "8/8 success criteria" to mean all eight were exercised.
- **FR-021**: Win rate MUST NOT be reported as a quality signal on any window with fewer than 20
  trades.

### Key Entities

- **Single-pair variant**: a model differing from the 11-pair model solely in the pairs it is
  trained and evaluated on; own storage location, own complete parameter set.
- **LINK-restricted trade set**: the LINK trades of either model on a window; the unit of
  comparison for US2 and US3.
- **Selection comparison**: shared / variant-only / control-only counts plus the differing trades'
  outcomes.
- **Power report**: pooled trade count, achieved noise floor, delta and interval.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The variant's effective entry gate and exit stack are identical to the 11-pair
  model's, and the ungated-gate warning is absent from every run used as evidence.
- **SC-002**: For every window evaluated, the LINK trade-set comparison is reported as three counts
  plus the differing trades' outcomes.
- **SC-003**: Per-trade expectancy on LINK is reported for both models with intervals, on every
  window with at least one LINK trade.
- **SC-004**: Every delta is reported with its interval and the applicable noise floor; no delta
  smaller than that floor is described as a win or a loss.
- **SC-005**: The pair-dominance rotation is stated explicitly in the conclusion.
- **SC-006** *(conditional — only if US4 runs; see the scope note in Requirements)*: The pooled
  evaluation reports its trade count and achieved floor, and states the minimum effect it could
  have resolved.
- **SC-007**: The conclusion answers, in plain terms, whether cross-pair training helps LINK
  selection, harms it, or cannot be distinguished — and if "cannot be distinguished", states what
  sample size would have been required.
- **SC-008**: No entry-gate, exit, label or architecture parameter differs between variant and
  control.

## Assumptions

- LINK's trade rate in the variant is assumed comparable to its rate within the 11-pair model. A
  materially different rate is itself a finding under US2.
- The 11-pair model's existing evaluations serve as the control; it is not retrained here.
- LINK's history begins 2022-01-14, which bounds how far back folds can reach.
- Pooled evaluation is expected to remain underpowered; the feature is designed so a null there is
  still informative through US2 and US3.

## Non-Goals

- Tuning any parameter. The ATR band, stop-loss grace and rvol thresholds are closed as being at
  their cross-window optima and inside the noise band.
- Changing the label, the entry gate, the exit stack or the model architecture.
- Expanding the pair universe, or building variants for additional single pairs. If LINK answers
  the question, ZEC and XRP do not need building; if it does not, the rotation table already
  suggests they will not either.
- Any GAN work.
- **Declaring a winner on a single-window P&L delta.** At 8 W1 trades this is the specific error
  the spec is shaped to prevent.
