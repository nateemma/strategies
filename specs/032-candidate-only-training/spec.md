# Feature Specification: Candidate-Only Training for NNNC Models

**Feature Branch**: `032-candidate-only-training` (spec directory; work is committed to master)

**Created**: 2026-10-13

**Status**: Draft

**Input**: User description: "Candidate-only training for NNNC models (feature 032). [...] Goal: retrain so the model's capacity goes to ranking the rule's candidates instead of re-learning the rule. Mechanism: a default-off, training-only row filter in the shared training path [...] applied AFTER windowing [...] Arms: ARO and GBR first, 15 walk-forward folds each, 24 h grace, held-pair add off [...] Evaluation, staged and pre-registered: (1) within-candidate AUC [...] (2) simulator book vs the width book on the early half f00-f07; (3) late half f08-f14 confirmatory; (4) only then real walk-forward arms [...] A width-relative label is out of scope [...]" (full text in the conversation of 2026-10-13)

## Background *(diagnostic evidence; research/model_calibration/findings.md)*

**The model mostly learns the rule.** The gbb_path label can only be positive on bars that pass
the gbb rule (guard_metric < 0.3 AND bb_width > 0.035). About 20% of bars pass it, and about half
of those are positive. The NNNC models score 0.93–0.98 AUC overall, but only 0.53–0.62 on the
rule-eligible bars. Most of the overall AUC is the rule.

**A model-free book matches the model books.** The width book takes the rule's candidates whose
band width is at or above the previous fold's count-matched cut. In real 15-fold backtests at 24 h
grace (commit f470817), width minus model is:

| | width − model (pp) | paired verdict |
|---|---|---|
| ARO | +1.48 | UNCHANGED |
| GBR | +0.22 | UNCHANGED |
| GFM | +24.31 | UNCHANGED (p 0.08) |
| MLX | −1.09 | UNCHANGED |

**But the model does know something width does not (ARO and GBR).** Re-weighting the width book's
trades by whether the model agrees (simulator, commit 5d562b2, EXPLORATORY):

| | agree, pp per trade (early / late) | disagree, pp per trade (early / late) |
|---|---|---|
| ARO | 0.146 / 0.096 | 0.074 / 0.011 |
| GBR | 0.114 / 0.082 | 0.051 / 0.010 |

So the information is there, but the model spends most of its capacity re-deriving the rule.
Training only on the rule's candidates removes that task. The model's job becomes ranking the
candidates.

**The training path today.** Training windows are built per row; window i ends at bar i and
carries bar i's label. The training rows include every bar, labelled SELL / HOLD / BUY. On an
eligible bar the label is never SELL, because the SELL label needs guard_metric > 0.3. The
validation split, which drives early stopping, is built the same way. Class weights are computed
from the training labels.

## Clarifications

### Session 2026-10-13

- Q: How is each fold's entry threshold chosen for the candidate-only models? → A: At the same
  overall entry rate as the control (its existing target rate over all training-window bars), with
  bars that fail the predicate counted as scoring 0. This keeps the trade count close to the
  current model's, so the comparison isolates ranking quality, and uses the training window only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A default-off candidate-only training switch (Priority: P1)

A researcher declares, on a new model class, a training row filter: a predicate over a bar's
indicator values. With it declared, the model is trained and validated only on windows whose last
bar satisfies the predicate. Without it, training is unchanged.

**Why this priority**: Nothing can be trained or measured until the switch exists. It must leave
every existing model, weight file and recorded result untouched.

**Independent Test**: With no filter declared, every existing model key, strategy fingerprint and
reference trade record is unchanged, and a training-data build is identical row for row. With the
filter declared, a scripted training-data build keeps exactly the windows whose last bar is
eligible, each with its original, unbroken window and its own label.

**Acceptance Scenarios**:

1. **Given** no filter declared, **When** the training data is built for any existing NNNC class,
   **Then** the tensors, labels, weights and class weights are identical to today's, and every
   existing model key and strategy fingerprint is unchanged.
2. **Given** a filter declared, **When** the training data is built, **Then** the kept samples are
   exactly the windows whose last bar satisfies the predicate. Each kept window is the same
   contiguous window it was before filtering; no window joins bars that were not adjacent.
3. **Given** a filter declared, **When** the samples are kept, **Then** the labels and any
   per-sample weights are filtered with the same mask, so sample i keeps its own label and weight.
4. **Given** a filter declared, **When** class weights are computed, **Then** they come from the
   kept training labels only. A class with no kept samples (SELL, on the gbb rule) gets no weight
   and causes no error.
5. **Given** a filter declared, **When** early stopping is evaluated, **Then** the validation split
   is filtered with the same predicate.
6. **Given** a filter declared, **When** the model key is computed, **Then** it differs from the
   same class without the filter, so the candidate-only weights never overwrite or load as the
   existing weights.
7. **Given** a filter declared, **When** the strategy enters trades, **Then** an entry requires
   the bar to satisfy the same predicate, so the model is only ever asked to rank bars it was
   trained on.
8. **Given** a filter that keeps no rows in a fold, **When** training runs, **Then** the run stops
   with a clear error rather than training on nothing.

---

### User Story 2 - Measure whether the retrained model ranks candidates better (Priority: P2)

The researcher trains the candidate-only versions of ARO and GBR over the 15 walk-forward folds,
then measures ranking within the candidates and the simulated books, before any backtest.

**Why this priority**: Training is the expensive step (about 3 hours per model). The cheap
measurements decide whether a backtest is worth running.

**Independent Test**: A report with, per model and half: within-candidate AUC for the retrained
and the current model, and simulated book pp and trades for the retrained model, the current
model and the width book.

**Acceptance Scenarios**:

1. **Given** the retrained models, **When** their per-bar scores are collected for every test
   fold, **Then** the within-candidate AUC is computed on exactly the same eligible bars and labels
   as the current model's 0.53–0.62.
2. **Given** the scores, **When** the simulated books are built, **Then** the simulator's
   fidelity on the retrained model's own entries is checked first and reported.
3. **Given** the early half f00–f07, **When** the simulated retrained book is compared with the
   width book, **Then** the result is reported against the pre-registered stage rule.
4. **Given** the early-half rule passes, **When** the late half f08–f14 is simulated, **Then** the
   result is reported as the confirmatory simulator check.

---

### User Story 3 - Pre-registered walk-forward judgement (Priority: P3)

If the simulator stages pass, the researcher commits a pre-registration, runs the candidate-only
arms as real walk-forward backtests, and judges each against its model control and the width
book.

**Why this priority**: Only a real backtest can decide adoption. It comes last because it costs a
backtest chain per model and is only worth running if the cheap stages pass.

**Independent Test**: The committed pre-registration, the arm records (15 folds per model), and
the verdicts with paired and risk results.

**Acceptance Scenarios**:

1. **Given** a committed pre-registration, **When** the judge runs, **Then** it refuses if any arm
   record predates the pre-registration commit.
2. **Given** the arms, **When** they are judged, **Then** each gets a verdict against its model
   control (wf_aro_g24, fs_gbr_gr) and against its width book (w_aro, w_gbr), with early and late
   gains, paired verdict, risk verdict and trade counts.
3. **Given** a stage rule fails, **When** the report is written, **Then** the stage at which it
   failed and the evidence are recorded as a result. The negative result is recorded with the same
   weight as a positive one.

---

### Edge Cases

- A fold's training span has very few eligible windows: the kept count per fold is reported, and a
  fold with no kept rows stops the run (US1 scenario 8).
- The first rows of a training span have zero-padded windows: they are kept or dropped by the same
  rule as any other row (their last bar's eligibility), exactly as today's padding.
- The rule depends on the bar's indicator values. These must be the same values the strategy
  computes at entry, so training and inference select the same bars.
- On eligible bars the SELL class is empty. The three-way output head is kept, so the predictor
  and threshold code are unchanged. Its SELL output is simply never trained towards.
- The held-pair add is the NNNC default (since 2026-10-13). The arms and their comparisons all turn
  it off, so the comparison is with the add-off controls and width books.
- Model weights for the candidate-only classes are new files under new keys. A missing weight file
  for a fold stops the backtest (strict walk-forward), as for every model today.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The shared training path MUST accept an optional, strategy-declared training row
  filter. When none is declared, training data, model keys, fingerprints and every recorded result
  MUST be unchanged.
- **FR-002**: The filter MUST be applied after windows are built: a sample is kept if and only if
  the last bar of its window satisfies the predicate. Raw rows MUST NOT be removed before windowing.
- **FR-003**: Labels and per-sample weights MUST be filtered with the same mask as the windows.
  Class weights MUST be computed from the kept training labels only. A class with no samples MUST
  NOT cause an error.
- **FR-004**: The validation split used for early stopping MUST be filtered with the same
  predicate.
- **FR-005**: The predicate MUST be declared by the strategy (the NNNC side), not written into the
  shared training code, so the shared code stays strategy-agnostic.
- **FR-006**: Declaring the filter MUST change the model key of that class and only that class.
  Existing classes' model keys MUST be unchanged. The model-key audit MUST classify the new
  setting.
- **FR-007**: A class with the filter MUST require, at entry, that the bar satisfies the same
  predicate.
- **FR-008**: Each fold's entry threshold for a candidate-only model MUST be chosen on that fold's
  training window only, at the same overall entry rate as its control, with bars that fail the predicate counted as scoring 0 (Clarification Q1).
- **FR-009**: Candidate-only arm classes MUST be built for ARO and GBR on all 15 folds, at 24 h
  grace with the held-pair add off. Everything except the training filter, the entry predicate
  and the threshold MUST match the 24 h controls (wf_aro_g24, fs_gbr_gr).
- **FR-010**: A within-candidate measurement MUST report AUC on rule-eligible bars per model and
  half, for the retrained and the current model on identical bars and labels.
- **FR-011**: The simulator MUST be checked for fidelity on each retrained model's own entries
  before its simulated book is reported.
- **FR-012**: The stage rules MUST be committed before the stage they govern is looked at:
  - the early-half simulator rule before the early-half books are computed;
  - the late-half simulator rule and the walk-forward rule before the late-half books are computed.
- **FR-013**: The walk-forward arms MUST be judged with the existing judge against the model control
  and against the width book. The judge MUST refuse records that predate the pre-registration.
- **FR-014**: Training MUST run one fold at a time, one trainer at a time.
- **FR-015**: Promoting a candidate-only model to a default MUST NOT be part of this feature.

### Key Entities

- **Training row filter**: the strategy-declared predicate over a bar's indicator values, here
  the gbb rule (guard_metric < 0.3 AND bb_width > 0.035), and whether it is declared.
- **Candidate-only model**: a model class with the filter declared. It has its own model key and
  weights per fold.
- **Candidate-only arm**: the per-fold strategy classes for one model, with the filter, the entry
  predicate, 24 h grace, add off and the fold thresholds.
- **Stage rule**: the pre-registered pass condition for one evaluation stage.
- **Width book**: the existing model-free baseline (w_aro, w_gbr) the arms must beat.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With no filter declared, the reference arms (arm D wf_aro, the four 24 h controls and
  the four width books) reproduce their recorded per-fold trades 15/15, and all existing model keys
  and fingerprints are unchanged.
- **SC-002**: On a scripted training-data build, 100% of kept samples equal the corresponding
  unfiltered window and label, and 0 kept samples have an ineligible last bar.
- **SC-003**: Candidate-only weights exist for all 15 folds of ARO and GBR (30 trained folds).
- **SC-004**: The within-candidate AUC of each retrained model is reported per half next to the
  current model's, on the same bars.
- **SC-005**: The simulator reproduces each retrained model's backtested entries at least 95% (pair
  and candle) before its books are used.
- **SC-006**: Each stage's outcome (pass, fail, or not reached) is recorded against its committed
  rule. The run ends with a verdict per model, including the stage that stopped it.

## Assumptions

- The eligibility predicate is the gbb rule as the label uses it: guard_metric < 0.3 AND bb_width >
  0.035. The rvol condition of the label is not part of the predicate; rvol is already an entry gate.
- The label, features, architecture, seed, training spans and grace are unchanged from the
  controls. Only the kept training samples change.
- Scalers are untouched: the feature list does not change, so no scaler regeneration is needed.
  The scaler-leak rule (constitution XI) applies exactly as for the controls.
- The evaluation is the 2023–26 walk-forward span. Its early half selects and its late half
  confirms. There is no distant-era window for NNNC models (no 2022-trained folds). The verdict is
  labelled PROVISIONAL (one era), as for 031's NNNC books.
- GFM and MLX are out of scope here. MLX's book already equals the width book; GFM's model rarely
  agrees with width. Either can be added later with the same switch.
- Out of scope: a width-relative label, sizing by the model, model architecture changes, and
  promotion to a default.
