# Feature Specification: Tail-Targeted Breakout Labelling

**Feature Directory**: `specs/003-breakout-tail-labelling`
**Created**: 2026-09-04
**Status**: Draft
**Host**: `NNNC_BreakoutOpen`

## Overview

The breakout arm's training label is mis-specified relative to the risk the
strategy takes to hold a position. It rewards the *average* forward move in a
distribution where the average is nearly meaningless, and the resulting trades
cannot pay for their own stop.

Measured 2026-09-04 (`regime/breakout_tail_analysis_README.md`):

- The forward-gain distribution from breakout-condition bars is severely
  fat-tailed — skew 3.4–7.9, excess kurtosis 35–225 against a Gaussian's 0/0,
  p99/median 8–12×. **The top 10% of bars carry ~40% of all upside; the top 25%
  carry ~65%.**
- At the current setting the median labelled gain is 2.83% while the operative
  stop is −4% to −6%. **Only 26% of labelled bars ever clear the stop**, giving a
  reward-to-risk of 0.57.

The arm currently misses its own break-even win rate on the out-of-sample window
by roughly one point (58.8% achieved against 57.7% required). A label whose
rewarded outcome is usually smaller than the risk taken is a credible cause, and
it is the only structural explanation not yet tested — seven prior interventions
all operated downstream of the label.

This feature tests whether re-specifying the label to target the tail closes that
gap, and what else must move with it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Re-specify the label to target the tail (Priority: P1)

As the strategy owner, I want the breakout label to reward moves large enough to
pay for the stop, so that the model is trained on opportunities the strategy can
actually convert.

**Why this priority**: It is the headline hypothesis and every other story either
supports it or depends on it. If the label cannot be made economic while staying
learnable, the feature ends here.

**Acceptance Scenarios**:

1. **Given** a candidate (horizon, gain-threshold) pair, **When** its label
   population is measured, **Then** the share of labelled bars clearing the
   operative stop and the resulting reward-to-risk are reported alongside label
   density.
2. **Given** a candidate that raises reward-to-risk above 1.0, **When** its
   density is checked, **Then** it is rejected if density falls below the level
   at which the classifier collapses to all-Hold.
3. **Given** a trained arm on the new label, **When** it is compared to the
   control, **Then** acceptance is decided on realised out-of-sample return, not
   on any classifier metric.

---

### User Story 2 - Make the hold window consistent with the label (Priority: P1)

As the strategy owner, I want the exit machinery to permit the move the label
promises, so that the strategy is not trained on one horizon and harvested on
another.

**Why this priority**: Equal to P1 because the headline candidate is
*unusable without it*. The best-economics cell peaks at a 17-hour median while a
forced exit fires at 12 hours — the label's payoff would be systematically cut
short. Changing the label alone would produce a false negative.

**Acceptance Scenarios**:

1. **Given** a candidate label, **When** its median time-to-peak is measured,
   **Then** it is compared against the forced-exit window and any conflict is
   reported before the arm is trained.
2. **Given** a label whose payoff peaks beyond the forced-exit window, **When**
   the arm is run, **Then** the hold window is extended to match, and the change
   is scoped to this strategy rather than applied to all families.
3. **Given** an extended hold window, **When** the run completes, **Then** no
   experimental edit remains in shared framework code.

---

### User Story 3 - Establish whether the change reaches the trades (Priority: P1)

As the strategy owner, I want to know whether a label change altered trading at
all, so that a null result is not mistaken for a refutation.

**Why this priority**: Six model-side changes in the preceding study left
out-of-sample trading *byte-identical* because the entry gate selected the bars.
Without this check, "no improvement" and "no effect reached the trades" are
indistinguishable, and the feature would draw a false conclusion either way.

**Acceptance Scenarios**:

1. **Given** any trained arm, **When** it is compared to the control, **Then**
   the comparison reports whether the executed trades differ at all.
2. **Given** an arm whose trades are identical to the control, **Then** it is
   recorded as INCONCLUSIVE (the gate absorbed it), never as a refutation.

---

### User Story 4 - Choose cheaply before spending retrains (Priority: P2)

As the strategy owner, I want candidates screened by calculation before any model
is trained, so that expensive runs are spent only on configurations that can
plausibly work.

**Why this priority**: Valuable but not blocking — it reduces cost rather than
enabling the outcome. Three analysis scripts already produce the needed figures
in minutes against 8–45 minutes per retrain.

**Acceptance Scenarios**:

1. **Given** the candidate set, **When** screening runs, **Then** each candidate
   has density, reward-to-risk, stop-clearing share, time-to-peak and
   path-reachability recorded without training anything.
2. **Given** screening results, **When** a candidate fails the density floor or
   conflicts irreconcilably with the hold window, **Then** it is excluded before
   any retrain is spent.

---

### User Story 5 - Re-tune the risk side against the new label (Priority: P3)

As the strategy owner, I want the stop band and entry threshold re-examined once
the label changes, so that settings fitted to the old label are not assumed to
transfer.

**Why this priority**: Contingent — only worth doing if a new label is adopted.
Both were previously swept against the *old* label; the stop band showed an
inverted-U peaking exactly at the inherited value, and the entry threshold left
win rate frozen across a 3.5× range. Neither result is transferable to a label
targeting substantially larger moves.

**Acceptance Scenarios**:

1. **Given** an adopted label, **When** the risk side is re-swept, **Then**
   results are reported against that label's control, not the historical one.
2. **Given** a re-sweep, **When** any setting is proposed, **Then** it is
   accepted only on out-of-sample return.

---

### Edge Cases

- **Label collapses to all-Hold.** Raising the threshold reduces density; below
  roughly one bar in eighteen the classifier stops predicting the signal class.
  Two of the five measured candidates already fall below this line.
- **The change never reaches the trades.** The entry gate may select the same
  bars regardless of the model, as it has for six prior changes.
- **Path risk rises with the horizon.** A longer window gives more opportunity to
  hit the stop before the peak; the best-economics candidate more than doubles
  the share of labels that would stop out first.
- **The hold-window change is global.** The forced-exit rule lives in shared
  framework code used by every family and by a live dry run.
- **An arm silently runs default configuration.** A variant without its own
  parameter file inherits class defaults instead of the tuned configuration; this
  produced a spurious 12-point result in the preceding study.
- **In-sample improvement masquerading as success.** Two of the three evaluation
  windows fall inside the training period.
- **Fewer, larger trades reduce statistical power.** Raising the threshold cuts
  trade count, widening the confidence interval on any measured difference.

## Requirements *(mandatory)*

### Functional Requirements

**Screening (no training)**

- **FR-001**: Every candidate MUST be characterised before training on: label
  density as a share of bars, median labelled gain, share of labelled bars
  clearing the operative stop, reward-to-risk against that stop, median
  time-to-peak, and share of labels that would hit the stop before reaching their
  peak.
- **FR-002**: Candidates whose density falls below the all-Hold collapse floor
  MUST be excluded before any retrain is spent.
- **FR-003**: Screening MUST NOT load or train a model.

**Label candidates**

- **FR-004**: The candidate set MUST include at least one configuration that
  requires no exit-side change, so the label effect can be isolated from the hold
  window.
- **FR-005**: The candidate set MUST include the best-economics configuration
  identified by screening, together with its required hold-window change.
- **FR-006**: A path-aware variant that additionally constrains adverse
  excursion before the peak MUST be evaluated at tail settings, where path risk
  is materially higher than at current settings.
- **FR-007**: An alternative trigger built on a different indicator basis MUST be
  retained as a candidate, so the result is not specific to one label form.
- **FR-008**: Each candidate MUST cite the measurement motivating it.

**Hold window**

- **FR-009**: Where a candidate's time-to-peak exceeds the forced-exit window,
  the hold window MUST be extended for the host strategy only.
- **FR-010**: Any change to shared framework code MUST be reverted after
  measurement and MUST NOT be left in place.
- **FR-011**: A hold-window change MUST be measured both with and without the
  label change, so their contributions are separable.

**Evaluation protocol**

- **FR-012**: Arms MUST be trained on a window ending before the out-of-sample
  evaluation period begins.
- **FR-013**: Acceptance MUST be decided on realised return in the out-of-sample
  window alone. Returns from windows inside the training period MUST be reported
  and labelled as in-sample, and MUST NOT drive acceptance.
- **FR-014**: Acceptance MUST NOT be based on any classifier metric. Three
  distinct metrics have each failed to predict return in the preceding study.
- **FR-015**: Every arm MUST be checked for whether its executed trades differ
  from the control. Identical trades MUST be recorded as inconclusive rather than
  as evidence against the candidate.
- **FR-016**: Every arm MUST carry its own strategy parameter file so it runs the
  tuned configuration rather than class defaults, and this MUST be verified by
  comparing exit-configuration fingerprints between arm and control.
- **FR-017**: Trade counts MUST be reported; an arm with too few trades to
  distinguish from the control MUST be reported as underpowered.
- **FR-018**: Realistic execution constraints MUST remain active.

**Scope control**

- **FR-019**: Interventions already refuted on this arm MUST NOT be re-tested —
  the stop band at the old label, the profit-target and return caps, the exit
  signal, the exit-label form, and the entry threshold at the old label.
- **FR-020**: No change may be promoted beyond the host strategy without being
  measured on a second family.
- **FR-021**: No per-pair special-casing.
- **FR-022**: The feature MUST record a negative outcome as a result. "The label
  was not the problem" is a valid and useful conclusion.

### Key Entities

- **Candidate**: a labelling configuration — horizon, gain threshold, optional
  adverse-excursion constraint, indicator basis — together with the hold-window
  change it requires and the measurement that motivates it.
- **Screening record**: the pre-training characterisation of a candidate
  (density, economics, timing, reachability) used to admit or exclude it.
- **Arm**: a trained model plus its evaluation results across the three windows,
  its trade-difference status against the control, and its configuration
  fingerprint.
- **Control**: the current parked configuration, against which every arm is
  compared.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every candidate has a complete screening record before any model is
  trained for it.
- **SC-002**: At least one candidate achieves a labelled population whose typical
  rewarded move exceeds the risk taken to hold it, while remaining above the
  density floor.
- **SC-003**: Each trained arm is classified as improved, unchanged, worsened, or
  inconclusive-because-absorbed, with the trade-difference check recorded.
- **SC-004**: The out-of-sample return of the best arm is stated against the
  control, with in-sample windows reported separately and excluded from the
  verdict.
- **SC-005**: The independent contributions of the label change and the
  hold-window change are separately reported.
- **SC-006**: No shared framework file is left modified at the end of the
  feature.
- **SC-007**: Every arm's configuration fingerprint is verified to match the
  control except for the parameter under test.
- **SC-008**: The feature concludes with an explicit verdict on whether label
  mis-specification was a material cause of the arm's shortfall, including the
  case where it was not.

## Assumptions

- The operative stop for evaluating label economics is the adaptive band the
  strategy actually applies, not the static value in configuration — the latter
  was confirmed inert for this family.
- The density floor below which the classifier collapses is taken from the
  existing risk-flagging convention rather than re-derived.
- Two of the three evaluation windows fall inside the training period. This is
  accepted so the deployment-regime window stays genuinely out-of-sample; those
  two are reported as in-sample and excluded from acceptance.
- The host arm's entry gate is loose enough for model changes to reach the
  trades. This is evidenced by its trades differing across all three windows on a
  prior retrain, unlike the tightly-gated family where they were identical.
- Screening figures already gathered on three pairs are representative enough to
  admit or exclude candidates; full evaluation uses the full traded set.
- Prior refutations on this arm remain valid, since they were measured on the
  same host under the same protocol.

## Dependencies

- The three existing screening scripts, which produce the distribution, economics
  and path-reachability figures without training.
- The arm-running harness, which pins evaluation windows, records configuration
  fingerprints and emits trade-difference information.
- The host strategy and its parked configuration as the control.

## Out of Scope

- The tightly-gated mean-reversion family, whose entry gate absorbs model-side
  change — a threshold variant there produced identical out-of-sample trades.
- Re-testing interventions already refuted on this arm (FR-019).
- New data sources or additional indicators beyond those already in the feature
  set.
- Per-pair tuning.
- Promotion of any change to shared framework defaults.
