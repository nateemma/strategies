# Feature Specification: Label-free Direct Policy Optimisation

**Feature Branch**: `009-labelfree-direct-policy`

**Created**: 2026-09-10

**Status**: Draft

**Input**: User description: "Feature: labelfree-direct-policy. This is research/label-free-study brief 01; the feature must be specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented. Read AGENT_GUIDE.md, then research/label-free-study/README.md, 00-evaluation-protocol.md and briefs/01-direct-policy-optimisation.md. Implement research/labelfree/sim.py per protocol §1 in MLX, and its calibration test against the frozen NNNC_MLX signal on W1 (regime/regimedata.py). Do not write any policy code until the calibration test passes and its report is in briefs/results/01-calibration-<date>.md. Then implement the policy/loss variants in the brief, logging every configuration to research/labelfree/trials.jsonl before running it. Run everything sequentially (AGENT_GUIDE operational rules). Stage only the files you create. Success criteria: the acceptance and kill gates of research/label-free-study/00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md."

## Overview

The repository's ML strategies all learn a hand-designed forward-looking label, then trade it
through thresholds, guards and exits designed separately. Prior measurement showed the label
the model learns *best* is the one that cannot pay (MCC 0.78 → 0.28 as EV/signal rises 3.6% →
9.0%), and that the model captures under 4% of the label's own ceiling. This feature tests
whether removing the label — training a small policy directly on net-of-cost realised return —
converts more of the available signal into P&L.

The feature has a hard internal ordering. A shared trading simulator must first be proven to
reproduce the production backtester's P&L on a frozen signal. Until that calibration passes,
no policy result is admissible evidence, and no policy code may be written.

## Clarifications

### Session 2026-09-10

- Q: For the blocking calibration gate, must the simulator reproduce the production strategy's four exit paths, or replay the entry/exit times a real backtest already produced? → A: Replay the realised entry/exit timestamps as a fixed series; the simulator computes only P&L accounting (fees, fill timing, liquidity cap, portfolio layer). It is NOT required to reproduce the production exit stack.
- Q: Should the training objective include the deployment exit stack (take-profit, adaptive stop, grace) that the wrapper keeps switched on, or train on band-crossing exits only? → A: Band-crossing exits only; the resulting simulator-vs-arbiter P&L gap for the trained policy MUST be measured and reported before acceptance.
- Q: Should the trend-rule baseline be pinned to the protocol's example parameters or searched for its strongest form? → A: Pinned to 20/60-bar EMA sign with band ±0.1, frozen and recorded before any policy configuration runs, preserving its "zero parameters learned from P&L" property.
- Q: The market data has no bid/ask, so how is the half-spread inside the objective obtained? → A: Estimate it per bar from high/low using a published OHLCV spread estimator (Corwin-Schultz or Abdi-Ranaldo) with a conservative floor, so cost scales with the illiquidity that causes it.
- Q: SC-002's absurdity bar compared a policy's return on its own fold range against the control's W1 figure, which is not commensurable (measured: a flat 0.28-exposure policy shows +289% from beta on f08's 4-year training range, while a good policy shows -16.9% on its 90-day test range). How should the lookahead sanity ceiling be expressed? → A: As a PAIRED comparison on identical data — leaky must exceed clean by >=2.0 log units of cumulative net return on the same range and seed, and the clean run must not exceed buy-and-hold over that range by more than the resolvable floor. Immune to horizon and regime, and it can genuinely fail.
- Q: For the single-window runs in W1/W2/W3, should training keep the 730-day minimum even though that leaves the distant window trainable on only ~18 pairs? → A: Yes, keep the 730-day minimum; report the pair-universe size per window and run the control on the identical universe within each window, since the comparison is always within a window and never across windows.

## Terminology

One concept, one name. These are used interchangeably in the surrounding study documents and are
pinned here so the plan, tasks and reports do not drift apart.

| Canonical term | Means | Also written as (avoid) |
|---|---|---|
| **arbiter** | The production backtester. It is the final authority on P&L; the simulator exists only for training speed | "production backtester", "freqtrade backtest" |
| **control** | The current production classifier at its production configuration, whose pooled walk-forward record already exists | "production control", "NNNC_MLX" |
| **screening fold** | The single fold used to narrow the variant grid before the pooled run | "richest fold", "f07" |

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trustworthy simulator (Priority: P1)

A researcher needs a fast, differentiable trading simulator whose profit-and-loss agrees with
the production backtester. They replay the exact trades a real backtest already produced for a
frozen, already-understood signal, and confirm the simulator reproduces their P&L to within a
stated tolerance. This isolates the accounting layer — fees, fill timing, liquidity cap, stake
and portfolio sizing — which is what every downstream brief depends on. Only then is the
simulator usable as a training environment.

**Why this priority**: It blocks everything. A simulator that disagrees with the arbiter
silently converts every downstream number into fiction, and the protocol names this as a kill
condition rather than a bug. It also has standalone value: briefs 2, 4, 5 and 6 all consume it.

**Independent Test**: Run the frozen production classifier's signal through both the simulator
and the production backtester over the deployment window; compare total return, trade count and
per-trade profit. Delivers a reusable, trusted simulator plus a written calibration report,
independent of any policy work.

**Acceptance Scenarios**:

1. **Given** the entry and exit timestamps the production backtester produced for the frozen
   classifier signal over the deployment window, **When** that fixed trade series is replayed
   through the simulator, **Then** per-trade profit agrees within 1 basis point, total return
   within 0.10 percentage points, and the trade counts match exactly.
2. **Given** a calibration run that agrees, **When** the run completes, **Then** a calibration
   report is written naming the window, signal, both P&L figures, the trade-count match and the
   measured discrepancy.
3. **Given** a calibration run that disagrees by more than the tolerance, **When** the
   discrepancy is observed, **Then** the simulator is declared wrong, the discrepancy is
   recorded, and no policy code is written until it agrees.
4. **Given** the simulator, **When** it decides at a bar, **Then** it uses only bars at or
   before that bar and fills at the following bar, with no other forward-looking quantity.

---

### User Story 2 - Leakage and gradient sanity ceiling (Priority: P2)

Before trusting any training result, the researcher deliberately breaks the simulator's timing
— shifting features forward by one bar so the policy can see the future — and confirms the
trained policy earns an implausibly large return. A policy that cannot exploit deliberate
lookahead has a broken learning path, and any subsequent "no edge" result would be meaningless.

**Why this priority**: It is the cheapest possible check that the training path actually
learns, and it separates "the objective found nothing" from "the gradient never flowed". It
runs immediately after the simulator and costs one short training run.

**Independent Test**: Train one policy configuration with lookahead deliberately enabled and
one without, on the same range and seed; confirm the leaky run exceeds the clean run by at least
2.0 log units and that the clean run does not beat buy-and-hold over that range by more than the
resolvable floor.

**Acceptance Scenarios**:

1. **Given** deliberately shifted features, **When** a policy is trained on them, **Then** its
   cumulative net log return exceeds that of the clean policy on the same range and seed by at
   least 2.0 log units, demonstrating the learning path works.
2. **Given** the same configuration with correct timing, **When** it is trained, **Then** its
   cumulative net return does not exceed buy-and-hold over the same range by more than the
   resolvable floor, demonstrating the leak was the shift and not the simulator.
3. **Given** a leaky run that fails to clear the bar, **When** that is observed, **Then** the
   learning path is treated as defective and no variant sweep is run until it is fixed.

---

### User Story 3 - Policy and loss variant sweep with full trial accounting (Priority: P3)

The researcher trains a small long-only exposure policy on net-of-cost realised return across
the brief's variant grid — two objectives, two turnover controls, three cost scales — with at
least five seeds each. Every configuration is written to a permanent trial log *before* it is
run, so the eventual multiple-testing correction counts every attempt, not just the ones worth
reporting.

**Why this priority**: This is the hypothesis under test, but it is worthless without stories 1
and 2 establishing that the environment is faithful and the learner works.

**Independent Test**: Run the grid on the screening fold; confirm the trial log contains
one entry per configuration written before its result, and that reported figures are
interquartile means across seeds rather than best-seed picks.

**Acceptance Scenarios**:

1. **Given** a configuration about to be evaluated, **When** the run starts, **Then** its full
   description already exists in the trial log with no result attached.
2. **Given** a completed configuration, **When** its performance is reported, **Then** the
   figure is the interquartile mean across at least five seeds with a bootstrap interval, and
   no single-seed figure is presented as the result.
3. **Given** two seeds that produce identical trades, **When** results are aggregated, **Then**
   they are counted as one measurement.
4. **Given** the grid on the single fold, **When** the best two configurations are identified,
   **Then** only those proceed to the full pooled multi-fold run.
5. **Given** any configuration, **When** it is trained, **Then** costs, spread and the fill cap
   were inside the objective it was trained against.

---

### User Story 4 - Verdict against every mandatory baseline (Priority: P4)

The researcher evaluates the surviving configuration through the pooled walk-forward harness
against all four mandatory baselines, stresses it at elevated costs and a tighter fill cap,
computes the overfitting statistics over the whole trial log, and reaches one of three verdicts:
accept, not resolved, or kill.

**Why this priority**: It converts a training result into an admissible finding. It cannot run
before story 3 produces a candidate.

**Independent Test**: Run the candidate and all four baselines through the pooled harness;
confirm the report states the return delta with its interval against the resolvable floor, and
that a delta below the floor is reported as "not resolved" rather than "no effect".

**Acceptance Scenarios**:

1. **Given** a candidate configuration, **When** it is evaluated, **Then** it is compared
   against the production control, buy-and-hold of the whitelist, a zero-learned-parameter
   trend rule with a no-trade band, and the production control at its cost-aware operating
   point — all four, net of full costs.
2. **Given** a return delta whose interval includes zero or which falls below the resolvable
   floor, **When** the verdict is written, **Then** it reads "not resolved", never "no effect".
3. **Given** a candidate that clears the floor, **When** it is stressed, **Then** it is
   re-evaluated at elevated fee multiples and a tighter fill cap, and the fee level at which
   the edge reaches zero is reported.
4. **Given** the completed trial log, **When** the overfitting statistics are computed, **Then**
   a backtest-overfitting probability above the threshold fails the feature regardless of the
   point estimate.
5. **Given** any reported return figure, **When** it is presented, **Then** trades per year and
   mean holding time appear beside it.
6. **Given** a candidate policy trained on band-crossing exits, **When** it is re-run through the
   arbiter with the deployment exit stack active, **Then** both P&L figures, their difference, and
   the share of trades exited by the deployment stack rather than the band are reported before any
   acceptance decision is made.

---

### User Story 5 - Reproduction, report and audit (Priority: P5)

A second, independent process reconstructs the accepted result from the saved configuration and
seed and obtains identical trades. The researcher then writes the study's standard result
report and audits the trial log for completeness.

**Why this priority**: The protocol makes a result inadmissible without it, and it is the step
most easily skipped once the interesting numbers are known.

**Independent Test**: From a clean process, load the saved configuration and seed, re-run, and
compare the trade list byte-for-byte against the original.

**Acceptance Scenarios**:

1. **Given** an accepted configuration, **When** a fresh process re-runs it from the saved
   config and seed, **Then** the resulting trades are identical to the original.
2. **Given** a completed run, **When** the result report is written, **Then** it states the
   hypothesis, config path and seed list, trial count, pooled return with interval, folds
   positive, trades per year, holding time, the baseline table, cost stress, the overfitting
   statistics, the verdict, and what changes for the next brief.
3. **Given** a verdict of kill or not-resolved, **When** the work stops, **Then** the report is
   still written and the negative result recorded with the same weight as a positive one.
4. **Given** the finished feature, **When** the trial log is audited, **Then** every
   configuration that touched any fold appears in it.

---

### Edge Cases

- **The calibration never converges.** The simulator and backtester disagree by more than the
  tolerance after the budgeted attempts. The feature stops at story 1 with a recorded
  discrepancy; no policy result exists to report.
- **The leaky sanity run does not separate from the clean run.** Treated as a broken learning path,
  not as evidence about the hypothesis, matching the constitution's rule that an inert
  parameter is a defect rather than a null result.
- **The policy learns to stay flat permanently.** A zero-trade arm is inconclusive, never a
  zero-return result; it must be reported as such with its trade count.
- **The policy's edge appears only at turnover far above the control's.** Flagged, and required
  to survive the doubled-fee stress before it can be accepted.
- **In-sample advantage exceeds deployment-window advantage by more than the floor.** Recorded
  as overfitting; the configuration is stopped rather than tuned further.
- **Two configurations tie within the noise floor.** Neither is declared better; the floor
  governs, and a ranking inside it is a ranking of noise.
- **The fill cap binds on most candidate trades.** The unfilled remainder carries forward; if
  the edge exists only where the cap binds, it is phantom and must be reported as such.
- **A fold produces too few trades to bootstrap.** Reported as inconclusive for that fold rather
  than folded silently into the pooled figure.
- **The deployment exit stack dominates the policy's exits.** If most trades are closed by the
  take-profit, adaptive stop or grace rather than by the policy's band, the objective the policy
  was trained on is not the one being scored; this must be reported as a threat to the result, not
  absorbed into it.

## Requirements *(mandatory)*

### Functional Requirements

**Simulator**

- **FR-001**: The simulator MUST accept per-pair feature tensors produced by the existing
  feature pipeline and scalers, so that inputs are identical to those the production classifier
  sees.
- **FR-002**: The simulator MUST use only pair-agnostic features, excluding raw price, raw
  volume, and any quantity directly proportional to price.
- **FR-003**: The policy MUST see only bars at or before the decision bar and emit a target
  long-only exposure between zero and one per pair; fills MUST occur at the following bar's
  open or a volume-weighted proxy. During calibration the simulator MUST use the arbiter's exact
  fill convention — the following bar's open, not a volume-weighted proxy — so that a per-trade
  comparison is meaningful. Any volume-weighted proxy used in training MUST be reported as a
  deviation from the calibrated configuration.
- **FR-004**: The training objective MUST include taker fee per side at the real exchange tier,
  half-spread, and a volatility-scaled slippage term. The training cost scale MUST be a swept
  hyperparameter; evaluation MUST always use full or pessimistic cost.
- **FR-004a**: Because the available market data is open/high/low/close/volume only and contains no
  bid or ask, the half-spread MUST be estimated per bar from high and low using a published
  OHLCV-based spread estimator, with a conservative floor so the estimate cannot collapse to zero
  on quiet bars. A single flat spread constant MUST NOT be used, because it would under-cost
  illiquid pairs by the widest margin exactly where this repository has previously found phantom
  edges. The estimator and floor MUST be recorded with every trial.
- **FR-005**: Executed exposure change MUST be capped at what a stated maximum fraction of the
  candle's quote volume can fill, with the unfilled remainder carried to the next bar. Training
  MUST use a smooth approximation so gradients survive where the cap binds; evaluation MUST use
  the hard cap.
- **FR-006**: The simulator MUST replicate the production maximum concurrent positions, stake
  sizing and tradable balance ratio, so its P&L is comparable to the backtester's.
- **FR-007**: The simulator MUST emit per-bar net return, turnover, position, and a trade list
  in the same shape as the existing backtest archive, so existing pooling and comparison tooling
  works unchanged.
- **FR-008**: The simulator MUST be validated by replaying, as a fixed series, the entry and exit
  timestamps the production backtester actually produced for the frozen classifier signal over the
  deployment window, and matching that backtester on all three of: per-trade profit within 1 basis
  point, total return within 0.10 percentage points, and exactly equal trade counts. The per-trade
  comparison MUST be reported alongside the total, so that offsetting errors cannot pass on the
  total alone.
- **FR-008a**: The simulator is NOT required to reproduce the production strategy's exit stack —
  its take-profit, adaptive stop, stop grace and model sell signal. Calibration therefore validates
  the P&L accounting layer only: fee application, fill timing, the liquidity cap, stake sizing and
  the portfolio layer. Exits driven by the policy's own band are covered by FR-013 and the
  arbiter re-run in FR-027, not by this gate.
- **FR-009**: No policy code MAY be written until FR-008 passes and its calibration report
  exists.
- **FR-010**: The system MUST provide a deliberately-leaky mode that shifts features forward by
  one bar, used solely to prove the learning path can exploit lookahead. On the same range and
  seed, a policy trained in this mode MUST exceed the clean policy by at least 2.0 log units of
  cumulative net return; failing that, the learning path is defective and no variant sweep may run.
  Returns MUST be reported in log space, because a leaky run's linear return can overflow.

**Policy and training**

- **FR-011**: The policy MUST consume the same normalised feature sequence and window length the
  production classifier uses, pooled across the whitelist.
- **FR-012**: The backbone MUST begin with the smallest architecture that can work — a
  single-layer recurrent block or the existing small state-space block. A transformer backbone
  MUST NOT be the starting point.
- **FR-013**: The system MUST implement both turnover controls: a learned no-trade band emitted
  by the head, and a plain exposure head with an explicit turnover penalty. Both MUST be run.
- **FR-014**: The system MUST implement both objectives: a mean-variance utility over pooled
  batches, and a robust objective combining pooled and worst-window risk-adjusted return. Both
  MUST be run and logged as separate trials.
- **FR-014a**: The training objective MUST use the policy's own band-crossing exits only. It MUST
  NOT model the deployment exit stack (take-profit, adaptive stop, stop grace), which remains
  active in the deployed wrapper per FR-028.
- **FR-014b**: Because FR-014a makes the trained objective differ from what the arbiter scores,
  the gap MUST be quantified: for the candidate policy, the simulator's P&L and the arbiter's P&L
  over the same range MUST both be reported, with their difference and the share of trades whose
  exit was decided by the deployment stack rather than the band. This measurement is a
  precondition of acceptance, not a diagnostic.
- **FR-015**: Training MUST use anchored folds identical to the existing fold plan, with one
  policy per fold trained only on that fold's training range.
- **FR-015a**: Single-window runs MUST use the same 730-day minimum training length as the pooled
  folds. Because available history begins in 2021 and only part of the universe reaches that far
  back, the tradable pair universe will be smaller in the distant window; its size MUST be
  reported for each window, and the production control MUST be run on that identical universe
  within the same window. Results MUST NOT be compared across windows of differing universe.
- **FR-016**: Each configuration MUST be trained with at least five seeds, reported as an
  interquartile mean with a stratified bootstrap interval. A best-seed figure MUST NOT be
  reported as the result. Seeds producing identical trades count as one measurement.
- **FR-017**: Early stopping MUST use the training objective only; no validation split may be
  drawn from a test fold.
- **FR-018**: When any forward-looking quantity is used in training, the maximum horizon MUST be
  purged and one horizon embargoed after each fold boundary.

**Evaluation and accounting**

- **FR-019**: Every configuration evaluated on any fold MUST be appended to the trial log
  *before* its result is known, recording seed, objective variant, cost scale, architecture,
  band width and threshold.
- **FR-020**: Primary evidence MUST be the pooled walk-forward result across the full fold set
  with trades pooled before any statistic. Single-window figures are smoke tests only.
- **FR-021**: All four mandatory baselines MUST appear in every report: the production control,
  equal-weight buy-and-hold of the whitelist, a trend rule with a no-trade band and no
  P&L-learned parameters, and the production control at its cost-aware operating point.
- **FR-021a**: The trend-rule baseline MUST be pinned to a 20/60-bar exponential-moving-average
  sign with a ±0.1 no-trade band, frozen and recorded before any policy configuration is run. Its
  parameters MUST NOT be searched or tuned, preserving its defining property of having no
  parameter learned from P&L; consequently it contributes no entries to the trial log.
- **FR-022**: Every reported return figure MUST be accompanied by trades per year and mean
  holding time.
- **FR-023**: The accepted configuration MUST be re-evaluated at elevated fee multiples and a
  tighter fill cap, reporting the fee level at which the edge crosses zero.
- **FR-024**: The report MUST compute the deflated risk-adjusted return and the
  backtest-overfitting probability over the full trial log.
- **FR-025**: A return delta below the resolvable floor MUST be reported as "not resolved",
  never as "no effect".
- **FR-026**: An accepted result MUST be reproduced from its saved configuration and seed by a
  separate process producing identical trades.

**Delivery and workflow**

- **FR-027**: A thin production-strategy wrapper MUST expose the trained policy so the candidate
  can be re-run through the real backtester as the arbiter, entering when exposure crosses the
  band's upper edge and exiting when it crosses the lower edge.
- **FR-028**: The wrapper MUST leave the existing stop and grace logic unchanged, so the
  comparison isolates the entry and exit signal.
- **FR-029**: The wrapper MUST pin its model path explicitly, to avoid the recorded silent
  model-resolution failure.
- **FR-030**: All runs MUST execute sequentially; concurrent runs are prohibited.
- **FR-031**: Only files created or modified by this feature MAY be staged; blanket staging is
  prohibited.
- **FR-032**: A result is admissible only if the feature directory holds a specification, plan,
  task list, and a clean cross-artefact analysis produced both before implementation and again
  on completion.
- **FR-033**: The calibration report and the final result report MUST be written to the study's
  results directory, and the final report MUST link the feature directory and both analysis
  reports.

### Key Entities

- **Trial record**: One evaluated configuration. Holds seed, objective variant, turnover
  control, cost scale, architecture, band width, threshold, fold identity and timestamp. Written
  before the result exists; the result is attached afterwards. The complete set drives the
  multiple-testing correction.
- **Policy configuration**: The reproducible description of a trained policy — architecture,
  objective, hyperparameters, feature list, window length, fold and seed. Must be sufficient for
  a fresh process to reproduce identical trades.
- **Simulated run**: Per-bar net return, turnover and position series plus a trade list shaped
  like the existing backtest archive, so downstream pooling works unchanged.
- **Fold**: One anchored train/test split from the existing fold plan, with its training range,
  test range, purge and embargo.
- **Calibration report**: The evidence that the simulator matches the arbiter — window, frozen
  signal, both P&L figures, trade counts, measured discrepancy and verdict.
- **Result report**: The study's standard per-brief write-up, including the verdict and what it
  changes for the next brief.

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Blocking gate — simulator calibration**

- **SC-001**: Replaying the production backtester's own realised entry/exit timestamps for the
  frozen classifier signal over the deployment window through the simulator reproduces that
  backtester on all three of: every matched trade's profit within 1 basis point, total return
  within 0.10 percentage points, and identical trade counts. Until all three hold, no policy
  result is admissible and no policy code exists.
- **SC-002**: On the **same evaluation range and seed**, a deliberately-leaky training run (features
  shifted forward one bar) exceeds the clean run by **at least 2.0 log units** of cumulative net
  return — roughly 7× linear — **and** the clean run's cumulative net return does not exceed its
  **exposure-matched** benchmark (buy-and-hold scaled to the clean policy's own mean exposure) by
  more than the resolvable floor. The benchmark must be exposure-matched: a long-only policy that
  is only partly invested beats a fully-invested benchmark in any falling market purely by holding
  less, which is beta rather than leakage. The comparison is paired on
  identical data, so it is unaffected by the horizon or market regime of the range chosen; a fixed
  multiple of any single-window figure is not, and would be untestable across ranges whose
  buy-and-hold outcomes differ by orders of magnitude.

**Acceptance gates (protocol §5 — all six required)**

- **SC-003**: Pooled walk-forward annualised return is at least 3.35 percentage points per year
  above the production control, with a bootstrap 95% interval excluding zero.
- **SC-004**: The configuration is positive in at least 10 of the 15 folds, and positive in each
  of the three evaluation windows when run single-window, with the production control run on the
  identical pair universe within each window and that universe's size reported alongside each
  window's result.
- **SC-005**: The configuration beats all four mandatory baselines net of full costs.
- **SC-006**: Backtest-overfitting probability is at most 0.2 and the deflated risk-adjusted
  return is above zero, both computed over the complete trial log.
- **SC-007**: The return delta remains positive at 1.5× fees.
- **SC-008**: A separate process reproduces the accepted result from its saved configuration and
  seed, producing identical trades.

**Kill gates (protocol §5 — any one stops the feature)**

- **SC-009**: If no configuration clears SC-003 within the budgeted work, the verdict is
  recorded as "not resolved" — never "no effect" — and the feature stops.
- **SC-010**: If any configuration's in-sample advantage exceeds its deployment-window advantage
  by more than 3.35 percentage points per year, it is recorded as overfitting and stopped.
- **SC-011**: If simulator and backtester P&L disagree by more than the SC-001 tolerance on the
  calibration signal, the simulator is declared wrong and nothing downstream counts as evidence
  until it is fixed.

**Brief-specific kill gates**

- **SC-012**: If calibration has not passed within 5 fix-and-retest attempts, the feature stops
  and the simulator is fixed before any modelling. Each attempt is one calibration run against
  the arbiter; there is no wall-clock bound.
- **SC-013**: If the best pooled configuration is below +3.35 percentage points per year against
  the production control with an interval including zero, the verdict is "not resolved", the
  shared simulator is handed to the next brief, and this family stops after that brief.
- **SC-014**: If turnover at the accepted operating point exceeds 3× the control's, the result
  is flagged and MUST additionally survive 2× fees with the delta still positive.

**Process gates**

- **SC-015**: Every configuration evaluated on any fold appears in the trial log, each written
  before its result was known; the count used for the overfitting correction equals the number
  of log entries.
- **SC-016**: Every reported return figure is an interquartile mean across at least five seeds
  with a bootstrap interval, and appears beside its trades per year and mean holding time.
- **SC-017**: The feature directory holds a specification, plan, task list and a clean
  cross-artefact analysis produced before implementation and again on completion; the result
  report links the feature directory and both analyses.
- **SC-018**: For the candidate policy, the simulator's P&L and the arbiter's P&L over the same
  range are both reported, together with their difference and the share of trades whose exit was
  decided by the deployment exit stack rather than the policy's band. No configuration is accepted
  until this reconciliation is on record.

## Non-Goals

The following adjacent approaches are explicitly out of scope. Each was excluded on recorded
evidence; none may be introduced into this feature without new evidence.

- **Return-conditioned generative planners** — diffusion planners and decision transformers
  provably copy lucky trajectories in stochastic environments.
- **Training a policy inside a learned market simulator** — agents exploit simulator artefacts;
  there is no alpha-transfer record. This feature's simulator replays *real* bars; it does not
  generate them.
- **Recursive self-training on generated data** — collapses distribution tails first, and the
  tails carry the risk.
- **Fine-tuning time-series foundation models** — zero-shot out-of-sample explanatory power is
  negative; a features-only screen belongs to a different brief.
- **Online or continual fine-tuning during evaluation** — leak-free re-evaluations shrink the
  reported gains substantially; scheduled retraining on a rolling window is the safe default and
  is what the fold plan already does.
- **Reinforcement learning** — deferred to a later brief, which must beat the differentiable
  baseline this feature establishes before it is justified.
- **Gradient-free search over the policy** — reserved for the non-differentiable pieces in a
  later brief, under strict trial accounting.
- **Full decision-focused optimisation layers** — long-only with caps is a closed-form
  projection; such layers polarise forecasts and churn.
- **Additional hand-crafted labels** — the measured problem is the objective, not label
  scarcity; labels re-enter only as a warm-start prior in the next brief.
- **Per-pair code or per-pair parameter overrides** — prohibited by the constitution; fixes must
  be structural and uniform across the universe.

## Assumptions

- **Windows and folds come from the protocol, not from this feature.** The deployment window is
  the sole single-window acceptance window; the other two windows are reported only. The pooled
  walk-forward fold plan, its minimum training length, test fold length and pair exclusions are
  taken unchanged from the existing harness.
- **The resolvable floor is 3.35 percentage points per year**, the measured pooled walk-forward
  floor. Single-window differences below roughly 7.6 percentage points are noise and are not
  treated as evidence.
- **The production control is the current production classifier at its production
  configuration**, whose pooled walk-forward record already exists and is reused rather than
  recomputed.
- **The frozen calibration signal** is the production classifier's buy decision held until its
  own exit rule fires — a fixed, already-understood signal, not a retrained one.
- **The variant grid is twelve configurations** — two objectives × two turnover controls × three
  cost scales — each at five or more seeds, screened on **fold f08** before the best two proceed to
  the full pooled run. f08 is the fold with the most control trades (71 of the 378 pooled; f07 has
  58), which is the property that matters for resolving differences between configurations. It is
  deliberately **not** selected by return — f08 returns −0.14% while f07 returns +8.08%, and
  screening on a favourable fold would bias selection toward configurations that only work in
  benign conditions.
- **Evaluation cost scale is full or pessimistic**, regardless of the cost scale used in
  training.
- **The arbiter is the production backtester.** The simulator exists for training speed; every
  candidate is finally re-run as a real strategy through the existing fold runner.
- **Available history is uneven across pairs, and coverage is measured as days of actual data
  inside the training range** — never by start date, because one pair (XRP) begins early but carries
  a 911-day delisting gap that a start-date test would miss. Under the ">= 180 days in range" rule,
  the distant window trains on 9 of the 10 pairs (only XRP is excluded), six of them on partial
  history of 345-533 days rather than the full 729. This asymmetry is accepted because every
  comparison is made within a window against a control facing the identical universe.
- **Existing feature scalers are reused unchanged.** This feature does not alter the feature
  include list; if it ever did, scalers would have to be regenerated before any comparison.
- **Runs are sequential** on a single Apple Silicon machine; the models are small and expected
  to train in minutes per fold.
- **The study directories `research/labelfree/` and the briefs' results directory do not yet
  exist** and are created by this feature.
- **Only this feature's own files are staged**, committed directly to the strategies repository;
  the outer upstream checkout receives nothing.
