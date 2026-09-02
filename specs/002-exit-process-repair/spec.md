# Feature Specification: Exit-Process Characterisation and Repair

**Feature Branch**: `002-exit-process-repair` (spec directory; this project commits directly to master)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "Look at the exit process to see if it can be fixed. Characterise what is driving the exit statistics (presumably the trailing stop) and identify a set of approaches that might address it. Use cheap tests where they exist; at minimum it must not require a re-train."

## Context

Feature 001 validated the entry side and closed it. The exit side has never been characterised on its own terms. Measured on the current tuned configuration across three pinned windows (167 trades):

| exit reason | n | share | mean | median | total | median duration | win rate |
|---|---|---|---|---|---|---|---|
| **stop** | 83 | **50%** | −0.87% | −1.20% | **−72.45%** | **1.00h** | **37.4%** |
| take-profit | 42 | 25% | +2.69% | +2.62% | +112.98% | 3.38h | 100% |
| ROI | 36 | 22% | +3.00% | +3.00% | +107.90% | 3.25h | 100% |
| time-based unclog | 6 | 4% | +1.22% | +1.13% | +7.33% | 12.00h | 100% |

**The stop is half of all exits and the only loss-making one.** It returns −72.45% against the +228.21% earned by the other three. The mix is stable across all three windows (stop share 45.1 / 54.2 / 40.0%), so this is structural rather than an artifact of one era.

Two measurements make this an open question rather than "the stop is working as intended":

1. **The stop fires fast** — median holding time 1.00h against 3.3h for the profitable exits.
2. **37.4% of stopped trades exit in profit** — so it is cutting winners as well as losers.

Whether that is correct behaviour (cutting losers early, as designed) or a defect (stopping out on noise before the entry thesis resolves) is decidable from data the project already holds, without running a single backtest.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Know what the stop is actually doing (Priority: P1)

As the researcher, I need to know whether the stop is cutting noise or cutting losers, answered from what price actually did after each stopped trade, so that the question is settled by measurement rather than by argument about stop philosophy.

**Why this priority**: It is the cheapest work in the feature — no backtests, no retraining, only the existing trade list and price history — and it determines which interventions are worth trying. Delivered alone it is a complete answer to "is the exit broken?", which is the question asked. If the stop turns out to be cutting genuine losers, the feature ends here with that finding and no code changes.

**Independent Test**: Produce the counterfactual replay report and read the verdict. No strategy change and no backtest is required to obtain it.

**Acceptance Scenarios**:

1. **Given** every stopped trade, **When** price history following the exit is replayed, **Then** the report states for each trade how far it would have recovered or fallen over a range of holding extensions.
2. **Given** the replayed outcomes, **When** they are aggregated, **Then** the report answers whether holding longer would have improved total return, and by how much.
3. **Given** that 37.4% of stopped trades exited in profit, **When** the analysis is produced, **Then** the profitable and unprofitable stops are reported **separately**, since they may be distinct failure modes with opposite remedies.
4. **Given** each stopped trade, **When** the cause is attributed, **Then** the report distinguishes stops that fired at the early-window boundary from those that fired against the later level.
5. **Given** the replay, **When** it is run, **Then** it uses only the recorded trades and existing price history — no model is loaded and no training occurs.

---

### User Story 2 - A ranked set of candidate repairs (Priority: P2)

As the researcher, I want the interventions worth testing enumerated **from the Phase 1 evidence**, each with the specific measurement that motivates it, so that effort goes to the failure mode actually observed rather than to a list drawn up in advance.

**Why this priority**: Depends on Story 1's findings. Its value is in *excluding* interventions that the evidence does not support, which is what keeps the test phase small.

**Independent Test**: Review the candidate list; each entry must cite the Phase 1 measurement that motivates it and state its expected effect. A candidate with no supporting measurement is not on the list.

**Acceptance Scenarios**:

1. **Given** the Phase 1 report, **When** candidates are enumerated, **Then** each cites the specific measurement motivating it.
2. **Given** a candidate, **When** it is listed, **Then** it is confirmed to require no retraining.
3. **Given** the anticipated candidates, **When** the list is produced, **Then** any that Phase 1 evidence does not support is explicitly excluded with the reason.

---

### User Story 3 - Cheap screening before expensive testing (Priority: P3)

As the researcher, I want candidate exit rules scored by re-deriving each trade's outcome from price history, so that only candidates that survive that screen consume a backtest.

**Why this priority**: The screen is where the cost saving lives. A rule that cannot improve outcomes on trades that already happened will not improve them in a backtest, and can be discarded for the cost of a calculation.

**Independent Test**: Score two or more candidate rules against the recorded baseline outcomes and rank them, with no backtest run.

**Acceptance Scenarios**:

1. **Given** a candidate exit rule, **When** it is screened, **Then** its outcome is derived from recorded trades and price history without running a backtest.
2. **Given** a candidate that fails the screen, **When** results are reported, **Then** it is rejected without consuming a backtest.
3. **Given** the screen, **When** it is validated, **Then** replaying the *current* rule reproduces the recorded outcomes, confirming the screen models the live exit stack correctly.

---

### User Story 4 - Confirm survivors under real execution (Priority: P4)

As the researcher, I want candidates that pass the screen confirmed by full evaluation across all three windows, because the screen cannot model position limits, one-open-trade-per-pair blocking, or the knock-on effect of an earlier exit freeing capital for a later entry.

**Why this priority**: Necessary before adopting anything, but only worth spending on candidates the screen has already endorsed.

**Independent Test**: Each surviving candidate is evaluated against the unchanged baseline over the same pinned windows and accepted or rejected on its own.

**Acceptance Scenarios**:

1. **Given** a candidate, **When** it is evaluated, **Then** it runs over all three pinned windows with realistic execution constraints active.
2. **Given** a candidate, **When** it is judged, **Then** the decision uses realized profit and risk-adjusted return, never a model-fit metric.
3. **Given** a candidate, **When** it improves the deployment-regime window but reverses sign in the temporally distant window, **Then** it is rejected as era-specific.
4. **Given** a candidate, **When** its evaluation produces an outcome identical to the baseline, **Then** it is reported as having had no effect — a defect to investigate, never a result about the intervention.
5. **Given** any candidate, **When** it is specified, **Then** it changes exit behaviour only; entry behaviour is untouched.

---

### Edge Cases

- **A stopped trade has too little subsequent price history.** Trades near the end of a window cannot be replayed over the longest extension. Those must be excluded from the affected horizon and the exclusion count reported, never silently treated as no-recovery.
- **The screen and a full evaluation disagree.** The screen cannot model position limits or capital freed by an earlier exit. Disagreement is expected and informative; the full evaluation is authoritative, and the size of the disagreement must be reported as a measure of how far the screen can be trusted.
- **An intervention shifts trades between exit categories rather than improving them.** Converting stops into equally-bad time-based exits is not an improvement. Judgement is on total outcome, not on the share of any exit reason.
- **A candidate improves total return while worsening drawdown.** Both must be reported; a return gain bought with disproportionate drawdown is a trade-off for the researcher to weigh, not an automatic acceptance.
- **The profitable and unprofitable stops want opposite remedies.** Widening the stop may help the trades that recovered while worsening the ones that fell further. If so, the report must say that a single change cannot serve both, rather than averaging them into one recommendation.
- **A parameter is adjusted but nothing changes.** The exit side has multiple parameters that look tunable but do nothing. Any unchanged outcome must be treated as an inert-parameter defect until proven otherwise.

## Requirements *(mandatory)*

### Functional Requirements

**Characterisation (Story 1)**

- **FR-001**: The analysis MUST replay, for every stopped trade, what price did after the exit, over a range of holding extensions from roughly one hour to roughly one day.
- **FR-002**: The analysis MUST report, per stopped trade, the best and worst outcome reachable within each extension.
- **FR-003**: The analysis MUST report whether each stopped trade would have reached the take-profit or ROI level had it been held.
- **FR-004**: The analysis MUST report profitable and unprofitable stops as **separate populations** throughout.
- **FR-005**: The analysis MUST attribute each stop to the mechanism that triggered it, distinguishing the early-window boundary from the later level.
- **FR-006**: The analysis MUST state a single verdict on whether the stop is predominantly cutting noise or cutting losers, with the aggregate figure supporting it.
- **FR-007**: The analysis MUST run without loading a model and without training.
- **FR-008**: The analysis MUST report how many trades were excluded from each holding extension for insufficient subsequent history.

**Candidate identification (Story 2)**

- **FR-009**: Every candidate intervention MUST cite the Phase 1 measurement that motivates it.
- **FR-010**: Every candidate MUST be confirmed to require no retraining before it is listed.
- **FR-011**: Anticipated candidates that the evidence does not support MUST be explicitly excluded, with the reason recorded, rather than omitted silently.
- **FR-012**: The candidate list MUST distinguish interventions that adjust an existing setting from those that enable currently-dormant behaviour, since the latter carry materially more risk.

**Screening (Story 3)**

- **FR-013**: The screen MUST score a candidate exit rule by re-deriving trade outcomes from recorded trades and price history, without running a backtest.
- **FR-014**: The screen MUST be validated by reproducing the recorded baseline outcomes under the current rule before any candidate is scored against it.
- **FR-015**: The screen MUST report, for each candidate, the change in total outcome relative to the recorded baseline.
- **FR-016**: A candidate that does not improve the baseline under the screen MUST be rejected without consuming a full evaluation.

**Confirmation (Story 4)**

- **FR-017**: Every surviving candidate MUST be evaluated over all three pinned windows.
- **FR-018**: Every evaluation MUST run with realistic execution constraints active.
- **FR-019**: Every evaluation MUST be decided on realized profit and risk-adjusted return. Model-fit metrics MUST NOT be used.
- **FR-020**: The headline verdict MUST be the result in the deployment-regime window; other windows are persistence checks and context.
- **FR-021**: A candidate that reverses sign in the temporally distant window MUST be rejected.
- **FR-022**: Every evaluation MUST report its trade count; a zero-trade or few-trade result is inconclusive, not a null.
- **FR-023**: Every evaluation MUST be checked for inertness — an outcome identical to the baseline means the change did not take effect and MUST be reported as a defect, not as evidence about the intervention.
- **FR-024**: No candidate may alter entry behaviour. Entry parameters were validated by a separate study and are fixed for this feature.
- **FR-025**: No candidate may require retraining.
- **FR-026**: No candidate may introduce per-pair special-casing.
- **FR-027**: Candidates MUST NOT adjust the settings that are known to be inert under the active stop mechanism; doing so produces a guaranteed null that is easily misread as a finding.
- **FR-028**: The baseline for this feature MUST be the exit configuration that produced the 167 characterised trades — an early-window of 5.6 hours at a −0.27 level, a take-profit of 0.024, an ROI level of 3%, and model exit signals disabled. Characterisation therefore requires no re-run.
- **FR-028a**: The output of the recent parameter search MUST be treated as one **candidate** among others and screened on the same footing, not adopted as the baseline. It is not privileged by being more recent.
- **FR-028b**: The baseline MUST be captured explicitly at the start of the work rather than read from configuration at each step. The values above come from the strategy's parameter file, which **overrides** the values declared in the strategy class; the class fallback differs on six of eight exit settings, so a step that reads the class instead of the file would silently measure a different baseline.

### Key Entities

- **Stopped trade**: One recorded trade that ended at the stop. Carries its pair, entry and exit times, realized outcome, whether it ended in profit, and the mechanism that triggered the stop.
- **Counterfactual outcome**: For one stopped trade and one holding extension — the best and worst reachable outcome over that extension, whether the take-profit or ROI level was reached, and whether sufficient history existed to evaluate it.
- **Candidate intervention**: A proposed change to exit behaviour. Carries the motivating measurement, whether it adjusts an existing setting or enables dormant behaviour, its screen result, and its confirmation result.
- **Evaluation arm**: One candidate measured over one window. Carries trade count, exit-reason mix, realized profit, risk-adjusted return, drawdown, and an inertness check against the baseline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The question "is the stop cutting noise or cutting losers?" is answered with a stated verdict and the aggregate measurement behind it.
- **SC-002**: Every stopped trade is replayed over every holding extension, or explicitly excluded with a counted reason.
- **SC-003**: Profitable and unprofitable stops are reported separately at every stage, and the report states whether one intervention can serve both.
- **SC-004**: The screen reproduces the recorded baseline outcomes under the current rule, establishing that it models the live exit stack correctly.
- **SC-005**: Every candidate carries a citation to the measurement that motivates it; none is listed on prior expectation alone.
- **SC-006**: At least one candidate is rejected at the screen stage without consuming a full evaluation, demonstrating the cost saving the screen exists for.
- **SC-007**: No step of the feature loads a model for training or produces new model weights.
- **SC-008**: Every confirmed candidate reports trade count, exit-reason mix, realized profit, risk-adjusted return and drawdown for all three windows.
- **SC-009**: An accepted candidate improves risk-adjusted return in the deployment-regime window without reversing sign in the temporally distant window.
- **SC-010**: A finding that the exit is already well-calibrated, and that no candidate improves it, counts as a successful completion of this feature.
- **SC-011**: The whole analysis re-runs from a clean checkout with no manual path or parameter edits.

## Assumptions

- **Baseline trades**: The 167 recorded trades across the three windows, produced by the entry configuration validated in feature 001 and the exit configuration named in FR-028, are the input to characterisation. Because the baseline is that same configuration, characterisation does not need re-running. If the exit configuration changes before the work starts, characterisation is re-run rather than carried over.
- **Holding extensions**: Roughly 1, 3, 6, 12 and 24 hours, chosen to bracket the observed 1.00h stop duration and the 3.3h profitable-exit duration, and to reach well beyond the 12-hour time-based exit.
- **Acceptance bar**: Improvement in risk-adjusted return with a consistent sign across all three windows. Absolute return alone is insufficient — a change that raises return and drawdown proportionally is exposure adjustment, not improvement.
- **Screen fidelity**: The screen models the exit stack — stop, take-profit, ROI and time-based exit — but not position limits, one-open-trade-per-pair blocking, or capital freed by an earlier exit. It is a filter, not a substitute for full evaluation.
- **Determinism**: A full evaluation with a fixed model is deterministic, so a repeated run producing an identical result set proves a parameter had no effect. Run-to-run variance is not a factor; variance from a differently trained model is out of scope because nothing retrains.
- **Dormant behaviour**: The model already emits exit predictions that are currently discarded. Enabling them requires no retraining, but is a change of kind rather than degree, and prior work records a large regression when exit handling was altered on a related strategy.
- **Governance**: Ratified constitution v1.0.0 applies. This feature is expected to exercise the "verify parameters are actually live" principle heavily, as the exit side holds several settings that look tunable and are not.

## Out of Scope

- Retraining, fine-tuning, or producing new model weights of any kind.
- Any change to entry behaviour or entry parameters.
- Re-tuning the settings that are inert under the active stop mechanism.
- New data sources beyond existing price history and recorded trades.
- Per-pair overrides or pair-specific exit rules.
- Changes to position sizing or the number of concurrent positions.

## Resolved Questions

- **Question 1 — the baseline exit configuration**: RESOLVED 2026-09-02. The configuration that produced the 167 characterised trades (early-window 5.6h / −0.27, take-profit 0.024, ROI 3%, model exits disabled). Chosen over the recent parameter-search output because characterisation is already done against it, and because that output pinned take-profit to the minimum of its allowed range — a boundary artifact that would become the yardstick for everything else if adopted as the baseline. It is instead screened as a candidate, which also yields a direct read on whether the search improved the exit or merely moved it. See FR-028, FR-028a, FR-028b.
