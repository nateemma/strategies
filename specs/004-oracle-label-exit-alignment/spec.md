# Feature Specification: Oracle-First Label/Exit Alignment for NNNC

**Feature Branch**: `004-oracle-label-exit-alignment`

**Created**: 2026-09-05

**Status**: Draft

**Input**: User description: "Oracle-first label/exit alignment for NNNC — investigate why perfect prediction loses money without the guards, and establish the process that should govern this kind of work."

## Context

A perfect-prediction ceiling test on 2026-09-05 (`regime/GUARD_ABLATION.md`)
produced two results that redefine what work on this strategy can be worth.

| arm | W1 tr | W1 ret% | W1 PF | W3 tr | W3 ret% |
|---|---|---|---|---|---|
| GUARDS ON, no model at all | 99 | 9.67 | 2.13 | 62 | 3.68 |
| GUARDS ON, real model (production) | 95 | 10.45 | 2.34 | 56 | 3.37 |
| GUARDS ON, PERFECT model | 89 | 11.12 | 2.71 | 55 | 4.41 |
| GUARDS OFF, no model at all | 6015 | −42.12 | 0.88 | 5745 | −5.38 |
| GUARDS OFF, real model | 1970 | −14.46 | 0.90 | 1712 | 19.29 |
| GUARDS OFF, PERFECT model | 1760 | −18.89 | 0.84 | 1588 | −11.69 |

**Model headroom is 0.68pp.** The whole span from no model to a perfect model
with guards on is 1.45pp on W1, of which the real model already captures 0.78pp.

**A perfect model loses money without the guards** — worse than the real model
on W1, and far worse on W3 where the real model makes +19.29%.

Together these say the guard chain is not filtering bad predictions. It is
filtering out bars where **the label itself is unprofitable under the current
exit stack**, admitting only the ~95 bars per window where the label and the
exit machinery happen to agree. The oracle self-validates on the guards-on rows
(PF 2.71 vs 2.34, win rate 75.3% vs 74.7%), so the guards-off rows describe the
label/exit relationship rather than a broken arm.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Locate the mechanism that destroys the labelled gain (Priority: P1)

As the researcher, I want to know **which exit mechanism** turns perfect
foreknowledge into a loss, so that I am changing the thing that is actually
broken rather than guessing.

**Why this priority**: Every other story depends on this answer, and Constitution
Principle VI forbids tuning before a diagnostic localises the fault. It is also
the cheapest work in the feature — the exit reason is already recorded on every
trade, so the primary diagnostic needs one existing backtest and no new runs.

**Independent Test**: A per-exit-reason breakdown of the oracle's guards-off
trades exists, giving trade count, total P&L contribution, mean P&L and mean
hold time for each reason, and the reasons are ranked by total P&L destroyed.
Delivered alone, this already tells the researcher where the loss comes from.

**Acceptance Scenarios**:

1. **Given** the oracle running guards-off on W1, **When** its trades are grouped
   by exit reason, **Then** each reason reports count, summed P&L, mean P&L and
   mean duration, and the reasons are ranked by total P&L contribution.
2. **Given** that breakdown, **When** it is compared against the same breakdown
   for the oracle running guards-ON, **Then** the difference identifies which
   reasons the guard chain is suppressing.
3. **Given** a candidate mechanism, **When** its share of destroyed P&L is
   stated, **Then** the statement is accompanied by the trade count supporting
   it, so a mechanism responsible for a handful of trades is not mistaken for
   the dominant cause.

---

### User Story 2 - Establish the oracle-first screening gate as standing process (Priority: P1)

As the researcher, I want a written, reusable rule that screens any proposed
label or exit change against the oracle **before** a model is trained, so that
no further training budget is spent on changes that provably cannot help.

**Why this priority**: This is the durable deliverable. The preceding programme
spent many retrains at 8–45 minutes each on changes that the ceiling test now
shows could never have been worth more than 0.68pp. The oracle needs no
training, so the same question is answerable in minutes. P1 alongside US1
because it is independent of what US1 finds and is valuable even if the
diagnosis is inconclusive.

**Independent Test**: A documented procedure exists stating what is measured,
on which windows, what constitutes a pass, and what a failing change means;
and it is applied to at least one worked example end to end.

**Acceptance Scenarios**:

1. **Given** a proposed label or exit change, **When** the gate is applied,
   **Then** it yields one of pass / fail / inconclusive using realised P&L only,
   and never a classifier metric.
2. **Given** a change that does not raise the oracle's guards-off return,
   **When** the gate is applied, **Then** the change is recorded as unable to
   help and is not trained.
3. **Given** an oracle result, **When** it is reported anywhere, **Then** it is
   labelled as a lookahead ceiling and never as an achievable return.

---

### User Story 3 - Find a label or exit configuration the oracle can profit from (Priority: P2)

As the researcher, I want at least one configuration where the oracle is
profitable with guards off, so that there exists an edge a model could learn
and model quality can begin to matter again.

**Why this priority**: This is the feature's point, but it depends on US1 to aim
it and US2 to judge it. P2 because a negative outcome here is still a complete
and valuable result.

**Independent Test**: A set of label and exit variants has been screened through
the gate, each with its guards-off oracle return recorded, and the best is
stated with its trade count and how it compares to the −18.89% baseline.

**Acceptance Scenarios**:

1. **Given** the mechanism identified in US1, **When** variants addressing it are
   screened, **Then** each is reported with guards-off oracle return, trade count
   and drawdown on every pinned window.
2. **Given** a variant that makes the oracle profitable guards-off, **When** it
   is reported, **Then** it is reported as a candidate for training, not as an
   adopted change.
3. **Given** that no variant makes the oracle profitable, **When** the feature
   concludes, **Then** that is recorded as the result, together with what it
   implies about the strategy's dependence on the guard chain.

---

### User Story 4 - Separate cost drag from label quality (Priority: P2)

As the researcher, I want the fee and slippage burden of high-trade-count arms
accounted for, so that the guards-off loss is not misattributed to the label
when it is partly the cost of trading 1760 times instead of 95.

**Why this priority**: Without it, every guards-off comparison in this feature is
confounded. Constitution Principle V requires evaluation under realistic
execution, and an eighteen-fold change in trade count is exactly where that
bites. P2 because it qualifies the other stories rather than standing alone.

**Independent Test**: The total cost burden of each arm is quantified, and each
guards-off result is reported both gross and net of it.

**Acceptance Scenarios**:

1. **Given** two arms with materially different trade counts, **When** they are
   compared, **Then** the cost burden of each is reported alongside the return.
2. **Given** the oracle's guards-off loss, **When** its cost component is
   removed, **Then** the residual is stated, and whether it remains a loss.

---

### User Story 5 - Record what the ceiling means for the strategy's future (Priority: P3)

As the researcher, I want a written conclusion on what the 0.68pp headroom means
for further model work, so that the finding governs later decisions instead of
being rediscovered.

**Why this priority**: Documentation of a settled result. P3 because it consumes
no measurement budget and depends on everything above.

**Independent Test**: A written conclusion exists stating the headroom, what it
forecloses, and what would have to change for model work to pay.

**Acceptance Scenarios**:

1. **Given** the ceiling measurement, **When** the conclusion is written, **Then**
   it states the headroom figure, the windows it was measured on, and its
   dependence on the current gate.
2. **Given** a future proposal for model or GAN work on this family, **When** it
   is assessed against this conclusion, **Then** the conclusion is sufficient to
   accept or reject it without new measurement.

### Edge Cases

- **The oracle's advantage is an artefact of the label's own lookahead.** The
  label is computed from future bars, so the oracle cannot lose on signal
  quality; if it still loses, the cause is necessarily execution-side. This is
  the feature's core inference and must be stated explicitly rather than assumed.
- **A mechanism destroys P&L on very few trades.** A reason with large mean loss
  but tiny count is not the dominant cause; every attribution must carry its
  trade count.
- **Guards-off changes the pair mix, not just the trade count.** More permissive
  entry may admit systematically different instruments, so a guards-off loss may
  reflect which pairs trade rather than which bars.
- **A variant makes the oracle profitable by trading almost nothing.** A
  configuration that reaches profitability by collapsing to a handful of trades
  has not solved the problem; a minimum trade count is required for a pass.
- **The oracle is profitable guards-off but the label becomes unlearnable.** A
  label that no model can predict is useless however good its economics — but
  learnability may not be used as an acceptance criterion, only recorded.
- **W2/W3 disagree with W1.** The oracle uses no trained weights, so the usual
  in-sample objection is weaker for oracle-only arms; the treatment of this must
  be stated rather than left to interpretation.
- **A change is absorbed.** If a variant produces trades identical to its
  control, the change never reached the trades and the arm is inconclusive, not
  refuted.

## Requirements *(mandatory)*

### Functional Requirements

**Diagnosis (US1)**

- **FR-001**: The study MUST produce a per-exit-reason breakdown of the oracle's
  guards-off trades, reporting for each reason the trade count, summed P&L
  contribution, mean P&L per trade and mean hold duration.
- **FR-002**: The study MUST produce the same breakdown for the oracle running
  guards-on, so the difference shows what the guard chain suppresses.
- **FR-003**: The study MUST rank exit reasons by total P&L destroyed and
  identify the dominant mechanism, and MUST report the trade count supporting
  every attribution.
- **FR-004**: The study MUST determine whether labelled gains are systematically
  unreachable because an adverse excursion trips the stop first, reusing the
  existing path-analysis capability rather than building new machinery.
- **FR-005**: The study MUST determine whether labelled gains are systematically
  truncated by a profit cap or a hold-time limit before the labelled move
  completes.
- **FR-006**: The diagnosis MUST be completed and reported BEFORE any label or
  exit parameter is varied (Constitution VI).

**The gate (US2)**

- **FR-007**: The study MUST define an oracle-first screening gate specifying
  what is measured, on which windows, what constitutes pass, fail and
  inconclusive, and what a failing result licenses.
- **FR-008**: The gate MUST judge on realised P&L only and MUST NOT admit any
  classifier metric as an acceptance input.
- **FR-009**: The gate MUST require a minimum trade count for a pass, so that
  profitability achieved by trading almost nothing does not qualify.
- **FR-010**: The gate MUST treat a variant whose trades are identical to its
  control as inconclusive, not refuted.
- **FR-011**: The gate MUST be documented in a durable location in the
  repository and applied to at least one worked example end to end.
- **FR-012**: Every reported oracle figure MUST be labelled as a lookahead
  ceiling, and no oracle configuration may be promoted toward live trading.

**Search (US3)**

- **FR-013**: Variants MUST be screened cheapest-first, and MUST address the
  mechanism identified by the diagnosis rather than being chosen speculatively.
- **FR-014**: Each variant MUST be reported with guards-off oracle return, trade
  count and drawdown on every pinned window.
- **FR-015**: Label-side and exit-side changes MUST be varied separately before
  being combined, so their contributions are attributable (Constitution VIII).
- **FR-016**: A variant that makes the oracle profitable guards-off MUST be
  reported as a candidate for training, never as an adopted change.
- **FR-017**: If no variant succeeds, that outcome MUST be recorded as the
  result together with what it implies about the strategy's dependence on the
  guard chain.

**Costs (US4)**

- **FR-018**: The study MUST quantify the fee and slippage burden of each arm and
  report guards-off results both gross and net of it.
- **FR-019**: The study MUST state whether the oracle's guards-off loss survives
  removal of its cost component.
- **FR-020**: The study MUST report whether guards-off changes the traded pair
  mix, so a difference in instruments is not misread as a difference in bars.

**Conclusion and hygiene (US5, cross-cutting)**

- **FR-021**: The study MUST record a written conclusion on the model headroom,
  the windows it was measured on, and what it forecloses.
- **FR-022**: Acceptance MUST be judged on W1; W2 and W3 MUST be reported and
  labelled, and the treatment of in-sample status for arms that use no trained
  weights MUST be stated explicitly.
- **FR-023**: Every arm MUST carry its own strategy parameter file, so no arm
  silently runs class defaults.
- **FR-024**: No shared framework file may be left modified; hold-window or exit
  changes MUST be made as per-strategy overrides.
- **FR-025**: Arms MUST be executed sequentially.
- **FR-026**: Any arm record used as evidence MUST be confirmed to come from a
  successful run, not a failed or stale one.

### Key Entities

- **Oracle arm**: an evaluation whose entry signal is the training label itself,
  with all other machinery identical to production. A diagnostic ceiling and
  screening instrument only; never a deployable configuration.
- **Exit-reason breakdown**: for one arm, the partition of its trades by the
  mechanism that closed them, with count, summed P&L, mean P&L and mean duration.
- **Screening gate**: the rule mapping a proposed label or exit change to pass,
  fail or inconclusive on the oracle's guards-off return, plus the minimum trade
  count and the liveness requirement.
- **Variant**: one label or exit configuration under test, with its own parameter
  file and its own arm records on each pinned window.
- **Cost burden**: the fee and slippage total attributable to an arm, used to
  separate cost drag from label quality.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The dominant mechanism destroying the oracle's guards-off P&L is
  identified and stated with the share of total loss it accounts for and the
  number of trades involved.
- **SC-002**: The exit-reason breakdown covers 100% of the oracle's guards-off
  trades, so no P&L is unattributed.
- **SC-003**: The oracle-first screening gate is documented with explicit pass,
  fail and inconclusive conditions and a minimum trade count, and is applied to
  at least one worked example.
- **SC-004**: Every variant screened is reported with realised return, trade
  count and drawdown on all three pinned windows, and no variant is accepted or
  rejected on a classifier metric.
- **SC-005**: The label-side and exit-side contributions are reported separately
  before any combined variant is reported.
- **SC-006**: The cost burden of each guards-off arm is quantified, and the
  oracle's guards-off loss is stated both gross and net of cost.
- **SC-007**: The feature concludes with an explicit statement of whether any
  configuration makes the oracle profitable guards-off on W1, and if so which.
- **SC-008**: No shared framework file is modified at the end of the feature.
- **SC-009**: Every oracle figure reported is labelled as a lookahead ceiling,
  and no oracle configuration is promoted toward live trading.
- **SC-010**: The written conclusion on model headroom is sufficient to accept or
  reject a future model or GAN proposal on this family without new measurement.

## Assumptions

- The oracle's entry signal is the same label the production model is trained on,
  so the ceiling measured is the ceiling *for that label*, not for the strategy
  in general. A different label has a different ceiling, which is why label
  variants are in scope.
- The exit reason recorded on each trade is a faithful record of which mechanism
  closed it; where several could apply, the first to fire is the one recorded.
- The 0.68pp headroom figure is specific to the current gate. A configuration
  that admits materially more trades would have to have its headroom
  re-measured, and the conclusion says so.
- W2 and W3 fall inside the model's training window, but the oracle uses no
  trained weights. For oracle-only arms the in-sample objection therefore
  concerns the *label parameters* rather than fitted weights, and all three
  windows carry comparable evidential weight for those arms; W1 remains the
  acceptance window so that oracle and trained arms are judged consistently.
- Fee and slippage settings in the evaluation configuration are taken as
  representative; this feature does not re-derive them.
- Guards-off is reachable by neutralising each guard threshold rather than by
  disabling the guard block, which keeps the `volume > 0` data-sanity condition
  active. This was verified: out-of-range parameter values are honoured, not
  clamped.

## Non-Goals

- No model training, GAN work, or architecture change. The headroom is 0.68pp
  and that is settled.
- No further guard threshold tuning. The leave-one-out ablation is complete:
  `guard_metric` is inert and no other guard can be removed for gain.
- No re-litigation of feature 003's finding that improving a label's
  reward-to-risk ratio did not improve realised return. That finding stands and
  constrains what this feature may conclude from economics alone.
- No new data sources.
- No per-pair special-casing.
- No promotion of any oracle configuration toward live trading, under any result.
