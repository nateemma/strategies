# Feature Specification: Oracle-Instrumented Exit Diagnosis

**Feature Branch**: `006-oracle-exit-diagnosis`

**Created**: 2026-09-07

**Status**: Draft

**Input**: User description: "Use the oracle as an instrument to investigate whether the exit stack closes trades before their edge materialises."

## Context

### The paradox

An evaluation whose entry signal IS the training label — so it knows with
certainty that every bar it enters reaches that label's target — **earns LESS
than the real model** on the acceptance window:

| NNMT, acceptance window | trades | return |
|---|---|---|
| real model | 176 | **+10.708%** |
| perfect entry, current label | 145 | +8.733% |
| perfect entry, prior label | 155 | +6.150% |

Perfect foreknowledge of the entry should dominate a guess. It does not, and the
measured reasons are all on the exit side.

### What was measured (2026-09-07)

1. **The exit closes the trade before the promised move arrives.** For perfect
   entries whose target is genuinely reached, the median trade lives **12 bars**
   while the promised move arrives at a median of 7 bars but a **p90 of 35**,
   against a label horizon of 48. **In 24.6% of trades the exit fires before the
   guaranteed move happens.** This is the dominant leak.
2. **The label's target sits BELOW the profit-taking level** — a 2.0% target
   against a 2.2% take-profit. A bar can satisfy its label exactly and never
   trigger a profitable exit. An earlier study measured 12.6% of labelled bars
   never reaching the take-profit at all.
3. **Path-blindness**: 6.6% of perfect entries breach the stop band on the way
   to their target. **This figure is provisional and understates the problem** —
   it was computed with the wrong family's stop band (−4% to −6% instead of −2%
   to −4%), so the true rate is higher. It MUST be recomputed per family before
   being used.
4. **Unexplained**: only **84.1%** of perfect entries meet their own label when
   reconstructed from raw candle data. The obvious cause was tested and
   rejected — measuring from the signal bar gives 81.4% and from two bars back
   64.1%, so the entry bar is the best alignment and there is no off-by-one.

**Mechanism evidence.** Against the real model, perfect entries avoid only **2 of
77** stop exits but lose **15 of 39** profit-taking exits. Entry quality barely
moves the stop count. That is what makes this an exit-side question.

### Why this is not a repeat of earlier exit work

The exit has been studied here several times and judged well calibrated: the
profit-taking level was found right and robust, stop-tuning proved to be pair-mix
overfit that reversed in a distant era, and several exit parameters were found
inert. **Every one of those studies was run on a real model**, where entry noise
dominates and an exit change moves a handful of trades inside a wide band.

A perfect-entry instrument removes that confound entirely: **entry quality is
perfect and CONSTANT across arms, so any difference in outcome is attributable to
the exit alone.** That is a new instrument for a question this repository has
only ever answered noisily. It is the justification for revisiting the exit, and
it is also the reason the specific changes already refuted must not be re-run.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trust the instrument before using it (Priority: P1)

As the researcher, I want the 15.9% of perfect entries that do not satisfy their
own label explained, so that every conclusion drawn from this instrument rests on
a measurement I can defend.

**Why this priority**: This is measurement integrity and it blocks everything
else. Every ceiling figure recorded in this repository carries this slop, and an
exit conclusion built on a mis-specified instrument would be worse than no
conclusion. It requires no training and no new configuration.

**Independent Test**: The compliance rate is either explained and corrected, or
quantified as a known and bounded error with a stated effect on every downstream
figure.

**Acceptance Scenarios**:

1. **Given** the perfect-entry arm's trade list, **When** each entry is checked
   against the label definition on the same data the strategy saw, **Then** the
   compliance rate is reported and any shortfall is attributed to a named cause.
2. **Given** an identified cause, **When** it is corrected, **Then** compliance
   is re-measured and the effect of the correction on the headline figures is
   stated.
3. **Given** a shortfall that cannot be eliminated, **When** it is reported,
   **Then** it is expressed as a bound on every conclusion this feature draws,
   rather than being set aside.

---

### User Story 2 - Establish what a perfect exit would be worth (Priority: P1)

As the researcher, I want to know what a perfect entry earns when the exit simply
waits for the move it was promised, so that I have an upper bound on everything
the exit is currently leaving on the table.

**Why this priority**: Without this bound, no exit change can be judged. It
converts "the exit closes too early" from an observation into a number, and it
tells us whether the remaining exit work is worth minutes or weeks. Needs no
training.

**Independent Test**: A perfect-entry arm exists whose exit is held to the
label's own horizon, with its return compared to both the current exit and the
real model.

**Acceptance Scenarios**:

1. **Given** perfect entries, **When** the exit is made to wait for the label
   horizon, **Then** the return is reported alongside the current exit's return
   and the difference is stated as the exit's opportunity cost.
2. **Given** that bound, **When** it proves small, **Then** the feature states
   that the exit is not the constraint and re-scopes the remaining work.
3. **Given** that bound, **When** it proves large, **Then** the individual exit
   mechanisms are ranked by how much of the gap each one accounts for.

---

### User Story 3 - Attribute the loss to specific exit mechanisms (Priority: P2)

As the researcher, I want each exit mechanism's contribution to the gap measured
separately, so that any change is aimed at the mechanism actually responsible.

**Why this priority**: The programme's standing rule is to diagnose before
tuning. With entry quality held constant, a per-mechanism attribution is finally
clean. P2 because it is only worth doing if US2 finds a gap worth attributing.

**Independent Test**: Each exit mechanism is neutralised in turn on the
perfect-entry arm, and its individual effect is reported alongside the joint
effect so interaction is distinguished from addition.

**Acceptance Scenarios**:

1. **Given** the perfect-entry arm, **When** each exit mechanism is relaxed in
   turn, **Then** its individual effect on realised return and on the exit-reason
   mix is reported.
2. **Given** those results, **When** a mechanism changes nothing, **Then** it is
   reported as INERT rather than as correct.
3. **Given** the individual effects, **When** they are compared with the joint
   effect, **Then** the report states whether they sum, so interaction is visible.

---

### User Story 4 - Resolve the ordering of the target and the profit-taking level (Priority: P2)

As the researcher, I want to know whether a target that sits below the
profit-taking level is an ordering error, so that the label certifies a move the
exit can actually harvest.

**Why this priority**: It is a specific, cheap, falsifiable hypothesis about a
known 12.6% failure, and the perfect-entry instrument is the only place it can be
tested without the label's accuracy confounding it.

**Independent Test**: The relationship between the two levels is varied on the
perfect-entry arm and its effect on the share of trades that exit at a profit is
reported.

**Acceptance Scenarios**:

1. **Given** perfect entries, **When** the profit-taking level is placed below,
   at, and above the target, **Then** the share of trades harvesting the promised
   move is reported for each.
2. **Given** an ordering that proves better, **When** it is reported, **Then** it
   is stated whether the gain comes from harvesting more moves or from taking
   smaller ones.

---

### User Story 5 - Confirm anything found on the real model (Priority: P1)

As the researcher, I want every candidate exit change re-measured on the real
model, because a perfect-entry improvement that does not survive contact with a
real model is not a change worth making.

**Why this priority**: P1 despite being last, because it is the ONLY step that
can license a promotion. The standing finding is that a perfect-entry instrument
bounds what a LABEL can pay and cannot settle how the EXIT should harvest — the
same change was worth almost nothing at the ceiling and several points on a real
model. This step is inside the feature, not a follow-on.

**Independent Test**: Each candidate is measured on the real model across the
required seeds, and its verdict is decided by that measurement rather than by the
perfect-entry result.

**Acceptance Scenarios**:

1. **Given** a candidate that helps under perfect entries, **When** it is applied
   to the real model, **Then** its effect is reported on the acceptance window
   with paired resolution and across the required seeds.
2. **Given** a candidate that helps under perfect entries but not on the real
   model, **When** reported, **Then** it is REJECTED for promotion and recorded
   as an instrument-only result.
3. **Given** a candidate that helps both, **When** promoted, **Then** its
   capacity behaviour under a larger book is reported first.

---

### User Story 6 - Record what the exit can and cannot be blamed for (Priority: P3)

As the researcher, I want the outcome recorded in a form that settles the exit
question for both families, so that it is not reopened without new information.

**Why this priority**: Documentation of a settled result; consumes no measurement
budget and depends on everything above.

**Independent Test**: A written conclusion states the exit's opportunity cost, the
mechanism responsible, what transferred to the real model, and what is foreclosed.

**Acceptance Scenarios**:

1. **Given** the measurements, **When** the conclusion is written, **Then** it
   states the bound, the responsible mechanism, and the windows measured on.
2. **Given** a future proposal to change the exit, **When** assessed against this
   conclusion, **Then** it can be accepted or rejected without new measurement.

### Edge Cases

- **The compliance gap has no identifiable cause.** Then it is quantified as a
  bound on every figure this feature produces, and the feature says so rather
  than proceeding as if the instrument were exact.
- **The perfect exit is worth little.** Then the exit is not the constraint, the
  paradox is explained by the target's own weakness rather than by the exit, and
  the remaining stories are re-scoped.
- **A mechanism helps under perfect entries and hurts on the real model.** This
  is the expected failure mode and is a result, not an anomaly: perfect entries
  only enter bars that reach their target, so an exit that waits is rewarded
  there and punished on entries that never get there.
- **The acceptance window and the in-sample windows disagree.** The third window
  retains only 7.5% of its gross flow as profit against the first window's 29.8%,
  and its four worst trades exceed the entire window's profit, so its sign is
  close to arbitrary under perturbation. Windows are reported with that ratio.
- **A change moves no trades.** Identical trade sets mean the change never
  reached the trades; that is INCONCLUSIVE, never a null result.
- **An arm produces zero trades.** Inconclusive, never a zero-return result.

## Requirements *(mandatory)*

### Functional Requirements

**Instrument integrity (US1)**

- **FR-001**: The study MUST report the share of perfect entries that satisfy
  their own label, measured against the same data the strategy consumed.
- **FR-002**: Any shortfall MUST be attributed to a named cause, or reported as a
  quantified bound on every figure the feature produces.
- **FR-003**: Where a cause is corrected, the effect of the correction on the
  previously reported figures MUST be stated.

**The bound (US2)**

- **FR-004**: The study MUST measure what perfect entries earn when the exit is
  held to the label's own horizon, and report the difference against the current
  exit as the exit's opportunity cost.
- **FR-005**: If that opportunity cost is not materially larger than the study's
  resolution, the study MUST state that the exit is not the constraint and MUST
  re-scope the remaining work rather than executing it as written.

**Attribution (US3)**

- **FR-006**: Each exit mechanism's individual contribution to the gap MUST be
  reported, with its effect on both realised return and the exit-reason mix.
- **FR-007**: A mechanism that changes neither trades nor return MUST be reported
  as INERT rather than as correct.
- **FR-008**: The study MUST state whether individual mechanism effects sum to
  the joint effect, so interaction is distinguished from addition.

**Target ordering (US4)**

- **FR-009**: The study MUST measure the effect of placing the profit-taking
  level below, at, and above the label's target.
- **FR-010**: Where an ordering proves better, the study MUST state whether the
  gain comes from harvesting more moves or from taking smaller ones.

**Real-model confirmation (US5)**

- **FR-011**: Every candidate MUST be re-measured on the real model, and its
  promotion verdict MUST rest on that measurement, never on the perfect-entry
  result.
- **FR-012**: A candidate that helps under perfect entries but not on the real
  model MUST be recorded as an instrument-only result and MUST NOT be promoted.
- **FR-013**: Any promoted change MUST have its capacity behaviour under a larger
  book reported before promotion.

**Evidence standards (cross-cutting)**

- **FR-014**: Acceptance MUST be judged on realised return on the out-of-sample
  window only; the other two MUST be reported and labelled in-sample.
- **FR-015**: Every window MUST be reported with the share of its gross flow
  retained as profit, not only its trade count, so a window whose result is a
  thin residual is visibly less informative.
- **FR-016**: Resolution MUST be computed by a paired comparison over the trades
  that differ between an arm and its control; a flat threshold MUST NOT be used.
  Where arms share entries and differ only in exits, the pairing MUST be per
  trade on the difference in outcome.
- **FR-017**: No classifier metric may be an acceptance input.
- **FR-018**: Any result used to accept or reject MUST be measured across at
  least two training seeds where a model is trained, with the spread reported.
- **FR-019**: Every perfect-entry figure MUST be labelled a lookahead bound and
  MUST NOT be reported as achievable; no such configuration may be promoted.
- **FR-020**: An arm whose trades are identical to its control MUST be reported
  as INCONCLUSIVE, and a zero-trade arm MUST be reported as inconclusive rather
  than as a zero return.
- **FR-021**: Every arm MUST carry its own parameter file, and any strategy that
  is a pure re-parameterisation of another MUST be made to load the other's model
  rather than training its own.
- **FR-022**: Arms MUST be executed sequentially.
- **FR-023**: No shared framework mechanism may be left modified; the live dry
  run depends on it.
- **FR-024**: Any arm record used as evidence MUST come from a confirmed
  successful run.
- **FR-025**: The entry guards MUST NOT be opened on either family; both have
  been measured to lose materially when they are.

**Conclusion (US6)**

- **FR-026**: The study MUST record the exit's opportunity cost, the mechanism
  responsible, what transferred to the real model, and what the result forecloses.

### Key Entities

- **Perfect-entry arm**: an evaluation whose entry signal is the training label
  itself. Diagnostic only; never deployable.
- **Exit opportunity cost**: the difference between what perfect entries earn
  under the current exit and under an exit that waits for the label horizon.
- **Exit mechanism contribution**: for one mechanism, its individual effect on
  realised return and exit-reason mix when relaxed alone.
- **Window informativeness**: the share of a window's gross flow retained as
  profit, used to weight how much its verdict counts.
- **Instrument compliance**: the share of perfect entries that satisfy their own
  label, and the bound any shortfall places on the feature's conclusions.

## Success Criteria *(mandatory)*

- **SC-001**: The compliance of the perfect-entry instrument is stated as a
  figure, with any shortfall either explained or expressed as a bound on every
  conclusion.
- **SC-002**: The exit's opportunity cost under perfect entries is stated as a
  single number with the windows and trade counts it was measured on.
- **SC-003**: Each exit mechanism has its individual contribution reported, any
  inert mechanism is named as inert, and the sum is compared to the joint effect.
- **SC-004**: The ordering of the label's target and the profit-taking level is
  resolved, with the mechanism of any gain stated.
- **SC-005**: Every candidate is re-measured on the real model, and no candidate
  is promoted on a perfect-entry result alone.
- **SC-006**: Every window is reported with the share of gross flow it retains,
  and no sub-threshold conclusion rests on the least informative window.
- **SC-007**: No configuration is accepted on a classifier metric, and every
  perfect-entry figure is labelled a lookahead bound.
- **SC-008**: No shared framework mechanism is left modified at the end of the
  feature.
- **SC-009**: The written conclusion is sufficient to accept or reject a future
  exit proposal on either family without new measurement.

## Assumptions

- The perfect-entry instrument reflects the production entry path apart from the
  prediction source, so differences between it and the real model are
  attributable to entry quality rather than to plumbing. US1 tests this
  assumption rather than relying on it.
- The label horizon is a meaningful reference for how long a trade "should" be
  held. It is the window over which the label's promise is defined, so an exit
  that closes earlier is provably closing before the promise matures — but it is
  not a claim that holding that long is optimal.
- Exit mechanisms can be varied through configuration without modifying shared
  framework code. Where one cannot, it is out of scope for this feature rather
  than a reason to modify shared code.
- Two seeds are sufficient to detect an effect several times larger than the
  spread. A comparable configuration showed a headline advantage of nearly ten
  points that proved to be seed luck, with a spread several times the effect.
- Fills are realistic because entry size is capped in proportion to available
  volume rather than rejected, so liquidity risk appears as reduced position size
  rather than as impossible trades.

## Non-Goals

- Entry-side work of any kind. A perfect entry is worth less than nothing on one
  family, and opening the other's gate loses materially despite large headroom.
- Label fidelity. A path-aware label was built, passed the screening gate on both
  families, and produced nulls on both.
- Model architecture, augmentation, new data sources, and per-pair special-casing.
- Re-testing the two exit changes already refuted at the CURRENT configuration:
  the adaptive stop multiplier of 15 on the single-task family, measured at
  roughly three points of loss, and the exit parameters found inert under the
  current stop machinery.
  **This does NOT close the stop band generally.** The earlier study's band arms
  were all measured against a label that lost 19% at the ceiling, so they compared
  rearrangements of a losing hand and say nothing about the band at a label that
  pays. Re-asking that question at the current label is IN scope.
