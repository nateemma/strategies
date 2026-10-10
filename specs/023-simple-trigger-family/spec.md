# Feature Specification: SimpleTrigger Strategy Family

**Feature Branch**: `023-simple-trigger-family`

**Created**: 2026-10-10

**Status**: Draft

**Input**: User description: "I want to add a new strategy family called SimpleTrigger. It is
essentially the same as NNNC, but with no model involved. The goal is to be able to evaluate the
overall framework (entry triggers, gain/loss targets, custom stoploss, entry/exit guards), and
also to assess the possible performance without the model involved. My guess is that it would
look something like SimpleStrategies, but those have extra logic trying to deal with regimes that
we don't want, at least initially (one possible investigation might be to add regime detection
and regime-specific parameters). The strategies should be able to use any of the label types that
are causal (i.e. no lookahead). It should be a goal to be able to use these strategies to better
investigate/validate approaches examined in recent features 21 and 22. Further investigations
should make it possible to assess things like making the stoploss logic coupled to the gain/loss
targets, since that has already caused issues, plus we should be able to evaluate the whole
framework together. The overall goal is to provide the best environment that we can then use for
NNNC and focus on model improvements without the framework interfering."

## Context

An NNNC strategy is two things layered together: a **model** that gives an entry opinion, and a
**framework** around it. The framework is the entry trigger, the guard chain, the gain/loss
targets, the exit rules (take-profit, max hold, exit signal) and the custom stoploss (ATR band,
grace period, volume confirmation). At present the two can't be separated cleanly, so every
NNNC result mixes model effects with framework effects.

The evidence that this matters (all from the 15-fold pooled walk-forward harness):

| arm | model? | result |
|---|---|---|
| `gbb` trigger alone, guards open (021) | no | +2.64pp, 7/15 folds |
| tightened static guard constant, `wf_gbbr` (022) | no | +25.34pp, 13/15 folds |
| arm D: `gbb` + 24 indicators + per-fold threshold, guards open (incumbent) | yes | +39.98pp, 15/15 folds |

The open question the project now faces is model-side: how much of arm D comes from the model?
That question can only be answered against a no-model arm built from **exactly the same
framework**. Three partial no-model mechanisms exist, and none of them is that arm:

- **`TriggerOnly` (021)**: has one trigger and one fixed exit, and its guards are always open.
  It can't vary the framework, which by design was the point for 021.
- **`NNNC_NoModel`**: still loads a model, applies no trigger, and accepts every bar. It measures
  the guard chain on its own, not trigger plus framework.
- **`SimpleStrategies`**: indicator strategies with a built-in bear/bull/sideways regime filter,
  `exit_profit_only` and their own exit logic. That framework is a different one.

Recorded framework defects show why the framework needs its own test bench:

- The post-grace stop sign flip turned a −4% stop into −46%.
- The take-profit floor was mis-scaled.
- The trailing stop was applied twice.
- The `entry_enable_guards` comment was stale.
- Exit parameters had no effect under the custom stop.
- The label and the exit stack disagreed (the oracle ceiling).

All of these defects were found indirectly, through NNNC runs where the model hid them.

## Clarifications

### Session 2026-10-10

- Q: How does SimpleTrigger relate to NNNC? → A: Sibling now, merge later. SimpleTrigger is a
  separate family that copies NNNC's framework and is held to it by the equivalence check
  (FR-008). NNNC is not restructured in this feature. Restructuring NNNC as SimpleTrigger plus a
  model gate is a recorded follow-up, to be done once the equivalence check has passed.
- Q: Which NNNC operating point does the default configuration mirror? → A: The guards-open
  research operating point (`NNNC_MLX_Open` / `NNNC_A_Raw_Open`, arm D), the one used throughout
  021–022.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run any causal trigger through the NNNC framework with no model (Priority: P1)

The researcher picks a registered causal trigger (for example `gbb`, `breakout`, `adx`, `empty`)
and a framework configuration. They run it as a SimpleTrigger strategy on the walk-forward
harness exactly as they would run an NNNC arm. Nothing is trained, and no per-fold model
artefact is needed. The run produces the same per-fold P&L and trade records, with the same
comparability guarantees, as any other arm.

**Why this priority**: Every other story depends on it. On its own it already answers "what does
this trigger plus this framework earn with no model at all?" for any causal label. It also
replaces `TriggerOnly` as the no-model instrument.

**Independent Test**: Re-run the 021 trigger-only arms and 022's `wf_gbbr` arm as SimpleTrigger
configurations and compare them with the recorded results.

**Acceptance Scenarios**:

1. **Given** the six 021 trigger-only arms with their recorded per-fold results, **When** each is
   re-expressed as a SimpleTrigger configuration and run over the same 15-fold plan, **Then**
   every fold's trade set is identical to the recorded one (same trade fingerprint).
2. **Given** 022's `wf_gbbr` arm (`atr_pct 0.013 / adx 20 / rvol 1.9`, all other guards open),
   **When** it is re-expressed as a SimpleTrigger configuration, **Then** it reproduces the
   recorded +25.34pp, 13/15 folds and 458 trades with identical trade sets.
3. **Given** a trigger name that isn't registered, or a trigger that fails evaluation, **When**
   the strategy runs, **Then** it enters no trades and reports why, and never falls back to a
   hand-written copy of the condition.
4. **Given** any SimpleTrigger run, **When** it completes, **Then** no model was loaded or
   trained and no saved-model directory was created for the strategy class.

---

### User Story 2 - Isolate and evaluate each framework component (Priority: P2)

The researcher wants to know what each framework component contributes and whether it is wired
correctly. The components are the trigger, each guard threshold, the take-profit, max-hold, the
exit signal, the custom stop's ATR band, grace and volume confirmation. The researcher sets any
single component to its documented neutral value (off or fully open), leaves everything else
fixed, and gets a per-component comparison against the full configuration. Each comparison
carries proof that the component actually changed behaviour.

**Why this priority**: The framework has been the source of recurring defects, and each one was
hidden by the model. Without the model, the trigger's bar set is fixed, so a component's effect
can be attributed directly, and an inert component shows up immediately (Principle IV).

**Independent Test**: Run the full configuration and one neutralised arm per component on the
same fold plan. Check that each neutralised arm either changes the trade set or is flagged as
inert.

**Acceptance Scenarios**:

1. **Given** a SimpleTrigger configuration, **When** one component is set to neutral, **Then**
   only that component's behaviour changes, and the run records a downstream measurement
   (pass rate, trade count or trade-set difference) proving the change was live.
2. **Given** a component whose neutralised arm produces a trade set identical to the full
   configuration, **When** results are reported, **Then** that component is flagged INERT as a
   defect, not reported as "no effect".
3. **Given** a set of component arms, **When** they are reported, **Then** the report shows every
   component's paired delta against the full configuration, with fold-level detail and a stated
   power for the comparison.

---

### User Story 3 - Measure the model's contribution against an identical no-model twin (Priority: P2)

For any NNNC arm, the researcher can build its **SimpleTrigger twin**. The twin has the same
trigger the NNNC label is built from, the same guards, targets, exits, stop, sizing and
execution constraints. The only difference is that no model gives an entry opinion. A paired
comparison of the two on the same folds isolates the model's entire contribution. The first
use is arm D's +39.98pp.

**Why this priority**: It is the project's one live thread. It turns "something real is there"
into a measured split between model and framework, and it gives NNNC work a fixed framework to
measure model changes against.

**Independent Test**: Build arm D's twin, run both on the 15-fold plan, and report the paired
model contribution with its confidence interval.

**Acceptance Scenarios**:

1. **Given** an NNNC arm, **When** its twin is built, **Then** a check confirms that every
   framework setting is identical between the two, and any mismatch blocks the comparison.
2. **Given** an NNNC arm whose model is replaced by an "accept every trigger bar" opinion,
   **When** it is run against its twin, **Then** the trade sets are identical. This proves the
   framework is shared and that the model is the only difference.
3. **Given** arm D and its twin, **When** both are run on the 15-fold plan, **Then** the report
   gives the model's contribution as a paired delta with a CI, split into early and late halves
   (Principle II), and reports any shared-trade fraction that limits the paired test.

---

### User Story 4 - Couple or decouple the stop and the gain/loss targets (Priority: P3)

The researcher chooses whether the exit's take-profit and stop are set **independently**, as they
are today, or are **derived from the label's gain/loss targets** (coupled). Both modes run through
the same exit definition, so the comparison measures only the coupling. Invariant checks catch
the known defect classes, sign flips and scale errors, before any result is reported.

**Why this priority**: Coupling has caused real defects, and label/exit mismatch is a recorded
reason the label and the trade disagree. This story makes the investigation possible. Running the
investigation itself is follow-up work.

**Independent Test**: Run one trigger in both modes and confirm the invariants hold in each. When
the label targets match the independent values, the two modes must produce identical trade sets.

**Acceptance Scenarios**:

1. **Given** coupled mode with label targets equal to the independent exit values, **When** run,
   **Then** the trade set is identical to decoupled mode.
2. **Given** any mode, **When** the stop is computed for any trade, **Then** the effective stop is
   never looser than its documented bound and never has the wrong sign, and a violation stops the
   run with a clear error.
3. **Given** a configuration, **When** it is run, **Then** the run record states which mode was
   used and the resolved take-profit and stop values.

---

### Edge Cases

- **Trigger that never fires in a fold**: the fold is reported as a zero-trade fold, which is
  inconclusive, not a zero return (quality gate 6).
- **Trigger whose label has no causal part** (pure-outcome labels such as `forward_mae` or
  triple barrier): it maps to the `empty` trigger (every bar passes). The run is labelled as such,
  so it can't be mistaken for a selective trigger.
- **Non-causal trigger**: a trigger that reads forward data or outcome columns must be rejected
  before the run starts, never discovered afterwards.
- **Parameter dict defined as a fresh literal in a configuration**: inherited keys are silently
  dropped. Configurations must merge from the family default, and a configuration missing a key
  that the family default has must be detected.
- **Guard thresholds and trigger conditions overlap** (for example, `gbb` carries its own rvol
  condition): the run reports each condition's pass rate separately, so a double filter is
  visible.
- **Renamed or new SimpleTrigger class**: it must not train anything or look for a model, so the
  renamed-strategy self-training hazard can't occur.
- **Live dry-run isolation**: SimpleTrigger defaults must not change any parameter that the live
  dry-run strategies inherit.

## Requirements *(mandatory)*

### Functional Requirements

**Family and entry**

- **FR-001**: The system MUST provide a strategy family named SimpleTrigger whose entry decision is
  a registered trigger combined with the guard chain, with no model opinion involved.
- **FR-002**: A SimpleTrigger strategy MUST accept any trigger in the existing trigger registry by
  name, and MUST evaluate it through the registry, the same definition the leaderboard and the
  021 arms used.
- **FR-003**: Only causal triggers MUST be usable. Each trigger offered to the family MUST pass an
  end-to-end lookahead check, comparing output computed on truncated history against output
  computed on full history over the overlap (Principle V), and the result MUST be recorded per
  trigger.
- **FR-004**: The system MUST provide a documented mapping from each label method to its causal
  trigger, or to `empty` when the label has no causal part. Label methods with no mapping MUST be
  listed as unsupported, not silently treated as `empty`.
- **FR-005**: The family MUST NOT include regime detection, regime filters or regime-specific
  parameters. In particular, it MUST NOT use SimpleStrategies' bear/bull/sideways filter or its
  `exit_profit_only` behaviour.
- **FR-006**: A SimpleTrigger strategy MUST NOT load, train or look for a model, and MUST NOT create
  model artefacts.

**Framework parity**

- **FR-007**: For every framework component (guards, gain/loss targets, take-profit, max hold, exit
  signal, custom stop with ATR band, grace and volume confirmation, position sizing, liquidity cap
  and execution constraints), a SimpleTrigger strategy MUST behave identically to NNNC under the
  same settings. SimpleTrigger is a sibling family: NNNC is NOT restructured in this feature, and
  parity is enforced by the equivalence check (FR-008), not by shared inheritance.
- **FR-007a**: The feature's report MUST record the merge as a follow-up: restructure NNNC as
  SimpleTrigger plus a model gate once FR-008 has passed, so the two can't drift apart.
- **FR-008**: The system MUST provide an equivalence check: an NNNC arm whose model accepts every
  trigger bar MUST produce the same trade set as its SimpleTrigger twin.
- **FR-009**: The family's default configuration MUST mirror the guards-open research operating
  point (`NNNC_MLX_Open`, as used by `NNNC_A_Raw_Open` / arm D), so arm D's twin needs no
  overrides. The production `NNNC_MLX` operating point is not a default; it can be reached only by
  explicit overrides.

**Component evaluation**

- **FR-010**: Every framework component MUST have a documented neutral value that disables it or
  opens it fully, and setting it MUST change only that component.
- **FR-011**: Every run MUST record per-component liveness evidence, such as the trigger pass rate,
  each guard's pass rate, and the count of exits by reason. A component whose neutralised arm
  leaves the trade set unchanged MUST be reported INERT.
- **FR-012**: The system MUST produce a per-component attribution report: a paired delta against
  the full configuration per component, fold-level results, the early/late halves split, and the
  comparison's power or noise floor.

**Model contribution**

- **FR-013**: For any NNNC arm, the system MUST be able to build its SimpleTrigger twin and MUST
  verify that the twin's framework settings are identical before a comparison is reported.
- **FR-014**: The model-contribution report MUST give a paired delta with a CI, the early/late
  split, and the shared-trade fraction. When too few trades are shared for the paired test to
  have power, the report MUST say so and MUST NOT report a verdict.

**Coupled targets**

- **FR-015**: The exit MUST support two modes: decoupled (take-profit and stop set independently,
  today's behaviour and the default) and coupled (take-profit and stop derived from the label's
  gain/loss targets through one documented mapping).
- **FR-016**: Both modes MUST go through the single shared exit definition, so that no exit logic
  is duplicated per mode or per family.
- **FR-017**: The system MUST enforce stop invariants on every trade: the correct sign, never
  looser than the documented bound, and the grace-period rescale preserving direction. A
  violation MUST stop the run.

**Harness and hygiene**

- **FR-018**: SimpleTrigger configurations MUST run on the existing walk-forward harness and fold
  plans, and their per-fold results MUST be directly comparable with NNNC arm results.
- **FR-019**: Configurations MUST merge parameter dictionaries from the family default (per the
  constitution's merge rule), and the system MUST detect a configuration that drops an inherited
  key.
- **FR-020**: Family defaults MUST NOT alter any parameter inherited by the live dry-run.
- **FR-021**: The family MUST NOT contain pair-specific behaviour (Principle X). Trigger
  pair-invariance is already enforced by the registry and MUST stay in force.
- **FR-022**: The 021 trigger-only arms MUST stay reproducible after `TriggerOnly` is superseded by
  SimpleTrigger.

### Key Entities

- **Trigger**: a named, causal boolean condition over closed bars, held in the existing registry.
  It is the entry rule.
- **Label-to-trigger mapping**: for each label method, its causal trigger (or `empty`) and a
  supported or unsupported status.
- **Framework configuration**: the complete set of guard, target, exit, stop, sizing and execution
  settings. It has a fingerprint, so two arms can be proven to share it.
- **Component**: one switchable part of the framework, with its neutral value and its liveness
  measurement.
- **SimpleTrigger arm**: a trigger plus a framework configuration plus a fold plan. It produces
  per-fold P&L, trades and liveness records.
- **Twin pair**: an NNNC arm and its SimpleTrigger twin with a matching configuration fingerprint.
- **Exit mode**: coupled or decoupled, with the resolved take-profit and stop values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All six 021 trigger-only arms and 022's `wf_gbbr` reproduce their recorded trade sets
  exactly in 15 of 15 folds when run as SimpleTrigger configurations.
- **SC-002**: A new trigger-plus-framework arm can be defined by naming a trigger and overriding
  settings alone, with no new entry or exit logic, and run over 15 folds with no training step.
- **SC-003**: The model-free equivalence check (FR-008) produces identical trade sets for arm D's
  framework in 15 of 15 folds.
- **SC-004**: Arm D's model contribution is reported as a paired delta with a CI and an early/late
  split, or explicitly as "not resolvable at this power".
- **SC-005**: Every framework component has an attribution row, and no component is reported as
  having "no effect" without its liveness evidence.
- **SC-006**: Every causal trigger offered to the family has a recorded end-to-end lookahead result,
  and zero triggers that fail it are runnable.
- **SC-007**: Coupled mode with matched targets reproduces decoupled mode's trade set in 15 of 15
  folds, and the stop invariants report zero violations across all runs.
- **SC-008**: No parameter inherited by the live dry-run changes as a result of this feature (a
  before/after diff of the dry-run's resolved parameters is empty).

## Assumptions

- **Out of scope**: regime detection and regime-specific parameters, named as possible later
  work. If such work is done, it is gated by the trackability test (Principle VII) before any
  build.
- **Out of scope**: running the coupled-versus-decoupled investigation itself. This feature makes
  it possible and proves the modes are equivalent where they should be.
- **Out of scope**: any model change, and promoting any configuration to live.
- **Causal** means the trigger part of a label. The outcome part of every label looks forward by
  definition and is never an entry rule. Pure-outcome labels map to `empty`.
- `TriggerOnly` is superseded by SimpleTrigger rather than deleted, so 021's recorded arms remain
  reproducible.
- `SimpleStrategies` stays untouched. It is a different framework, and SimpleTrigger doesn't
  inherit from it.
- The existing walk-forward harness, fold plans, outcome table and shared exit definition are
  reused. Their noise floor (~3–4pp/yr, with sub-1pp deltas unresolvable) applies to every
  comparison here.
- Every result is conditional on the known binanceus 15m gap-fill (~half the bars for some pairs
  are zero-volume fills). The volume guard remains mandatory.
