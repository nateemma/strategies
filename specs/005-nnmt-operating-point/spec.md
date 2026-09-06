# Feature Specification: NNMT Operating-Point Programme

**Feature Branch**: `005-nnmt-operating-point`

**Created**: 2026-09-06

**Status**: Draft

**Input**: User description: "Repeat the NNNC operating-point programme on NNMT."

## Context

A programme run on NNNC over 2026-09-05/06 changed what is known about where
this repository's edge comes from. None of it has been tested on the
multi-task family, and one of its changes has left NNMT in an inconsistent
state.

**What NNNC established** (all measured; `regime/GUARD_ABLATION.md`,
`CANDIDATE_STATUS.md`, `LABEL_TRIGGER_GATE.md`, `ARCHITECTURE_STUDY.md`):

- The entry guard chain selected ~99% of entries. Bypassing the model's entry
  opinion entirely cost +0.78pp on W1 and IMPROVED both in-sample windows.
- **Model headroom at the production gate was 0.68pp.** A perfect model was
  worth almost nothing more than the real one, so no model, GAN, label or
  architecture work could ever have paid there.
- Opening the quality guards and raising the confidence threshold moved
  selection to the model: ~25pp of volume-matched headroom, 4.5x the trades.
- The label gain threshold 0.007 was mis-specified against the operative stop;
  at the ceiling 0.02 turns W1 from −18.89% to +26.54%.
- Nine architecture variants were all worse, with the classifier metric pinned
  inside a 0.012 band — an information ceiling, not a model ceiling.

**Why NNMT, and why now.** Three of these have direct, untested consequences
for the multi-task family, and one is a defect introduced by the NNNC work:

1. **A live inconsistency.** NNMT does not override the label gain threshold,
   so it has just inherited the change from 0.007 to 0.02 — while every NNMT
   model on disk was trained at 0.007. Its next retrain silently changes target.
2. **NNMT did not inherit the operating point.** It overrides the entry guards
   with its own swept values (relative-volume 0.25 against NNNC's 2.0) and sets
   its confidence threshold explicitly, so the guards-open change passed it by.
   Whether it should have is the open question.
3. **The profit-taking level was lowered on another family's evidence.** NNMT
   has no parameter file and no override, so it inherits the change: a 31%
   reduction, untested on the multi-task family.

**Exactly two things have been inherited untested** — the label threshold and
the profit-taking level. Verified against NNMT's effective configuration rather
than assumed: every entry guard, the confidence threshold and the exit-side
relative-volume gate are unchanged, because NNMT overrides the guards and the
threshold with its own swept values, and the exit-side gate's new parameter
defaults to exactly the value that was previously hardcoded. Resolving the two
real changes is part of this feature, not a precondition for it.

A fourth item is worth noting but is not a change: NNMT's maximum hold is 21
days against NNNC's 10, because NNMT takes the shared default while NNNC pins
its own. That has always been so.

NNMT also has a genuine multi-task sell head, and its exit signal was once
measured as worth +10.4pp at an open gate — a result never followed up.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish whether NNMT's model matters at all (Priority: P1)

As the researcher, I want to know how much a perfect model would be worth on
NNMT at its current gate, so that I know whether any further work on this
family can pay before spending any of it.

**Why this priority**: On NNNC this single measurement invalidated years of
model-side work retrospectively. It requires no training, takes minutes, and
its answer determines whether the rest of the feature is worth executing at
all. Every other story is contingent on it.

**Independent Test**: A ceiling measurement exists for NNMT at its current
configuration, stated as the gap between a perfect entry signal and the real
model on the acceptance window, with trade counts.

**Acceptance Scenarios**:

1. **Given** NNMT at its current settings, **When** its entry signal is replaced
   by a perfect one, **Then** the difference in realised return on the
   acceptance window is reported with the trade count for each arm.
2. **Given** that measurement, **When** the model's entry opinion is instead
   bypassed entirely, **Then** the three-way comparison — no model, real model,
   perfect model — is reported, so the model's contribution is bounded from
   both sides.
3. **Given** a headroom of about a point or less, **When** the result is
   reported, **Then** the feature states explicitly that further model-side work
   on NNMT cannot pay at this gate, and the remaining stories are re-scoped
   rather than executed as written.

---

### User Story 2 - Resolve the two inherited changes (Priority: P1)

As the researcher, I want NNMT's label configuration and its trained models to
agree, and the profit-taking level it inherited on another family's evidence
judged on its own, so that the family is not sitting in a state where the next
retrain silently changes target and an untested exit level stands unexamined.

**Why this priority**: This is a defect, not an experiment, and it was
introduced by work done for another family. It should not be left pending
behind a research question. P1 alongside US1 because it is cheap and because
every trained arm in this feature depends on which label is correct.

**Independent Test**: Each inherited change is measured ALONE against the
pre-inheritance control on at least two seeds; one value is chosen for each with
a stated reason; and NNMT's configuration and models are made consistent with
both choices.

**Acceptance Scenarios**:

1. **Given** the two label settings, **When** each is screened at the ceiling,
   **Then** both results are reported and the choice between them is justified
   on realised return, not on economics or a classifier metric.
2. **Given** a chosen label, **When** the family is brought into line, **Then**
   configuration and models agree, and any model trained against the superseded
   label is either retrained or explicitly marked as stale.
3. **Given** that the chosen label differs from the one currently configured,
   **When** the change is made, **Then** it is scoped to NNMT and does not alter
   any other family.
4. **Given** the inherited profit-taking level, **When** it is measured alone
   against the control on the real model, **Then** its effect on realised return
   is reported per window across seeds and a value is chosen for NNMT.

---

### User Story 3 - Determine whether NNMT's guards bind (Priority: P2)

As the researcher, I want to know whether NNMT's entry guards or its model are
selecting the trades, so that I know which of them is worth tuning.

**Why this priority**: This is what made NNNC's headroom finding interpretable
rather than merely surprising. It needs no training. P2 because US1 already
bounds the model's contribution; this explains it.

**Independent Test**: A leave-one-out result exists over NNMT's guard chain
showing each guard's individual contribution, alongside an arm with the model
bypassed.

**Acceptance Scenarios**:

1. **Given** NNMT's guard chain, **When** each guard is neutralised in turn,
   **Then** each one's effect on realised return and trade count is reported.
2. **Given** those results, **When** a guard is found to change nothing at all,
   **Then** it is reported as inert rather than as beneficial.
3. **Given** the guards are found to bind as they did on NNNC, **When** the
   result is reported, **Then** it is stated whether the individual effects sum
   to the joint effect, so interaction is distinguished from addition.

---

### User Story 4 - Find NNMT's own operating point (Priority: P2)

As the researcher, I want NNMT's confidence threshold and guard settings chosen
on NNMT's own evidence, so that a setting tuned for a different family is not
inherited by accident.

**Why this priority**: NNNC's optimum is explicitly not transferable — its own
per-window optima disagreed with each other. NNMT's relative-volume setting is
already eight times looser, which is direct evidence the families differ. P2
because it is only worth doing if US1 finds headroom.

**Independent Test**: A threshold sweep exists across all three windows at an
opened gate, with the chosen value justified against per-window disagreement
rather than by picking the best single window.

**Acceptance Scenarios**:

1. **Given** an opened gate, **When** the confidence threshold is swept, **Then**
   results are reported for every window and the per-window optima are stated
   even where they disagree.
2. **Given** per-window optima that disagree, **When** a value is chosen,
   **Then** the choice is made by a stated decision rule rather than by the best
   window, and the value is treated as fixed rather than tuned per era.
3. **Given** a chosen operating point, **When** it is compared with NNMT's
   current one, **Then** the comparison is made across seeds, not from a single
   training run.

---

### User Story 5 - Correct the exit-side gate NNMT inherited (Priority: P2)

As the researcher, I want NNMT's exit-side relative-volume gate set on NNMT's
evidence, since it currently defaults to a value taken from another family that
is eight times tighter than NNMT's own entry setting.

**Why this priority**: It is a plausible long-standing defect, it is cheap to
test, and it interacts with the exit signal in US6. P2 because it is a
correction rather than a discovery.

**Independent Test**: The exit-side gate is swept and its effect on realised
return reported, with a value chosen or the default confirmed.

**Acceptance Scenarios**:

1. **Given** NNMT's current configuration, **When** the exit-side gate is swept
   including a fully-open setting, **Then** its effect on realised return and
   trade count is reported per window.
2. **Given** that the gate proves inert, **When** reported, **Then** it is
   recorded as inert rather than as confirmed-correct.

---

### User Story 6 - Test the multi-task exit signal (Priority: P3)

As the researcher, I want the exit signal evaluated on NNMT with its exit-side
gate correctly set, because NNMT has a real multi-task sell head and a prior
measurement of +10.4pp at an open gate that was never followed up.

**Why this priority**: The prior result is the strongest single unexplored
claim about this family. P3 because it depends on US5 — testing it against a
mis-set exit gate would repeat the error made on NNNC, where a naive test
showed almost nothing because the exit path had its own unopened guards.

**Independent Test**: The exit signal is evaluated both with the exit-side gate
at its chosen setting and fully open, so a null cannot be an artefact of the
gate.

**Acceptance Scenarios**:

1. **Given** the exit-side gate at its chosen value, **When** the exit signal is
   enabled, **Then** its effect on realised return is reported per window.
2. **Given** a near-zero effect, **When** it is observed, **Then** the exit path
   is checked for its own gating before the result is accepted as a null.
3. **Given** the prior +10.4pp claim, **When** the new measurement is reported,
   **Then** it states explicitly whether that claim reproduces.

---

### User Story 7 - Record the outcome and what it forecloses (Priority: P3)

As the researcher, I want NNMT's headroom and operating point recorded in a form
that governs later decisions, so that the finding is not rediscovered.

**Why this priority**: Documentation of a settled result. P3 because it consumes
no measurement budget and depends on everything above.

**Independent Test**: A written conclusion exists stating NNMT's headroom, the
chosen configuration, and what it forecloses for future model and augmentation
work on this family.

**Acceptance Scenarios**:

1. **Given** the measurements, **When** the conclusion is written, **Then** it
   states the headroom, the windows it was measured on, and its dependence on
   the chosen gate.
2. **Given** a future proposal for model or augmentation work on NNMT, **When**
   assessed against this conclusion, **Then** the conclusion is sufficient to
   accept or reject it without new measurement.

### Edge Cases

- **NNMT's headroom is negligible, as NNNC's was.** Then the feature's answer is
  that model work on this family cannot pay at this gate, and the later stories
  must be re-scoped rather than executed. This is a result, not a failure.
- **A comparison is run at a fixed confidence threshold across different label
  settings.** This is confounded: a different label shifts the confidence
  distribution and therefore the trade count. The operating point must be
  re-swept per configuration.
- **A result rests on a single training run.** Seed variance on the comparable
  NNNC configuration had a standard deviation several times larger than the
  effects being claimed. Single-seed arms are not evidence.
- **A strategy that is a pure re-parameterisation of another has no model of its
  own.** It will train one during evaluation, fitting on the evaluation window
  and returning different numbers with no error.
- **The third window is near-flat on NNMT at every gate previously tested**, so a
  conclusion driven by it needs more care here than it did on NNNC.
- **A ceiling improvement does not imply a realisable one.** On NNNC the model
  captured under 4% of a large ceiling gain.
- **A classifier metric improves while realised return falls.** This has now
  happened repeatedly in this repository, including monotonically.

## Requirements *(mandatory)*

### Functional Requirements

**Headroom (US1)**

- **FR-001**: The study MUST measure NNMT's model headroom at its current
  configuration as the difference in acceptance-window return between a perfect
  entry signal and the real model, reported with trade counts for both.
- **FR-002**: The study MUST also measure an arm with the model's entry opinion
  bypassed entirely, so the model's contribution is bounded from both sides.
- **FR-003**: The headroom measurement MUST require no model training.
- **FR-004**: If headroom is not materially larger than the study's resolution,
  the study MUST state that further model-side work on NNMT cannot pay at this
  gate, and MUST re-scope rather than execute the remaining work as written.
  Re-scoping means specifically: the guard attribution and the inherited-change
  resolution still run, because the first EXPLAINS the headroom result and the
  second is a defect fix that stands regardless of it; the operating-point and
  exit-side work does NOT run as written.

**Inherited changes (US2)**

- **FR-005**: The study MUST measure the ceiling at both the previously
  configured and the newly inherited label setting.
- **FR-006**: The study MUST choose one label setting and justify the choice on
  realised return, never on label economics or a classifier metric.
- **FR-007**: The study MUST leave NNMT's configuration and its trained models
  consistent with the chosen label, with any model trained against a superseded
  label either retrained or explicitly marked stale.
- **FR-008**: Any label change MUST be scoped to NNMT and MUST NOT alter the
  behaviour of another family. It MUST be implemented as an NNMT-level override,
  never by reverting the shared framework default that another family now uses.
- **FR-030**: The inherited profit-taking level MUST be measured ALONE against
  the pre-inheritance control, on the real model rather than at the ceiling, on
  at least two seeds, with a value chosen for NNMT on realised return.

**Guards (US3)**

- **FR-009**: The study MUST report each of NNMT's entry guards' individual
  contribution to realised return and trade count.
- **FR-010**: A guard that changes neither trades nor return MUST be reported as
  INERT rather than as beneficial.
- **FR-011**: The study MUST state whether the individual guard effects sum to
  the joint effect, so interaction is distinguished from addition.

**Operating point (US4)**

- **FR-012**: The confidence threshold MUST be swept across all three windows,
  and the per-window optima MUST be reported even where they disagree.
- **FR-013**: Where per-window optima disagree, the chosen value MUST follow a
  stated decision rule rather than the best single window, and MUST be treated as
  a fixed choice rather than tuned per era.
- **FR-014**: NNNC's operating point MUST NOT be assumed to transfer; any value
  adopted for NNMT MUST rest on NNMT measurements.
- **FR-031**: The operating point is the confidence threshold AND the guard
  settings together. The gate at which the threshold is swept MUST itself be
  chosen from the guard attribution by a stated rule, and reported, rather than
  left implicit.

**Exit side (US5, US6)**

- **FR-015**: The exit-side relative-volume gate MUST be swept, including a
  fully-open setting, with its effect reported per window.
- **FR-016**: The exit signal MUST be evaluated both at the chosen exit-gate
  setting and with that gate fully open, so a null cannot be an artefact of the
  gate.
- **FR-017**: The study MUST state explicitly whether the prior +10.4pp exit
  signal claim reproduces.

**Evidence standards (cross-cutting)**

- **FR-018**: Acceptance MUST be judged on realised return on the out-of-sample
  window only. The other two windows MUST be reported and labelled as in-sample
  and excluded from acceptance.
- **FR-019**: No classifier metric may be an acceptance input. Such metrics MUST
  be recorded as context only.
- **FR-020**: Any result used to accept or reject a configuration MUST be
  measured across at least two training seeds, with the spread reported. A
  single-seed result MUST NOT be treated as evidence.
- **FR-021**: Configurations differing in label MUST NOT be compared at a fixed
  confidence threshold; the operating point MUST be re-swept per configuration.
- **FR-022**: Every arm MUST carry its own parameter file so that none silently
  runs inherited defaults.
- **FR-023**: Any strategy that is a pure re-parameterisation of another MUST be
  made to load the other's model rather than training its own during evaluation.
- **FR-024**: Every ceiling figure MUST be labelled as a lookahead bound and MUST
  NOT be reported as an achievable return; no ceiling configuration may be
  promoted toward live trading.
- **FR-025**: Arms MUST be executed sequentially.
- **FR-026**: No shared framework mechanism may be left modified; the live dry
  run depends on it.
- **FR-027**: Any arm record used as evidence MUST be confirmed to come from a
  successful run.

**Conclusion (US7)**

- **FR-028**: The study MUST record NNMT's headroom, the chosen configuration,
  and what the result forecloses for future model and augmentation work on this
  family.
- **FR-029**: If a configuration change is recommended, the study MUST report its
  capacity behaviour under a larger book, since the comparable NNNC
  configuration retained materially less return as the book grew.

### Key Entities

- **Ceiling arm**: an evaluation whose entry signal is the training label
  itself. A diagnostic bound and screening instrument only; never deployable.
- **Bypass arm**: an evaluation with the model's entry opinion removed,
  establishing what the guards alone achieve.
- **Guard contribution**: for one guard, its individual effect on realised
  return and trade count when neutralised alone.
- **Operating point**: the confidence threshold and guard settings together,
  chosen on this family's own evidence.
- **Seed replicate**: the same configuration trained at a different seed, used
  to establish whether a difference is real.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: NNMT's model headroom at its current configuration is stated as a
  single figure with the trade counts and windows it was measured on.
- **SC-002**: The three-way comparison — no model, real model, perfect model —
  is reported for NNMT on all three windows.
- **SC-003**: Both inherited changes are resolved: a value is chosen for each on
  realised return across seeds, and NNMT's configuration and models agree with
  those choices.
- **SC-004**: Every one of NNMT's entry guards has its individual contribution
  reported, and any inert guard is identified as such.
- **SC-005**: A confidence threshold is chosen for NNMT by a stated decision
  rule, with per-window optima reported including any disagreement.
- **SC-006**: The exit-side gate and the exit signal are each reported with a
  measurement that cannot be confounded by the other.
- **SC-007**: Every accepted or rejected configuration is supported by at least
  two seeds, with the spread reported.
- **SC-008**: No configuration is accepted on a classifier metric, and every
  ceiling figure is labelled as a lookahead bound.
- **SC-009**: No shared framework mechanism is left modified at the end of the
  feature.
- **SC-010**: The written conclusion is sufficient to accept or reject a future
  model or augmentation proposal on NNMT without new measurement.

## Assumptions

- NNMT's existing trained models correspond to the previously configured label,
  and the newly inherited setting affects only future training. Inference by a
  model already on disk is unaffected by a label change.
- The multi-task nature of NNMT changes what the model can express but not the
  form of the measurements: headroom, guard contribution and operating point are
  all defined the same way as on NNNC.
- Seed variation is the dominant source of run-to-run noise, and two seeds are
  sufficient to detect an effect several times larger than the spread. Effects
  comparable to the spread would need more. The concrete anchor: on the
  comparable NNNC configuration a headline +9.97pp proved to be seed luck — the
  mean across three seeds was +3.07pp against a standard deviation of 6.38pp,
  and the arm lost the acceptance window on 3 of 3 seeds.
- The third window's near-flat behaviour on NNMT is a property of the family
  rather than an error, so a small effect there is weaker evidence than the same
  effect on another window.
- Fills are already realistic, because entry size is capped in proportion to
  available volume rather than rejected. Liquidity risk therefore appears as
  reduced position size, not as impossible trades.

## Non-Goals

- Auxiliary head weighting. Measured null: head weights moved entries less than
  a random seed even at an open gate. Closed.
- Task filters. Measured harmful. Closed.
- The COCONUT interventions. Both halves were closed on NNNC at an operating
  point where they could have registered. They are re-openable on NNMT ONLY if
  its headroom proves materially larger, and the study must say so explicitly if
  that is the case.
- Augmentation and generative enhancement. A separate follow-on.
- New data sources.
- Per-pair special-casing.
