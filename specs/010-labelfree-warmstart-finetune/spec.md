# Feature Specification: Label Warm-Start with P&L Fine-Tune

**Feature Branch**: `010-labelfree-warmstart-finetune`

**Created**: 2026-09-11

**Status**: Draft

**Input**: User description: "Feature: labelfree-warmstart-finetune. This is research/label-free-study brief 02; the feature must be specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented. Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md, briefs/01-direct-policy-optimisation.md (whose simulator and wrapper you will reuse — do not fork them) and briefs/02-label-warmstart-pnl-finetune.md. Implement arms A, B and C as configurations of the brief-1 trainer, with per-fold purge/embargo of the label horizon. Log each configuration to research/labelfree/trials.jsonl before running. Report against brief 1's best config and NNNC_MLX using regime/wf/compare_pooled.py, and write briefs/results/02-<date>.md using the protocol §6 template. Success criteria: the acceptance and kill gates of research/label-free-study/00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md."

## Overview

Brief 01 established that a policy trained purely on net-of-cost P&L does not convert signal
into P&L on the screening fold: every configuration that scored well did so by not trading,
and every configuration that genuinely traded lost 7% to 48% out of sample. That result
cannot distinguish two very different explanations — the profit objective is wrong, or
gradient descent from random initialisation cannot find a trading policy at this
signal-to-noise ratio.

This feature separates them. It asks whether the repository's existing lookahead labels,
which brief 01 discarded entirely, are a useful *prior* rather than a useless target: noisy
but informative about where tradeable moves begin. Three arms test that from different
directions — initialise from the trained classifier and fine-tune on P&L, train from scratch
on a blend of label and P&L objectives, or train on P&L with an auxiliary non-directional
target. All are measured against both the pure-label control and the pure-P&L control.

The decision this feature produces is directional for the whole study: whether labels should
be dropped or kept as a prior.

## Terminology

Carried forward from feature 009 so the two features do not drift apart.

| Canonical term | Means |
|---|---|
| **arbiter** | The production backtester. Final authority on P&L; the simulator exists for training speed |
| **label control** | The current production classifier at its production configuration (pure label) |
| **P&L control** | Brief 01's `utility-penalty-g0.5` — the best configuration that genuinely trades (mean exposure 0.130, −7.06% held-out), re-run under this feature's architecture per FR-004b. Brief 01's top-*ranked* configurations are excluded because they are degenerate: flat on 100% of bars, so any arm that trades beats them trivially |
| **screening fold** | The single fold used to narrow arms before any pooled run |
| **the floor** | The resolvable pooled walk-forward effect size, 3.35 percentage points per year |

## Clarifications

### Session 2026-09-11

- Q: The classifier's encoder and brief 01's policy are architecturally incompatible (classifier: conv1d(32,2,32) → LayerNorm → LSTM(32) → LayerNorm → bottleneck(32×32); policy: plain LSTM(64)). How should Arm A be made implementable? → A: Adopt the classifier's encoder architecture for ALL THREE arms, so the warm-start is literal and A/B/C differ only in initialisation and objective. Consequence: brief 02's policy differs from brief 01's, so the P&L control MUST be re-run under the new architecture or the comparison is confounded by architecture rather than by the label prior.
- Q: Brief 01 finished NOT RESOLVED and its four best-ranked configurations are degenerate (mean exposure 0.002-0.041, flat on 100% of bars). Which configuration is the P&L control? → A: `utility-penalty-g0.5`, the best arm that genuinely trades (mean exposure 0.130, -7.06% held-out). It is the only control against which "did labels help" is answerable. Its bar is low and the report must say so rather than presenting a win over it as a success.
- Q: Brief 02 names fold f07, but feature 009 measured f08 as carrying more control trades (71 vs 58) and f07 as the most profitable fold (+8.08% vs -0.14%). Which fold screens brief 02? → A: f08, consistent with feature 009 — more trades to resolve differences, not selected for being favourable, and directly comparable to brief 01's screen.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Warm-start from the trained classifier (Priority: P1)

A researcher initialises the policy's encoder from a classifier trained on the same fold's
training range, attaches a fresh exposure head, and fine-tunes on the net-of-cost profit
objective with a much smaller learning rate on the inherited encoder than on the new head.
This asks directly whether the classifier's learned representation is a better starting
point than random initialisation.

**Why this priority**: It is the cheapest arm that can overturn brief 01's ambiguity. If a
warm-started policy trades and profits where a randomly-initialised one did not, the failure
was optimisation, not objective. If it behaves identically, the objective is implicated.

**Independent Test**: Train the arm on the screening fold across at least five seeds and
compare its held-out return and mean exposure against brief 01's result on the same fold and
seeds. Delivers an answer about initialisation independent of the other two arms.

**Acceptance Scenarios**:

1. **Given** a classifier trained on the fold's own training range, **When** its encoder
   initialises the policy and the policy is fine-tuned on profit, **Then** held-out return
   and mean exposure are reported against the pure-P&L control on the same fold and seeds.
2. **Given** the inherited encoder, **When** fine-tuning runs, **Then** the encoder's
   learning rate is an order of magnitude below the head's, and both are recorded per trial.
3. **Given** a warm-started policy, **When** it is evaluated, **Then** the correlation
   between its exposure and the label it inherited is reported — near one means the
   fine-tune changed nothing, near zero means the prior was discarded, and both are findings.
4. **Given** the arm underperforms the pure-P&L control in-fold, **When** that is observed,
   **Then** it is recorded and the work proceeds to the blended arm rather than tuning this
   one, because a classifier tuned to the wrong label can legitimately hurt.

---

### User Story 2 - Blended label-and-profit objective (Priority: P2)

A researcher trains from random initialisation on a weighted sum of label cross-entropy and
the net-profit objective, sweeping the blend weight and testing two different labels the
repository has already shown behave very differently.

**Why this priority**: The supporting literature's central finding is that the best result
sits at a blend, not at either extreme. This arm is where that is tested, and it does not
depend on the classifier's architecture being reusable.

**Independent Test**: Sweep the blend weight across both labels on the screening fold at five
or more seeds; confirm every configuration is logged before running and that results are
reported as an interquartile mean across seeds.

**Acceptance Scenarios**:

1. **Given** a blend weight and a label, **When** the arm trains, **Then** both the label
   term and the profit term contribute to the reported loss, and the blend weight is recorded.
2. **Given** the blend sweep, **When** results are reported, **Then** no single blend weight
   is presented as "the best" — the sweep is a set of trials and enters the multiple-testing
   correction as such.
3. **Given** both labels, **When** they are compared, **Then** the two are reported
   separately rather than pooled, because they are known to behave differently.
4. **Given** any arm using a label, **When** folds are constructed, **Then** the label
   horizon is purged at each fold boundary and one horizon is embargoed after it, because the
   label term is forward-looking.

---

### User Story 3 - Auxiliary non-directional target (Priority: P3)

A researcher trains on the profit objective with an auxiliary head predicting forward
realised volatility — a target that carries no directional information at all.

**Why this priority**: It separates "labels help because they carry direction" from "any
auxiliary supervision regularises the objective". If the volatility head helps as much as the
directional label, the benefit was regularisation, not the label's content.

**Independent Test**: Train the arm at five or more seeds and compare against both the
pure-P&L control and the best blended arm.

**Acceptance Scenarios**:

1. **Given** the auxiliary head, **When** the arm trains, **Then** no directional label
   enters the objective at any point.
2. **Given** the arm's result, **When** it is compared to the blended arm, **Then** the
   report states whether directional content was necessary or whether any auxiliary target
   would have served.

---

### User Story 4 - Verdict against both controls (Priority: P4)

The researcher evaluates surviving arms through the pooled walk-forward harness against both
the label control and the P&L control, and reaches one of three verdicts: accept, not
resolved, or kill — recording in every case whether labels helped or hurt.

**Why this priority**: It converts arm results into the directional decision the study needs.
It cannot run before the arms produce a candidate that trades.

**Independent Test**: Run the surviving arms and both controls through the pooled harness and
confirm the report states the return delta against each control with its interval, measured
against the floor.

**Acceptance Scenarios**:

1. **Given** a surviving arm, **When** it is evaluated, **Then** it is compared against both
   controls, net of full costs, on the identical pair universe within each window.
2. **Given** no arm beats both controls by the floor, **When** the verdict is written, **Then**
   it reads "not resolved" and additionally records whether labels helped or hurt, since that
   decides the direction of the programme.
3. **Given** any arm whose exposure is degenerate, **When** results are reported, **Then** it
   is excluded from carry-forward and reported as inconclusive rather than as a return.

---

### Edge Cases

- **The classifier's representation cannot be transferred.** If the encoder cannot initialise
  the policy, the warm-start arm is not implementable as specified and the feature must say so
  rather than substituting a different arm silently.
- **The blended objective collapses to one term.** If one term's gradient dominates, the blend
  weight is nominal only; the realised contribution of each term must be measured, not assumed.
- **A warm-started policy reproduces the classifier exactly.** Exposure correlation near one
  means the fine-tune did nothing — a finding, not a failure, and it must be reported as such.
- **Every arm remains degenerate.** If no arm trades, the feature reports that labels did not
  rescue the objective, and the pooled runs are not spent.
- **The two labels disagree in direction.** One may help and the other hurt; they are reported
  separately and never averaged.
- **Purge or embargo is omitted on an arm that uses a label.** Any such result is void, because
  the label term is forward-looking.

## Requirements *(mandatory)*

### Functional Requirements

**Reuse and boundaries**

- **FR-001**: The feature MUST reuse brief 01's simulator, feature pipeline, trial log and
  deployable wrapper without forking them. Fixes MAY be contributed back; parallel copies MUST
  NOT be created.
- **FR-002**: The three arms MUST be configurations of the existing trainer, distinguished by
  initialisation and objective, so that differences between arms are attributable to those two
  choices rather than to separate implementations.
- **FR-003**: Any defect found in the reused components MUST be fixed in place, with the fix
  covered by a test that asserts on values rather than shapes.

**Arm A — warm-start**

- **FR-004**: The warm-start arm MUST initialise the policy's encoder from a classifier trained
  on the same fold's training range. The production model MUST NOT be reused across folds,
  because it was trained on data that later folds treat as held out.
- **FR-004a**: All three arms MUST use the classifier's encoder architecture, so that the
  warm-start is a literal inheritance of weights rather than a lossy re-encoding, and so that
  the three arms differ only in initialisation and objective. Brief 01's policy architecture is
  incompatible with those weights — measured, the classifier is a convolutional front-end,
  layer-normalised recurrent block and bottleneck, while brief 01's policy is a plain recurrent
  block of a different width.
- **FR-004b**: Because FR-004a changes the architecture relative to brief 01, the P&L control
  MUST be re-run under the new architecture. Comparing this feature's arms against brief 01's
  recorded numbers would confound the label prior with the architecture change, and the whole
  point of the comparison is to isolate the prior. The report MUST state both figures — the
  control as re-run here, and brief 01's original — so the size of the architecture effect is
  visible rather than absorbed.
- **FR-005**: The inherited encoder MUST train at a learning rate an order of magnitude below
  the newly-initialised head's, and both rates MUST be recorded per trial.
- **FR-006**: The arm MUST report the correlation between the policy's realised exposure and
  the label it was warm-started from.

**Arm B — blended objective**

- **FR-007**: The blended arm MUST optimise a weighted sum of label cross-entropy and the
  net-profit objective, with the weight swept across at least three values.
- **FR-008**: Both the production label and the alternative label MUST be tested, and results
  for the two MUST be reported separately.
- **FR-009**: The realised contribution of each loss term MUST be measured during training, so
  that a blend that has collapsed to one term is detected rather than assumed balanced.
- **FR-010**: The blend sweep MUST NOT be reported as identifying a best weight; every value is
  a trial and enters the multiple-testing correction.

**Arm C — auxiliary target**

- **FR-011**: The auxiliary arm MUST train on the profit objective plus a head predicting
  forward realised volatility, with no directional label entering the objective.
- **FR-012**: The report MUST state whether directional label content was necessary, by
  comparing this arm against the best blended arm.

**Leakage control**

- **FR-013**: Every arm whose objective contains a forward-looking term MUST purge the label
  horizon at each fold boundary and embargo one horizon after it.
- **FR-014**: The purge and embargo actually applied MUST be recorded per fold, and a run
  missing either MUST be void rather than caveated.

**Evaluation**

- **FR-015**: Every configuration MUST be appended to the trial log before it runs, with its
  result attached afterwards.
- **FR-016**: Each configuration MUST be trained at five or more seeds and reported as an
  interquartile mean with a bootstrap interval; a best-seed figure MUST NOT be reported.
- **FR-017**: Arms MUST be ranked on a held-out range, never on the range they trained on.
- **FR-018**: Both controls MUST appear in every comparison: the label control and the P&L
  control.
- **FR-019**: Every reported return MUST be accompanied by mean exposure, trades per year and
  mean holding time, so that a degenerate arm is visible rather than ranking on inaction.
- **FR-020**: An arm whose mean exposure falls below the participation floor MUST be reported
  as inconclusive and MUST NOT be carried forward.
- **FR-021**: A return delta below the floor MUST be reported as "not resolved", never as "no
  effect".
- **FR-022**: The verdict MUST record whether labels helped or hurt relative to the P&L
  control, regardless of whether any arm is accepted.

**Delivery**

- **FR-023**: All runs MUST execute sequentially.
- **FR-024**: Only files this feature creates or modifies MAY be staged.
- **FR-025**: A result is admissible only if the feature directory holds a specification, plan,
  task list and a clean cross-artefact analysis produced before implementation and again on
  completion.
- **FR-026**: The result report MUST follow the study's standard template and link the feature
  directory and both analysis reports.

### Key Entities

- **Arm**: One of the three experimental treatments, defined by its initialisation and its
  objective. Shares every other component with brief 01.
- **Trial record**: One evaluated configuration — arm, blend weight, label identity, seed,
  fold, learning rates, purge and embargo. Written before the result exists.
- **Warm-start source**: A classifier trained on a specific fold's training range, identified
  well enough that a fresh process can reproduce which weights were inherited.
- **Result report**: The study's per-brief write-up, including the directional finding about
  whether labels helped or hurt.

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Acceptance gates (protocol §5 — all six required)**

- **SC-001**: Pooled walk-forward annualised return is at least 3.35 percentage points per year
  above the control, with a bootstrap 95% interval excluding zero.
- **SC-002**: The configuration is positive in at least 10 of the 15 folds, and positive in each
  of the three evaluation windows when run single-window, with each control run on the identical
  pair universe within each window.
- **SC-003**: The configuration beats all mandatory baselines net of full costs.
- **SC-004**: Backtest-overfitting probability is at most 0.2 and the deflated risk-adjusted
  return is above zero, computed over the complete trial log including every arm and blend
  weight.
- **SC-005**: The return delta remains positive at 1.5× fees.
- **SC-006**: A separate process reproduces the accepted result from its saved configuration and
  seed, producing identical trades.

**Kill gates (protocol §5)**

- **SC-007**: If no configuration clears SC-001 within the budgeted work, the verdict is
  recorded as "not resolved" — never "no effect".
- **SC-008**: If any configuration's in-sample advantage exceeds its held-out advantage by more
  than 3.35 percentage points per year, it is recorded as overfitting and stopped.

**Brief-specific gates**

- **SC-009**: If no arm beats **both** the label control and the P&L control by the floor, the
  verdict is "not resolved" **and** the report records whether labels helped (arm above the P&L
  control) or hurt, because that decides the direction of the programme.
- **SC-010**: The blend sweep is never reported as identifying a best weight; the number of
  trials entering the overfitting correction equals the number of configurations run, including
  every blend weight and both labels.
- **SC-010a**: The P&L control is re-run under this feature's architecture, and the report
  states both that figure and brief 01's original, so the architecture effect is visible
  separately from the label-prior effect.
- **SC-011**: For the best arm, the correlation between its realised exposure and the label it
  was warm-started from is reported, with a correlation near one recorded as "the fine-tune
  changed nothing" and near zero as "the prior was discarded".
- **SC-012**: Every arm using a label reports the purge and embargo actually applied at each
  fold boundary; an arm missing either is void.

**Process gates**

- **SC-013**: Every configuration evaluated on any fold appears in the trial log, each written
  before its result was known.
- **SC-014**: Every reported return is an interquartile mean across at least five seeds with a
  bootstrap interval, and appears beside its mean exposure, trades per year and mean holding
  time.
- **SC-015**: No arm whose mean exposure falls below the participation floor is carried forward;
  such an arm is reported as inconclusive rather than as a return.
- **SC-016**: The feature directory holds a specification, plan, task list and a clean
  cross-artefact analysis produced before implementation and again on completion.

## Non-Goals

The following are explicitly out of scope. Each was excluded on recorded evidence.

- **Return-conditioned generative planners** — provably copy lucky trajectories in stochastic
  environments.
- **Training a policy inside a learned market simulator** — agents exploit simulator artefacts;
  no alpha-transfer record.
- **Recursive self-training on generated data** — collapses distribution tails, and the tails
  carry the risk.
- **Fine-tuning time-series foundation models** — zero-shot out-of-sample explanatory power is
  negative.
- **Online or continual fine-tuning during evaluation** — leak-free re-evaluations shrink the
  reported gains substantially.
- **Reinforcement learning** — a later brief, which must beat this family's best before it is
  justified.
- **Gradient-free search over the policy** — reserved for the non-differentiable pieces.
- **Full decision-focused optimisation layers** — long-only with caps is a closed-form
  projection.
- **New hand-crafted labels** — this feature uses the labels that already exist; inventing more
  is the route the repository's prior work already closed.
- **Per-pair code or per-pair parameter overrides** — fixes must be structural and uniform.
- **Forking brief 01's simulator or wrapper** — they are reused; defects are fixed in place.

## Assumptions

- **Windows, folds, statistics and the floor come from the protocol**, unchanged from feature
  009: the pooled walk-forward is primary, the resolvable floor is 3.35 percentage points per
  year, and single-window figures are smoke tests.
- **Screening fold is f08**, not the f07 named in the brief. Feature 009 measured f08 as
  carrying more control trades (71 vs 58) and f07 as the most *profitable* fold (+8.08% vs
  −0.14%); screening on the profitable fold biases selection toward configurations that only
  work in benign conditions. f08 also keeps this feature's screen directly comparable to brief
  01's, which matters because brief 01 is a control here.
- **The simulator is already calibrated.** Feature 009 reconciled it against the arbiter to
  zero difference per trade across 99 trades; this feature inherits that and does not re-derive
  it.
- **The simulator validates the accounting layer only** — fees, fill timing, liquidity cap,
  stake and portfolio sizing — and deliberately does not implement the deployment exit stack.
- **The participation floor and its mask normalisation are inherited** from feature 009, where
  a risk-adjusted objective was shown to have a degenerate optimum at zero exposure whenever
  nothing is profitable.
- **Coverage is measured as days of actual data inside the training range**, never by start
  date, because one pair begins early but carries a long delisting gap.
- **Pooled panels align on date, never by position.**
- **Runs are sequential** on a single machine; the models are small.
- **Only this feature's own files are staged**, committed directly to the strategies
  repository.
