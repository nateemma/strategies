# Feature Specification: Per-Fold Guard Tuning

**Feature Branch**: `022-perfold-guard-tuning`

**Created**: 2026-10-09

**Status**: Draft

**Input**: Implements `specs/021-label-exit-coordination/followup-perfold-guard-tuning.md`, which
carries the evidence for every design choice here.

## Context

Feature 021 established three things about the `gbb` label family, all on the 15-fold pooled
walk-forward harness (3.68 year-equivalent, one fixed exit, `exit_config_hash 5a18305d1fafa83e`):

| measurement | result |
|---|---|
| trigger alone, guards open, no model | **+2.64pp**, 7/15 folds — flat, below the 3–4pp/yr floor |
| learnability within the trigger | conditional MCC **+0.0504** vs +0.5927 unconditional; 13 of 13 triggers collapse ~90% |
| tail rankability within the trigger | top-1% precision lift **inverts** between windows (ρ = −0.648, p = 0.017) |
| **incumbent**: trigger + 24 indicators + **per-fold** model threshold, guards open | **+39.98pp, 15/15 folds** |
| one-off arm: **globally** tuned guards, no model | **+18.76pp**, 12/15, per-trade EV +0.014671 (95× the unguarded arm) |

The first three say there is nothing in this label to exploit. The last two say something is
being exploited. They reconcile only if the signal is real but **regime-dependent**, and so
available only to a mechanism that **re-tunes**. This feature tests that — and tests it in the
order the constitution requires, which is not the order the outline proposed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Can the motivating number be trusted? (Priority: P1)

Before any compute is spent, establish whether the **+18.76pp** that motivates this study is
an out-of-sample result or an artefact. It came from BaseStrategy's pre-2026-09-06 tuned guard
values, whose hyperopt window is unknown. If that window overlaps the 15 evaluation folds, the
number is in-sample and the premise of the whole study changes.

**Why this priority**: it is the cheapest task and it can invalidate everything downstream.
Feature 021 twice trusted a provenance field that asserted rather than measured — a hardcoded
`volume_filter_active: True`, and a `guards` field that recorded a CLI flag rather than the
effective value — and one of those was cited as evidence before anyone checked it.

**Independent Test**: produces a written provenance verdict with the evidence (commit dates,
any `.fthypt` records, the fold date ranges) and a yes/no on overlap, without running any arm.

**Acceptance Scenarios**:

1. **Given** the tuned guard values in `Framework/BaseStrategy.py`, **When** their origin is
   traced through git history and hyperopt records, **Then** the tuning window's date range is
   either established or explicitly reported as unrecoverable.
2. **Given** that date range and the 15 fold test ranges, **When** they are compared, **Then**
   the verdict records whether any evaluation fold falls inside the tuning window.
3. **Given** the incumbent arm, **When** its configuration is read from the strategy class and
   params file rather than from prose, **Then** the record confirms guards-open and a per-fold
   model threshold, or contradicts the premise and says so.

---

### User Story 2 — Is the guard optimum TRACKABLE, or merely variable? (Priority: P1)

Constitution Principle VII, non-negotiable: for any proposed adaptive or regime-conditional
parameter, build the rolling-optimum series and test whether a **trailing-best forecast beats a
CONSTANT**. A parameter whose optimum merely *varies* is not a candidate for adaptivity —
variation is necessary and nowhere near sufficient; predictability is the missing requirement.

**Why this priority**: this is the gate that can kill the feature in an afternoon, and the
prior is strongly against it. **Four adaptive schemes have already lost to static values in
this project**, and feature 021's own tail signal inverted between adjacent windows. Running
fifteen hyperopts before this check would be building adaptivity without establishing
trackability, which the constitution forbids.

**Independent Test**: for each fold, compute the best guard configuration on its training
window and the best on its test window, then compare the realised outcome of the
training-chosen configuration against the best single CONSTANT configuration across all folds.
Reports regret, not accuracy, because choosing between configurations is a decision and not an
estimate.

**Acceptance Scenarios**:

1. **Given** the guard configuration space and the 15 folds, **When** the per-fold optimum is
   computed on each training window, **Then** the series of chosen configurations is recorded
   with how often the winner changes.
2. **Given** that series, **When** the training-chosen configuration is scored on each fold's
   test window and compared to the best constant configuration, **Then** the result states
   whether trailing-best **beats** the constant, with a paired interval.
3. **Given** a result where trailing-best does NOT beat the constant, **When** the gate is
   evaluated, **Then** User Story 3 MUST NOT proceed and the feature terminates with that as
   its finding.

---

### User Story 3 — Does per-fold guard tuning trade? (Priority: P2)

Run arm **C**: the trigger plus quality guards tuned per fold on that fold's own training
window, with **no model**, over the same 15 folds and the same fixed exit as the existing arms.

**Why this priority**: it is the feature's headline question, but it is gated on US2. Without
trackability it measures noise at 15× the cost.

**Independent Test**: produces a pooled wallet return, a fold-positive count and a **paired**
confidence interval against arm A, with every fold's trade count recorded.

**Acceptance Scenarios**:

1. **Given** fold *i*, **When** guards are tuned, **Then** only that fold's training window is
   used and its test window is evaluated exactly once.
2. **Given** a candidate configuration admitting fewer than the minimum trades per fold,
   **When** the search evaluates it, **Then** it is rejected rather than accepted as a
   high-scoring sparse result.
3. **Given** arm C complete, **When** it is compared to arms A, B and D, **Then** all four
   share an identical `exit_config_hash`.
4. **Given** each tuned parameter, **When** its effect is checked, **Then** it is proven to
   have changed the trade list, or reported as INERT.

---

### User Story 4 — Does the model add anything on top? (Priority: P3)

Run arm **E**: per-fold tuned guards *and* a per-fold model threshold. Runs **only if arm C
succeeds**.

**Why this priority**: the only comparison in this feature that could rehabilitate model-side
work, but meaningless if C fails.

**Independent Test**: paired comparison of E against C on the same folds.

**Acceptance Scenarios**:

1. **Given** arm C did not clear its decision rule, **When** US4 is considered, **Then** it is
   not run.
2. **Given** both arms complete, **When** compared, **Then** the difference is reported with a
   paired interval on the symmetric difference of their trades.

---

### Edge Cases

- **A fold whose tuning window yields no configuration above the trades floor** — reported as
  unpowered for that fold, never silently substituted with guards-open or with zero.
- **A fold that trades zero times.** Arm B had one. Such a fold contributes no return
  observation and must not be averaged in as a 0.0pp result.
- **The tuning window's optimum is a corner of the search space** — recorded, because a
  parameter pinned at its bound is evidence the space is mis-centred, not evidence of an edge.
- **Two configurations producing an identical trade list** — the parameter separating them is
  inert, and the search is exploring a dimension that does not exist.
- **Provenance unrecoverable in US1** — arm B is then marked *untrustworthy* permanently rather
  than assumed clean, and C vs B loses its interpretation while C vs A keeps its own.
- **The search space cannot be reduced defensibly** — if no 2–3 guards can be justified as the
  binding ones, per-fold tuning may not be estimable on a ~90-day window at all. Discovering
  this during planning is a legitimate outcome and terminates the feature early.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The provenance of the pre-2026-09-06 tuned guard values MUST be established, or
  explicitly reported as unrecoverable, and recorded **before any arm is run**.
- **FR-002**: The provenance record MUST state whether the tuning window overlaps any of the 15
  evaluation fold test ranges, and arm B's +18.76pp MUST carry that verdict wherever it is
  quoted.
- **FR-003**: The incumbent's configuration MUST be verified from the strategy class and params
  file, not from prose or from any recorded provenance field.
- **FR-004**: A trackability gate MUST be run before arm C: the rolling per-fold optimum series,
  and a test of whether a trailing-best forecast beats the best single constant configuration.
- **FR-005**: The trackability gate MUST report **regret** against the best of the actual
  available configurations — never against a third value that was never a candidate.
- **FR-006**: If the trackability gate does not show trailing-best beating a constant, arm C
  MUST NOT be run, and that MUST be reported as the feature's finding rather than as a failure
  to complete.
- **FR-007**: Guard tuning MUST use only the fold's training window; each fold's test window
  MUST be evaluated exactly once.
- **FR-008**: `entry_enable_guards` MUST remain `True` in every arm. It also gates the
  `volume > 0` data-sanity check, and 34.26% of bars in this dataset have `volume <= 0`. Guards
  are opened by neutralising thresholds, never by this flag.
- **FR-009**: The exit policy MUST be held fixed and identical across all arms, proven by
  `exit_config_hash` equality. No task may change an exit parameter.
- **FR-010**: Only the guard space may be tuned. No model parameter, no exit parameter, no
  entry-timing parameter may enter the search.
- **FR-011**: The guard search space MUST be reduced to the thresholds with existing evidence
  of binding, and the reduction MUST be justified in writing before the search runs.
- **FR-012**: The search MUST reject any configuration admitting fewer than a pre-registered
  minimum number of trades per fold.
- **FR-013**: Every tuned parameter MUST be proven to have changed the trade list, measured
  from the hyperopt record and a per-fold trade-list hash — not from console output. A
  parameter that changes nothing MUST be reported as a defect.
- **FR-014**: Arms A, B and D MUST be reused from their existing records, not re-run.
- **FR-015**: All arm comparisons MUST use paired statistics on the symmetric difference of
  their trades.
- **FR-016**: Results MUST be reported per fold and additionally split into early and late
  halves of the 15, with an effect required to appear in both.
- **FR-017**: Any fold below the trades floor MUST be reported as unpowered and excluded from
  the pooled figure rather than contributing zero.
- **FR-018**: Every P&L claim MUST come from the pooled harness in `evaluation/`. Classifier
  metrics may be reported as context and MUST NOT be an acceptance input.
- **FR-019**: The decision rule in Success Criteria MUST be recorded in this specification
  before arm C runs and MUST NOT be altered afterwards.
- **FR-020**: Arm E MUST NOT run unless arm C clears its decision rule.

### Key Entities

- **Guard configuration**: a set of quality-threshold values applied at entry, with
  `entry_enable_guards` true. The object being tuned.
- **Fold**: one anchored walk-forward split with a training window and a test window; 15 exist
  and their date ranges are fixed by the shared fold plan.
- **Arm**: one configuration run across all 15 folds, carrying a pooled wallet return, a
  per-fold return series, a trade count per fold, and an exit-configuration hash.
- **Provenance verdict**: the written record of where a parameter value came from and whether
  its origin overlaps the data it is evaluated on.
- **Trackability result**: the rolling-optimum series plus the regret of trailing-best against
  the best constant.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The provenance verdict for the tuned guard values exists and is dated before the
  first arm-C fold runs.
- **SC-002**: The trackability gate produces a regret figure for trailing-best against the best
  constant, with a paired interval, across all 15 folds.
- **SC-003**: Arm C reports a pooled wallet return, a fold-positive count out of 15, and a
  paired confidence interval against arm A.
- **SC-004**: Every arm in the comparison shares one identical exit-configuration hash.
- **SC-005**: Every tuned parameter is shown to change the trade list in at least one fold, or
  is reported as inert.
- **SC-006**: No fold below the trades floor contributes a return observation to any pooled
  figure.
- **SC-007**: Each reported effect is accompanied by its early-half and late-half values.
- **SC-008**: The trackability gate completes within one working session, so it can kill the
  feature before the expensive phase begins.

### Pre-registered decision rule *(FR-019 — fixed before arm C runs)*

- **Arm C beats arm A by more than 3–4pp/yr with a paired interval excluding zero** → the guard
  chain is a genuine model-free lever and per-fold tuning is the mechanism.
- **Arm C within the floor of arm A** → arm B's +18.76pp was the tuning window leaking or a
  three-fold accident; the guard chain is not a lever. A clean negative, and a complete result.
- **Arm C comparable to arm D** → the model is ornamental; effort moves to gate and mechanism
  work.
- **Arm E materially exceeds arm C** → the model adds value on top of tuned guards, and
  model-side work is reopened.

## Assumptions

- The 15-fold plan, its date ranges and the pooled statistic (`wallet_profit_pct`) are reused
  from `evaluation/` unchanged, so arms remain comparable fold-for-fold with existing records.
- `gbb` is the trigger under test, because it is the incumbent's label and the only trigger with
  both a guards-open and a globally-guarded arm already measured.
- The minimum trades floor and the reduced guard space are set during planning, from existing
  ablation evidence, and recorded before the search runs.
- Arm B's result stands as measured; this feature does not re-run it, only qualifies it with the
  provenance verdict.
- The live dry-run (`MomentumRegimeBasket15mFast`) must observe no behaviour change; shared
  runtime code it inherits is not modified by this feature.

## Out of Scope

Each exclusion carries its evidence, so none is revisited by default.

- **Changing the exit policy** — calibrated by a prior take-profit ladder and stop sweeps.
- **Expanding the universe** — the edge is structurally high-volatility-alt; the universe has
  no lever.
- **Adding or changing features** — 24 indicators against 8 raw OHLCV is −1.23pp/yr with a CI
  spanning zero.
- **Re-tuning the label** — feature 021 found 13 of 13 triggers collapse identically, so label
  work is excluded by evidence rather than merely unexplored.
- **Promoting any configuration to production.** This feature produces evidence.
