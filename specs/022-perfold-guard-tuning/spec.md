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

## Clarifications

### Session 2026-10-09

**Asked:**

- Q: Given the guards are strongly super-additive, how should the reduced guard space be
  searched? → **A: a joint coarse grid over the three heaviest guards** — `atr_pct` × `adx` ×
  `rvol`, 4 levels each, 64 configurations. Bends Principle VIII's letter (a joint search) to
  keep its spirit: three dimensions and coarse levels are not the "many degrees of freedom" the
  principle warns about, and a sequential search is ruled out by super-additivity, which means
  individually-tuned values are not expected to compose. It also makes FR-005's "best of the
  actual candidates" a well-defined finite set — the 64 grid points.
- Q: Does the existing ablation finding that every live guard's effect flips sign across eras
  already discharge Principle VII's trackability gate? → **A: no, still run the gate.**
  Principle VII explicitly distinguishes an optimum that VARIES from one that is
  UNPREDICTABLE and says variation is "necessary but nowhere near sufficient". The ablation
  establishes variation; the gate tests predictability.

**Resolved from repository evidence, without asking:**

- Q: What is the minimum-trades-per-fold floor? → **A: 25 trades per fold (≥375 pooled).**
  `evaluation/README.md:18` records that pooling 15 folds into **378 trades** gives the
  **3.350 pp/yr** floor, and that 1–2pp/yr is unresolvable at any trade count. 378/15 ≈ 25.
  Arm B's 198 trades across 15 folds (one fold trading zero times) was roughly half the trade
  count the floor requires, which is why its +18.76pp cannot be read as resolved.
- Q: Which thresholds form the reduced search space? → **A: `atr_pct`, `adx`, `rvol`, and
  `guard_metric` is EXCLUDED as proven inert.** From `regime/GUARD_ABLATION.md`'s per-guard
  ablation: `atr_pct` −7.33pp, `adx` −5.95pp, `rvol` −3.42pp, `close_norm` −1.01pp,
  `bb_width` −0.17pp. **This corrects the premise this feature was proposed with** — the
  outline assumed `rvol` was dominant, which is true of its share of gate rejections but not
  of its return contribution; `atr_pct` and `adx` are "the heaviest lifters". And
  `guard_metric` is recorded as **completely inert**: byte-identical trades and returns on all
  three windows, zero extra trades, "the eighth recorded parameter that is accepted, applied
  and does nothing". Searching it would burn a dimension and trip FR-013 by construction.
- Q: What does "arm C comparable to arm D" mean numerically? → **A: the annualised difference
  is within the harness floor**, |C − D| < 3.35 pp/yr with a paired interval spanning zero.
  "Comparable" is a statement about non-resolvability, not about closeness.
- Q: Does arm E require retraining 15 models? → **A: no — it reuses the incumbent's existing
  per-fold artefacts.** All 15 are present on disk as
  `user_data/strategies/saved_data/NNNC_ARO_f{00..14}/NNNC_ARO_f{00..14}.safetensors`, named by
  the fold plan's `model_dir`. So the "one trainer at a time" constraint does not bind on arm
  E, and no model is fitted anywhere in this feature.

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
- **A grid corner wins because the guards are super-additive.** Individual removals cost −0.17
  to −7.33pp but removing all six costs −24.91pp, so the three searched guards interact. A
  configuration that wins jointly may be worse than either guard alone at its own best value;
  the per-guard marginal effect MUST NOT be inferred from the joint grid's winner.
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
- **FR-011**: The guard search space MUST be exactly `entry_atr_pct`, `entry_adx_threshold` and
  `entry_rvol_threshold`, searched as a joint coarse grid of 4 levels each (64 configurations).
  `entry_guard_threshold` MUST be excluded: it is recorded as completely inert (byte-identical
  trades across three windows). `entry_bb_width_threshold` and `entry_close_norm_threshold` are
  excluded as the two lightest contributors (−0.17pp and −1.01pp).
- **FR-012**: The search MUST reject any configuration admitting fewer than **25 trades in the
  fold being tuned**, derived from the harness's own power record: 378 pooled trades give the
  3.350 pp/yr floor, and 378/15 ≈ 25.
- **FR-013**: Every tuned parameter MUST be proven to have changed the trade list, measured
  from the hyperopt record and a per-fold trade-list hash — not from console output. A
  parameter that changes nothing MUST be reported as a defect.
- **FR-014**: Arms A, B and D MUST be reused from their existing records, not re-run.
- **FR-015**: All arm comparisons MUST use paired statistics on the symmetric difference of
  their trades.
- **FR-016**: Results MUST be reported per fold and additionally split into early and late
  halves of the 15, with an effect required to appear in both.
- **FR-017**: Any fold whose selected configuration admits fewer than 25 trades MUST be reported
  as unpowered and excluded from the pooled figure rather than contributing a 0.0pp observation.
- **FR-018**: Every P&L claim MUST come from the pooled harness in `evaluation/`. Classifier
  metrics may be reported as context and MUST NOT be an acceptance input.
- **FR-019**: The decision rule in Success Criteria MUST be recorded in this specification
  before arm C runs and MUST NOT be altered afterwards.
- **FR-020**: Arm E MUST NOT run unless arm C clears its decision rule, and MUST reuse the
  incumbent's existing per-fold model artefacts rather than retraining. No model is fitted
  anywhere in this feature.

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

Evaluated in the order below. **The first branch whose condition holds is the verdict**, so the
branches are total and mutually exclusive by construction — an earlier draft allowed two to hold
at once (a C that beats A *and* is comparable to D) and had no branch for a C that is worse than
A, nor for too few powered folds to decide.

**Branch 0 — POWER (evaluated first, before any comparison).** If fewer than **10 of the 15
folds are powered** (FR-017: ≥25 trades at the selected configuration), the verdict is
**`indeterminate`**. No return claim is made and no other branch is evaluated. A pooled figure
assembled from a third of the folds is not a result, and per-fold tuning that cannot reach the
trades floor in most folds has already answered the operating-point question.

**Branch 1 — HARM.** Arm C **below** arm A by more than the floor (annualised delta < −3.35
pp/yr) with a paired interval excluding zero → per-fold tuning is **actively harmful**, a
stronger negative than the null. It means the training window's optimum systematically
anti-predicts the test window, and it should not be reported as the same finding as Branch 3.

**Branch 2 — LEVER.** Arm C **above** arm A by more than the floor (> +3.35 pp/yr) with a
paired interval excluding zero → the guard chain is a genuine model-free lever and per-fold
tuning is the mechanism. The C-vs-D comparison is then reported as a *secondary* figure and
does not change this verdict: if C also lands within D's floor, the additional finding is that
the model is ornamental, but the primary verdict remains LEVER.

**Branch 3 — NULL.** Arm C within the floor of arm A (|delta| ≤ 3.35 pp/yr, or a paired
interval spanning zero) → arm B's +18.76pp was the tuning window leaking or a three-fold
accident; the guard chain is not a lever. A clean negative, and a complete result.

**Then, and only if Branch 2 held, arm E runs (FR-020).** Arm E above arm C by more than the
floor with a paired interval excluding zero → the model adds value on top of tuned guards and
model-side work is reopened. Otherwise the model is ornamental at this operating point.

## Assumptions

- The 15-fold plan, its date ranges and the pooled statistic (`wallet_profit_pct`) are reused
  from `evaluation/` unchanged, so arms remain comparable fold-for-fold with existing records.
- `gbb` is the trigger under test, because it is the incumbent's label and the only trigger with
  both a guards-open and a globally-guarded arm already measured.
- The minimum trades floor (25/fold) and the reduced guard space (`atr_pct`, `adx`, `rvol`) are
  fixed in this specification from existing power and ablation evidence, not deferred to
  planning.
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
