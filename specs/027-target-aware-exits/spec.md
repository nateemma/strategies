# Feature Specification: Target-Aware Exits

**Feature Branch**: `027-target-aware-exits` (spec directory; work is committed to master)

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "Target-aware exits: couple the exit stack to the label's gain/loss targets (follow-up to 023 US4/FR-015, which built a coupled-mode SWITCH that maps label targets onto exit parameters but never ran the investigation). Today the stop is decoupled everywhere: stop = ATR-adaptive static stop (cap -4%, floor -6%) plus a first-hours grace window, take-profit 2.2% net, max hold 10 days, none of which knows the label's targets. The labels the causal triggers use (gbb, breakout_gbb) are defined as a >=2% gain within HORIZON=48 bars (12h at 15m) and carry no loss barrier, so 023's mapping cannot couple the stop for them, and its max-hold mapping refuses a sub-day horizon. The purpose of coupling is NOT to copy label numbers into the exit; it is to make the exit CONSISTENT with what a label-positive trade must survive to reach its target -- chiefly, the stop must not be so tight that trades never get the chance to reach the target gain. Recorded evidence that the label and exit disagree: the oracle ceiling, H=24 needs fast exits, and the grace window exists because early stops killed trades. Candidate couplings to evaluate (pre-declared, each a switch, all in the ONE shared exit definition so NNNC inherits them and the framework stays strategy-agnostic): (1) stop depth derived from the measured adverse excursion of label-positive paths before they reach the target, measured causally on early data only; (2) volatility units: stop and take-profit expressed consistently; (3) time: the label horizon as a time-stop or tightening point; (4) grace window length/depth tied to the horizon and the winners' MAE profile; (5) post-target handling: lock/trail once the target is reached. Constraints: constitution VII -- a cheap feasibility measurement on EXISTING trades and price data gates the build: (a) share of stop-loss exits that would have reached the take-profit, (b) winners' MAE vs the stop actually set, (c) time-to-target vs 12h and 10 days, per trigger family; only couplings the measurement supports get built; parameters fixed on f00-f07, judged on f08-f14 with the paired P&L and risk verdicts, PROVISIONAL under gate 2; reuse 023-026 machinery; no pair-specific code; stop invariants must hold in every mode. Scope is framework optimisation for NNNC, not portfolio/sleeve construction. Success = either a recorded negative, or the supported couplings built as switches and each judged by half against the current decoupled exit on the reference arms (momentum st_bgo_mx and gated reversion st_book_rev2)."

## Background *(PROVISIONAL under quality gate 2)*

**What a trigger's label promises.** The causal triggers used by the reference arms come from
labels that call a bar positive when price gains at least 2% within the next 48 bars (12 hours).
Neither label has a loss barrier, so the label itself says nothing about how deep a winner may dip
on the way.

**What the exit does today.** It doesn't use those targets at all:

| exit component | reversion family (`gbb`) | momentum family (`breakout_gbb`) |
|---|---|---|
| grace window (shared) | −27% stop for the first 6.5 h | same |
| stop after grace | ATR-adaptive, −4% to −6% | same |
| take-profit | 2.2% net | none |
| time exits | unclog rules; max hold 10 days | unclog at 24 h |

The stop after grace (−4% to −6%) is two to three times the 2% target. The max hold (10 days) is
20 times the label horizon (12 hours).

**Reference arms** (late half f08–f14, the half every decision is judged on):

| arm | per-fold sum | early | late |
|---|---|---|---|
| momentum `st_bgo_mx` | +25.19 | +22.68 | +2.50 |
| gated reversion `st_book_rev2` | +54.43 | +50.23 | +4.20 |

**Evidence that label and exit disagree** (memory):
- **Oracle ceiling:** a perfect-foresight model lost money because the exit stack did not manage
  trades toward the outcome the label defined.
- **H=24 needs fast exits:** the model predicts direction, not path; patient exits sit through
  retraces into the stop.
- **The grace window exists because early stops killed trades** that went on to win.
- **Earlier stop tuning found plateaus** (ATR band, grace level, multiplier): no gain from moving
  the stop's numbers in isolation. That is why this feature ties the stop to the target rather
  than sweeping it again.

**What 023 built and why it is not enough.** 023 added a coupled-mode switch that copies a
label's take-profit, horizon and loss barrier into the exit, and proved it reproduces the decoupled
exit when the numbers match. For the labels in use it cannot couple the stop (no loss barrier),
and it refuses the 12-hour horizon (max hold is in whole days). The investigation itself was left
out of scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cheap feasibility: is the exit inconsistent with the targets? (Priority: P1)

Before building anything, the researcher measures, on existing trades and price data only, how the
current exit treats trades relative to their target, separately for each family and each half:

- **(a) Stopped too early:** the share of stop-loss exits whose price later reached the
  take-profit level (the label's 2% target for momentum, which has no take-profit) within the label
  horizon, and within the max hold.
- **(b) Winners' dip vs the stop:** the distribution of the deepest drawdown that trades reaching
  the target took before reaching it, against the stop level that was actually in force.
- **(c) Time to target:** for trades that reached the target, how long it took, against the
  12-hour horizon and the max hold; and what happened to trades still open past the horizon
  without reaching it.
- **(d) After the target:** for trades that reached the target but did not exit there, how much of
  the gain was later given back.

Each candidate coupling has a pre-declared support rule over these measurements. Only supported
couplings go on to be built. If none is supported, the feature stops and the negative is recorded.

**Why this priority**: Constitution VII. Stop-only tuning has already plateaued several times in
this project, so an afternoon's measurement must be able to show whether there is anything for a
coupling to fix before any build.

**Independent Test**: A feasibility report with measurements (a)–(d) per family and per half, each
candidate coupling's support rule, the measured values, and a SUPPORTED/NOT SUPPORTED verdict.

**Acceptance Scenarios**:

1. **Given** the reference arms' existing trades, **When** the measurements are computed, **Then**
   they use only price bars at or after each trade's entry and the trade's own recorded stop, with
   no new backtests.
2. **Given** a candidate coupling, **When** it is scored, **Then** its verdict follows the
   pre-declared rule, and the rule must hold in BOTH halves (an effect in one half only is
   NOT SUPPORTED).
3. **Given** a measurement that would need a value chosen from data (a percentile, a time cut-off),
   **When** that value is set, **Then** only f00–f07 data is used.
4. **Given** no coupling is supported, **When** the report is written, **Then** the feature stops
   with the negative and its numbers.

---

### User Story 2 - Build the supported couplings as switches in the shared exit (Priority: P2)

For each supported coupling, the researcher adds a switch to the single shared exit definition. A
switch that is off leaves every existing run unchanged. A switch that is on derives the exit
behaviour from the strategy's declared targets (gain target, horizon) and, where the coupling needs
it, an early-fixed measured profile (for example a winners' drawdown percentile), never from
per-pair settings.

**Why this priority**: The exit is shared by SimpleTrigger and NNNC. Building the couplings there,
as switches, is what lets NNNC inherit them without strategy-specific code.

**Independent Test**: Each switch off reproduces the reference arms exactly; each switch on changes
trades (liveness); the stop invariants hold in every combination.

**Acceptance Scenarios**:

1. **Given** every new switch off, **When** a reference arm runs, **Then** its trades and its
   framework fingerprint are unchanged.
2. **Given** a switch on, **When** a reference arm runs, **Then** its run record states the switch,
   the declared targets and the derived exit values, and the number of trades whose exit the
   coupling changed is non-zero.
3. **Given** any switch combination, **When** the stop is computed for any trade, **Then** it has
   the correct sign, is never looser than its documented bound, and the grace rescale never moves
   the stop price down for a trade still above it; a violation stops the run.
4. **Given** a coupling that needs a time horizon shorter than a day, **When** it is configured,
   **Then** the horizon is honoured exactly in bars, not rounded to days.

---

### User Story 3 - Walk-forward judgement of each coupling (Priority: P3)

Each built coupling is run as a walk-forward arm on both reference arms. Any coupling parameter is
chosen on f00–f07. The late half f08–f14 judges it against the same arm with today's decoupled exit,
using the paired P&L verdict and the risk verdict.

**Why this priority**: Feasibility splits only choose what to test. Changing an exit changes which
later entries happen (023 Addendum 7, 026), so only a walk-forward arm shows the real effect.

**Independent Test**: Per coupling and per reference arm: early choice, late paired P&L verdict,
late risk verdict, liveness counts.

**Acceptance Scenarios**:

1. **Given** a coupling arm, **When** it is judged, **Then** it is adopted for that family only if
   its late-half P&L verdict is IMPROVED against the decoupled exit on the same arm.
2. **Given** a coupling adopted for one family, **When** it is reported, **Then** the other
   family's result is shown too, and a coupling that helps one family and hurts the other is
   adopted as a per-family setting, not shared.
3. **Given** several couplings adopted, **When** they are combined, **Then** the combination is
   itself run and judged, not assumed additive.

---

### User Story 4 - Report (Priority: P4)

One report gives the feasibility table, the switches built, and each coupling's walk-forward result
by half and family, with every figure labelled PROVISIONAL, and states plainly which exit NNNC
should inherit.

**Why this priority**: The decision this feeds is the default exit NNNC runs with.

**Independent Test**: The report exists with all rows, halves, verdicts and labels.

**Acceptance Scenarios**:

1. **Given** the feature stopped at US1, **When** the report is written, **Then** it shows each
   measurement and why no coupling was supported.
2. **Given** US3 ran, **When** the report is written, **Then** it states, per family, the adopted
   exit or that the decoupled exit stays.

### Edge Cases

- **Momentum has no take-profit.** Measurements (a), (c) and (d) use the label's 2% target as the
  reference level for momentum, and the report says so.
- **A trade that hits the target and the stop inside the same candle.** The order is unknown from
  candle data. Such trades are counted separately and excluded from (a) and (b), and the report
  gives their count.
- **Trades still open at a fold's end** are measured only up to the fold's last bar and flagged.
- **Liquidity-capped stakes** (most entries) do not affect price-path measurements, which are in
  percent of entry price.
- **A coupling that loosens the stop** (for example a deeper stop from winners' drawdown) must
  still respect a hard floor; the floor is part of the documented bound the invariants check.
- **A post-target lock that fires on the same candle as the entry** is not possible: the target
  is only known reached on a closed candle after entry.
- **Unclog and other time exits** already close trades early; the time coupling must state how it
  interacts with them (it replaces, precedes, or is skipped when an unclog rule fires first).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST evaluate a pre-declared set of candidate couplings:
  - (C1) **stop depth from the target path**: the stop after grace is no tighter than an early-fixed
    percentile of the drawdown label-positive paths take before reaching the target;
  - (C2) **volatility units**: stop and take-profit expressed consistently (both scaled by the same
    volatility measure, or neither);
  - (C3) **horizon time rule**: when the label horizon passes without the target being reached,
    the trade is closed or its stop tightened;
  - (C4) **grace tied to the target**: the grace window's length follows the horizon and its depth
    follows the winners' drawdown profile;
  - (C5) **post-target handling**: once the target is reached, the gain is protected (exit, or a
    stop moved to lock a share of it).
  The set is fixed before any measurement.
- **FR-002**: The feasibility measurement (US1) MUST use only existing trades of the two reference
  arms (`st_bgo_mx`, `st_book_rev2`) and price data, with no new backtests, and MUST report
  measurements (a)–(d) per family and per half.
- **FR-003**: Each candidate coupling MUST have a pre-declared support rule over the measurements,
  fixed before they are computed, and MUST be SUPPORTED only if the rule holds in both halves. A
  coupling that is NOT SUPPORTED MUST NOT be built.
- **FR-004**: Any value a coupling derives from data (a percentile level, a time cut-off, a lock
  share) MUST be fixed on f00–f07 only, from a small pre-declared grid or by a pre-declared rule
  (for example an early median), and MUST be the same for every pair (constitution X). A
  per-family value is allowed.
- **FR-005**: Every built coupling MUST live in the single shared exit definition used by both
  SimpleTrigger and NNNC, as a switch that is off by default. No exit logic may be duplicated per
  coupling, family or strategy.
- **FR-006**: With every new switch off, every recorded arm (023–026, arm D) MUST keep its framework
  fingerprint, and both reference arms MUST reproduce their recorded trades in every fold. Arm D,
  which needs the model to run, is covered by its unchanged fingerprint.
- **FR-007**: The declared targets a coupling uses (gain target, horizon in bars) MUST be stated by
  the strategy, not read implicitly from a labeler's defaults, and MUST appear in the run record
  with the derived exit values.
- **FR-008**: A horizon shorter than a day MUST be honoured exactly in bars.
- **FR-009**: The stop invariants of 023 FR-017 (correct sign, never looser than the documented
  bound, grace rescale preserving direction) MUST hold in every switch combination, with any new
  hard floor included in the documented bound; a violation MUST stop the run.
- **FR-010**: Every coupling arm MUST report liveness: the number of trades whose exit the coupling
  changed, per half (constitution IV).
- **FR-011**: Each coupling MUST be judged per reference arm on f08–f14 against the same arm with the
  decoupled exit, with the paired P&L verdict (adoption) and the risk verdict (reported). Adoption
  requires a late-half P&L verdict of IMPROVED.
- **FR-012**: Adopted couplings MUST be re-run together and judged as a combination before the
  combination is recommended.
- **FR-013**: The report MUST cover the feasibility table, the switches built, each coupling's
  walk-forward result by family and half, the combination if run, and the recommended NNNC exit.
  Every figure MUST be labelled PROVISIONAL (gate 2).

### Key Entities

- **Declared targets**: a strategy's gain target and horizon (in bars), the reference a coupling
  derives from.
- **Winners' path profile**: per family, early-fixed statistics of label-positive paths (drawdown
  before target, time to target).
- **Coupling switch**: a named, default-off exit behaviour with its derived values and liveness
  counts.
- **Feasibility record**: per family and half, measurements (a)–(d), and per coupling the support
  rule, values and verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feasibility report gives measurements (a)–(d) for both families in both halves, and
  a SUPPORTED/NOT SUPPORTED verdict for each of the five candidate couplings against its
  pre-declared rule.
- **SC-002**: With all new switches off, every recorded arm's fingerprint is unchanged (arm D
  included), and both reference arms reproduce their recorded trades in 15 of 15 folds.
- **SC-003**: Every built coupling, switched on, changes the exit of a non-zero number of trades in
  each half, and no stop-invariant violation occurs in any run.
- **SC-004**: For every built coupling and each reference arm, the report states the late-half
  paired P&L verdict and risk verdict against the decoupled exit; otherwise it records the stop at
  US1 with the reason.
- **SC-005**: The report names, per family, the exit NNNC should inherit, with the evidence behind
  it.

## Assumptions

- Data, folds, harness, causality gate and quality gates are unchanged from 023–026. There is no
  distant era, so every result is PROVISIONAL (gate 2), with persistence tested by halves only.
- The reference arms are momentum alone (`st_bgo_mx`) and gated reversion alone (`st_book_rev2`),
  each with today's decoupled exit as its control. They are judged separately; combining them is
  out of scope (memory: framework, not portfolio).
- The declared targets for both families are the label targets used today: ≥ 2% gain within 48
  bars (12 hours at 15m). Whether the momentum label's definition differs in detail is checked in
  the plan.
- Winners' path profiles come from label-positive bars or target-reaching trades in f00–f07 only.
- 023's coupled-mode switch stays as it is. This feature may extend or reuse its mapping, but
  must not change what its existing arms do.
- The model (NNNC) is not run in this feature; it inherits any adopted switch through the shared
  exit, and a later NNNC feature evaluates it with the model.
- Per-bucket splits (US1) choose what to test and are never quoted as the expected gain.
