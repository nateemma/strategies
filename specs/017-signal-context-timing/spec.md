# Feature Specification: signal-context-timing

**Feature Branch**: `017-signal-context-timing`

**Created**: 2026-09-15

**Status**: Draft — clarifications pending

**Input**: Study whether NNNC_MLX entry-signal QUALITY is predictable from signal CONTEXT, and
if so whether a context filter is an adoptable gate.

## Context

Two measurements taken on 2026-09-15 (`research/posadj/RESULTS.md`,
`docs/WALKFORWARD_TODO.md` item 3) motivate this feature.

**1. Repeat signals mark worse trades.** Trades that received a second `enter_long` while open
did materially worse than those that did not:

| | n | win rate | mean `profit_ratio` | trailing-stop exits |
|---|---|---|---|---|
| no repeat signal | 345 | **0.783** | +0.01444 | 21.7% |
| received a repeat signal | 96 | **0.635** | −0.00301 | **36.5%** |

Fisher exact **p = 0.0049**, Mann-Whitney **p = 0.0079**.

**2. Most signals are never observed.** freqtrade admits an entry only when the pair has no open
position, so **~43% of generated signals are silently discarded** (≥54 of ~125 on fold f08) —
and not even counted, because `_collate_rejected` fires only on the `max_open_trades` path.

### The structural correction this feature must make

The motivating framing — "filter the bad signals" — **does not survive contact with the
mechanism, and the spec says so up front rather than discovering it in implementation.**

A repeat signal arrives *while a position is already open*. freqtrade has **already discarded
it**. A filter on such a signal is a **no-op**: it removes something that was never going to
trade. So finding 1, read as an entry filter, is unactionable.

Read correctly, finding 1 is not about the signal at all — it is **information about the open
position**. The repeat signal arrives *before* that trade's outcome is known, so it is a genuine
forward-looking predictor of a trade already in flight. Its actionable home is the **exit
side**: a de-risk or stop-tightening trigger, not an entry gate.

Signal context therefore splits in two, and the two halves have different actionable homes:

| context | applies to | actionable as |
|---|---|---|
| a position is already open on the pair | repeat signals | **exit-side** information |
| bars since the previous signal, previous signal's outcome, drawdown from recent high, burst position | **first** signals | entry gate |

### The confound that must be cleared first

"Received a repeat signal" may be little more than a restatement of **"is currently
underwater"**. If the model re-fires mainly after price has fallen, then the repeat signal is a
proxy for unrealised loss, and losing trades ending badly is close to tautological.

**Nothing downstream is evidence until that is excluded.** The effect must survive
conditioning on unrealised P&L at the moment the repeat signal arrives.

### Prior nulls this must be distinguished from

- `feedback_entry_timing_noise_regime_dominates` — entry age p = 0.49. Measured **entry age
  within a held basket position on the Momentum family**, not signal-quality conditioning on
  an ML family.
- `feedback_entry_band_in_noise` — a fill-price band on CppiSkim, buried in walk-forward noise.

Both are cautions, not blockers: this feature starts from an effect already measured at
p = 0.005, rather than searching for one.

### The counter-hypothesis that most likely kills this

`project_guard_ablation_model_adds_nothing` found that the **guard chain, not the model**,
selects entries — bypassing the model costs little. A context filter may simply be another
guard in disguise, which is why Principle IX's overlap check runs before any performance number.

## Clarifications

### Session 2026-09-15

- Q: Does scope include the exit side, or stay entry-only? → A: **Both.** Repeat signals are
  studied as exit-side information (a de-risk or stop-tightening trigger) because an entry
  filter on them is a no-op; first-signal context is studied as an entry gate. Entry-only would
  discard the one effect this feature has actual evidence for.
- Q: If the confound gate (FR-001) kills the repeat-signal effect, does the feature stop? →
  A: **No — the first-signal sweep continues, explicitly labelled EXPLORATORY.** It rests on
  different data and a different mechanism, so a tautology in the repeat-signal finding says
  nothing about it. But it will have lost its motivating prior, which makes the pre-declared
  multiple-testing correction load-bearing rather than a formality.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The effect is not just "losers lose" (Priority: P1) 🎯 MVP

A researcher establishes whether a repeat signal predicts a worse outcome **beyond** what the
position's current unrealised P&L already predicts.

**Why this priority**: it is the cheapest gate and the one that can kill the feature outright.
If the effect vanishes under conditioning, the motivating measurement is a tautology and
everything downstream is void.

**Acceptance Scenarios**:

1. **Given** trades that received a repeat signal, **When** outcomes are compared against
   trades matched on unrealised P&L at the same holding point, **Then** the difference is
   reported with a significance test.
2. **Given** the conditioned comparison, **When** the effect does not survive, **Then** the
   feature records "the repeat signal is a proxy for being underwater" and stops.
3. **Given** any comparison, **When** it is made, **Then** the matched control is drawn from
   trades alive at the same holding point — never from all trades, which would compare
   different holding durations.

---

### User Story 2 - The repeat signal as exit-side information (Priority: P2)

A researcher tests whether a repeat entry signal, arriving while a position is open, is
actionable on the **exit** side — as a de-risk or stop-tightening trigger.

**Why this priority**: this is where the measured p = 0.005 effect can actually act. The
signal arrives before the trade's outcome is known, so it is a forward-looking predictor of a
position already in flight. Depends on US1 clearing the confound.

**Acceptance Scenarios**:

1. **Given** a repeat signal on an open position, **When** a rule acts on it, **Then** the rule
   touches the exit stack only and leaves entries byte-identical, so exactly one thing changed.
2. **Given** such a rule, **When** it is evaluated, **Then** both return and drawdown are
   reported, since a de-risk trigger is expected to buy drawdown rather than return.
3. **Given** a rule that tightens a stop, **When** it fires, **Then** the count of trades it
   affected is reported per fold before any performance figure is read.
4. **Given** the incumbent's exit stack, **When** the rule is added, **Then** it is shown not to
   be a restatement of the existing stoploss-grace or ATR-adaptive behaviour.

---

### User Story 3 - Signal context measured where it can act (Priority: P3)

A researcher measures whether **first**-signal context predicts signal quality, using only
information available at signal time.

**Why this priority**: this is the half of the finding that can become an entry gate. Unlike
repeat signals, first signals do become trades, so their outcomes are observed without any
special instrument.

**Acceptance Scenarios**:

1. **Given** each first signal, **When** its context is computed, **Then** every value uses
   only bars at or before the signal.
2. **Given** the candidate context features, **When** they are tested, **Then** the number of
   hypotheses is declared in advance and the significance threshold is corrected for it.
3. **Given** a feature that predicts, **When** it is reported, **Then** its effect size is
   stated alongside its p-value, because a significant but tiny effect cannot clear the floor.

---

### User Story 4 - Masked signals observed honestly (Priority: P4)

A researcher gives the ~43% of discarded signals an observable outcome, using a backtest-only
configuration **as a measurement instrument only**.

**Why this priority**: masked signals have no outcome in the control. Without this they cannot
be studied at all — but the instrument is dangerous and must be fenced.

**Acceptance Scenarios**:

1. **Given** position stacking, **When** it is used, **Then** it is used only to observe
   outcomes, and its P&L is never reported as an edge or a delta against the control.
2. **Given** any figure derived from it, **When** published, **Then** it is labelled
   **not reproducible in dry/live**, per freqtrade's own statement.
3. **Given** the masked-signal outcomes, **When** compared to admitted signals, **Then** the
   comparison accounts for the fact that stacking creates independent trades with their own
   stops, which is not what a deployed change would do.

---

### User Story 5 - A rule that survives the harness (Priority: P5)

A researcher evaluates any adoptable rule through the pooled walk-forward against the incumbent.

**Acceptance Scenarios**:

1. **Given** a candidate rule, **When** its overlap with the incumbent guard chain is measured,
   **Then** that happens **before** any return figure is read.
2. **Given** a rule, **When** it is evaluated, **Then** it runs through `evaluation/` unchanged
   against the 3.350 pp/yr floor, with 7.633 pp as the resolution limit.
3. **Given** a result inside the noise floor, **When** reported, **Then** it is "not resolved",
   never "no effect".

### Edge Cases

- **The effect is entirely explained by unrealised P&L.** Most likely single outcome; a KILL of
  the repeat-signal arm at US1 with the tautology named — after which the first-signal sweep
  continues as EXPLORATORY (FR-023), not as a confirmatory test.
- **The exit-side rule just re-fires the existing stop.** stoploss-grace and the ATR-adaptive
  stop already react to adverse moves; a trigger that fires when they would have is inert.
- **The rule is redundant with the guard chain.** A filter that removes what rvol already
  removes is inert, reported as a defect under Principle IV.
- **A context feature predicts but the effect is tiny.** Significance is not size; anything whose
  **return consequence** is inside 7.633 pp/yr is unresolved regardless of p-value.
- **A large win-rate effect is mistaken for a large return effect.** They are different
  currencies. A 14.8-point win-rate shift can coexist with a return consequence far inside the
  floor, and FR-007 keeps the two apart by construction.
- **Position stacking's P&L leaks into a headline.** The one failure mode that would make this
  feature actively harmful — an undeployable number presented as an edge.
- **Testing five features finds one at p < 0.05 by chance.** Expected roughly one in four
  times; the correction must be declared before looking.

## Requirements *(mandatory)*

**The confound gate**

- **FR-001**: The repeat-signal effect MUST be re-tested conditioned on unrealised P&L at the
  moment the repeat signal arrives.
- **FR-002**: Controls MUST be matched to trades alive at the same holding point.
- **FR-003**: If the effect does not survive conditioning, the feature MUST stop and record the
  result as a proxy relationship, not a finding about signals.

**Causality**

- **FR-004**: Every context feature MUST be computable from bars at or before the signal,
  demonstrated by a truncation test rather than asserted.
- **FR-005**: A trade's own future MUST NOT enter any context feature.

**Multiple testing**

- **FR-006**: The number of context hypotheses MUST be declared before testing, with the
  significance threshold corrected accordingly.
- **FR-007**: Effect size MUST be reported with every p-value, in **two explicitly separate
  currencies**, because conflating them makes the resolution gate meaningless:
  - `effect_winrate_pp` — the difference in win rate, in win-rate percentage points. Reported,
    **never** compared against the return noise floor.
  - `effect_return_pp_yr` — the **annualised return consequence**, computed as
    `Δ mean profit_ratio × 9.0 × affected_trades_per_year`, where 9.0 pp per unit `profit_ratio`
    is the fixed-stake conversion (stake 900 on a 10,000 wallet). This is the only quantity the
    7.633 pp floor may be applied to.

**The measurement instrument**

- **FR-008**: Position stacking MAY be used ONLY to give masked signals an observable outcome.
- **FR-009**: No P&L, return or drawdown figure derived from a stacked run may be reported as
  an edge, a delta, or a candidate result.
- **FR-010**: Every stacked figure MUST carry the label *not reproducible in dry/live*.

**Redundancy (Principle IX)**

- **FR-011**: Any candidate rule's overlap with the incumbent guard chain MUST be measured
  before any performance figure is read.
- **FR-012**: A rule whose blocked set is largely the guard chain's MUST be reported as inert
  under Principle IV, not as evidence about signal context.

**Evaluation**

- **FR-013**: Any candidate rule MUST be evaluated through `evaluation/` unchanged, against the
  incumbent and the four mandatory baselines.
- **FR-014**: Liveness MUST be proven — the rule changed which trades happened — before any
  null is read as a finding.
- **FR-015**: A zero-trade or zero-change arm is inconclusive, never a zero result.

**The exit-side rule**

- **FR-019a**: The action on a repeat signal is **pre-registered and frozen**: tighten the
  current stop distance by a factor of **0.5**. The magnitude is not searched — an unbounded
  "tighten the stop" is a free parameter, and tuning it alongside the trigger is the joint
  search Principle VIII forbids. An outright exit is a **labelled sensitivity** that cannot
  produce the verdict.

- **FR-020**: Any repeat-signal rule MUST act on the exit stack only, leaving entry behaviour
  byte-identical, so the comparison changes exactly one thing.
- **FR-021**: Its return AND drawdown MUST both be reported; a de-risk trigger that buys
  drawdown at flat return is a success, and one that costs return is a trade-off, not a win.
- **FR-022**: It MUST be shown distinct from the incumbent's existing stoploss-grace and
  ATR-adaptive stop. "Distinct" is pre-registered as **trigger overlap below 0.70** — the share
  of the rule's triggers on trades where the existing stack fired anyway within 8 bars. At or
  above 0.70 the rule is a restatement of the existing stop.

**Exploratory continuation**

- **FR-023**: If FR-003 stops the repeat-signal arm, the first-signal sweep MUST continue
  labelled **EXPLORATORY**, and its findings MUST NOT be reported as confirmatory. The
  pre-declared correction of FR-006 applies unchanged and becomes the primary defence.

**Process**

- **FR-016**: All runs MUST execute sequentially.
- **FR-017**: Only files this feature creates or modifies MAY be staged.
- **FR-018**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.
- **FR-019**: The result MUST be written up whatever the outcome.

### Key Entities

- **Signal**: one `enter_long` the strategy produced, whether or not it became a trade.
- **Signal context**: causal facts about a signal's surroundings at signal time. Two of the four
  carry a window that MUST be fixed before testing, or they are silent free parameters inside a
  test claiming exactly four hypotheses: `drawdown_from_high` uses a **96-bar (1-day)** trailing
  high, matching the window convention used elsewhere in this repository; `burst_position` treats
  signals more than **24 bars (6 hours)** apart as separate bursts, chosen to exceed the 3-10h
  median hold so a burst cannot span a completed trade.
- **Masked signal**: a signal freqtrade discarded because the pair held a position.
- **Matched control**: a trade alive at the same holding point, used for the FR-001 comparison.

## Success Criteria *(mandatory)*

### Measurement validity (before any rule is built)

- **SC-001**: The repeat-signal effect is re-tested conditioned on unrealised P&L, with effect
  size and a significance test reported.
- **SC-002**: Causality of every context feature is demonstrated by a truncation test.
- **SC-003**: The hypothesis count is declared before testing and the threshold corrected.
- **SC-004**: Every stacked-run figure is labelled not reproducible in dry/live, and none is
  reported as an edge.

### Protocol §5 (any adoptable rule)

- **SC-005**: Pooled walk-forward return at least the control plus **3.350 pp/yr** with a
  bootstrap interval excluding zero.
- **SC-006**: Positive in at least 10 of 15 folds and in each of W1/W2/W3.
- **SC-007**: Beats all four mandatory baselines net of full costs.
- **SC-008**: Overfitting probability at most 0.2 with deflated Sharpe above zero over the full
  trial log.
- **SC-009**: The delta stays positive at 1.5× fees.
- **SC-010**: A separate process reproduces the accepted result from its saved configuration.
- **SC-011**: The in-sample (W2) advantage does not exceed the W1 advantage by more than the
  floor.

- **SC-018**: Any exit-side rule reports return and drawdown together, with the count of
  affected trades per fold stated before either.
- **SC-019**: The exit-side rule's trigger overlap with stoploss-grace and the ATR-adaptive stop
  is **below 0.70**. At or above that, the rule is inert — a restatement of the existing stop,
  reported as a defect under Principle IV, not a result.

### Kill gates

- **SC-012**: The repeat-signal effect not surviving conditioning on unrealised P&L is a
  **KILL**, recorded as a proxy relationship.
- **SC-013**: A rule redundant with the guard chain is a **KILL**, recorded as a defect under
  Principle IV and explicitly not as "signal context carries nothing".
- **SC-014**: A rule that does not change which trades happen is **inert**, reported as a
  defect, never as a null about signal context.
- **SC-015**: Any **`effect_return_pp_yr`** inside the 7.633 pp noise floor is **not resolved**,
  never "no effect". The floor is an annualised-return quantity and MUST NOT be applied to a
  win-rate difference: the motivating effect is 14.8 **win-rate** points, which would clear
  7.633 while saying nothing whatever about return.
- **SC-020**: An exit-side rule that fires on no trades, or only in fewer than 10 of 15 folds,
  is **inert** and cannot support any claim.

### Process gates

- **SC-016**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-017**: The result report is written whatever the outcome.

## Assumptions

- Fold geometry, purge and embargo come from `evaluation/foldplan.py` via
  `research/labelfree/folds.py`, unchanged.
- The pooled harness and its pinned constants (`BOOTSTRAP_SEED = 0`, floor 3.350 pp/yr, noise
  floor 7.633 pp) are used as-is and never recomputed.
- The incumbent is production `NNNC_MLX`; no model or training change is in scope, so no arm
  requires retraining.
- A negative result is the expected deliverable.

## Non-Goals

- **Position adjustment in either form** — both closed 2026-09-15 (unconditional REJECTED on
  drawdown; profit-gated NOT RESOLVED).
- **Independent concurrent trades per pair** with separate stops — needs an execution-model
  change freqtrade's live engine cannot express.
- **Any change to the model or its training.** This feature is about gates and mechanism, which
  is where this repository's recorded wins have come from.
- Raising `max_open_trades`, which interacts with the per-pair rule and cannot be studied
  alongside it.
