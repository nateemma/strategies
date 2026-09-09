# Feature Specification: Pooled Walk-Forward Evaluation Harness

**Feature Branch**: `007-pooled-walkforward-harness`
**Created**: 2026-09-09
**Status**: Draft
**Input**: Build a pooled walk-forward evaluation harness. A MEASUREMENT INSTRUMENT, not a strategy — the deliverable is statistical power, not P&L.

## Context: why this exists

Measured 2026-09-09, by bootstrapping the acceptance window's own per-trade P&L
(NNNC_MLX, W1, 99 trades, +9.831%):

| quantity | value |
|---|---|
| 95% CI on the W1 return statistic | [2.383%, 17.649%] |
| half-width — **the noise floor** | **7.633pp** (6.1pp when arms are paired) |
| sampling SD | 3.893pp |
| `compare.py` `POWER_THRESHOLD_PP` (since removed) | 1.000pp — ~7.6x too lenient |

Three confirmed consequences:

1. Two arms returned "IMPROVED" from the old threshold. Under a paired bootstrap both are
   **NOT RESOLVED**: rvol 2.5 vs 2.0 = +1.86pp, CI [-4.25, +8.00], p=0.550; grace -0.10 =
   +1.04pp, CI [-2.02, +4.29], p=0.532.
2. An entire parameter sweep fits inside the noise band — the NNNC `entry_rvol_threshold` sweep
   spans 6.764%..11.694% = 4.93pp against a ±6.1pp interval. Its "optimum" is a ranking of noise.
3. Optima are **not trackable**. Full sweeps of two parameters across three windows disagree on
   the best value every time, with contradictory rank correlations (rvol W1~W2 -0.46, W1~W3 -0.43,
   W2~W3 +0.79; grace W1~W2 +0.71, W1~W3 -0.09, W2~W3 -0.03), none significant at 6-7 rungs.

`compare.py` was rewritten (31183cb) to use the paired bootstrap. That made it **honest**, not
**powerful** — it now correctly says "not resolved" to nearly everything. This feature supplies
the power it reports on.

**A correction this feature carries:** the W1/W2 "reversals" recorded earlier are NOT evidence of
regime dependence. Two noise draws disagreeing is the simpler explanation and must not be
re-derived as a regime effect.

## What is actually achievable (established before planning, not during)

Data on disk spans **2021-01-01 .. 2026-09-07 (2,075 days, 5.68 yr)**. Trade rates from the
current acceptance window: NNNC 99/yr, NNMT 183/yr at 11 pairs.

| train window | test span available | NNNC pooled trades (11 / 22 pairs) | est. floor (11 / 22 pairs) |
|---|---|---|---|
| 24 mo | 3.68 yr | ~365 / ~730 | 3.98pp / 2.81pp |
| 18 mo | 4.19 yr | ~414 / ~829 | 3.73pp / 2.64pp |
| 12 mo | 4.68 yr | ~464 / ~928 | 3.53pp / 2.49pp |

**This is the honest ceiling and the spec is built on it.** Pooling every fold the data allows,
at double the universe, moves the floor from 7.633pp to roughly **2.5-2.8pp — a 2.7x improvement,
not 10x.** Resolving a 1pp effect would need a ~58x larger sample, which does not exist.

Therefore:

- Effects in the **3-7pp band become newly measurable**. That band contains the changes that have
  actually mattered here (the exit-defect repair was +4.5pp / +12.0pp; augmentation and label
  rebuilds are of that order). Today they sit inside the noise and cannot be confirmed.
- Effects at **parameter scale (1-2pp) remain invisible even after this feature ships.** Anything
  that returned "not resolved" this session would still return "not resolved". This is a limit of
  the available data, not of the harness, and the feature must not be sold as fixing it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Pool many genuinely out-of-sample folds (Priority: P1)

A researcher evaluates a strategy across repeated rolled-forward train/test splits, where every
test fold is strictly out-of-sample for the model tested on it, and all fold trades are pooled
into a single trade set before any statistic is computed.

**Why this priority**: Without this nothing else in the feature exists. It is also the only part
that removes the current structural defect — W2 and W3 lie inside the training range
(20220101-20250831), which is exactly why they cannot carry acceptance today.

**Independent Test**: Run the harness over a defined fold plan for one strategy; confirm every
fold's test range is disjoint from its own training range, and that the pooled trade count
materially exceeds the 99 of the single current window.

**Acceptance Scenarios**:

1. **Given** a fold plan, **When** the harness runs, **Then** each fold trains on its own range,
   tests on a later disjoint range, and stores its model under a fold-specific path.
2. **Given** a fold whose test range overlaps its training range, **When** the harness validates
   the plan, **Then** it REFUSES the fold with a named error rather than reporting its numbers.
3. **Given** completed folds, **When** results are pooled, **Then** the pooled set is the union of
   per-fold trades with fold identity retained on every trade.

### User Story 2 - Report the achieved noise floor (Priority: P1)

The researcher sees, as the harness's primary output, the measured bootstrap half-width on the
pooled trade set, next to today's 7.633pp single-window baseline.

**Why this priority**: The feature's entire justification is reducing the resolvable effect size.
An unmeasured claim of improvement would repeat the error that made this feature necessary.

**Independent Test**: Run the harness, read the reported floor, and confirm it is computed from
the pooled per-trade P&L rather than asserted.

**Acceptance Scenarios**:

1. **Given** a pooled trade set, **When** the report is produced, **Then** it states the bootstrap
   95% CI half-width, the pooled trade count, and the ratio against the 7.633pp baseline.
2. **Given** pooled folds, **When** the floor falls more slowly than 1/sqrt(n) predicts, **Then**
   the shortfall is REPORTED with the inter-fold return correlation that explains it, not hidden.
3. **Given** the achieved floor, **When** it is reported, **Then** the minimum resolvable effect
   size is stated explicitly so a reader knows what the instrument can and cannot see.

### User Story 3 - Feed the existing comparison tool (Priority: P2)

An arm-vs-control comparison consumes pooled fold output through the existing paired-bootstrap
comparison, so verdicts come from the pooled set rather than one window.

**Why this priority**: Delivers the power to the place decisions are actually made. Depends on
US1 and US2 but is a distinct, separately testable step.

**Independent Test**: Compare a pooled arm against a pooled control and confirm the verdict cites
the pooled trade count and CI.

**Acceptance Scenarios**:

1. **Given** pooled arm and control sets, **When** compared, **Then** trades pair on identity
   within fold and the verdict reports the pooled CI and p-value.
2. **Given** an arm differing from its control in fewer than the minimum testable number of
   trades, **When** compared, **Then** the result is inconclusive-underpowered, not a win.

### User Story 4 - Test whether more pairs buy samples without costing expectancy (Priority: P3)

The researcher measures whether expanding the traded universe raises the pooled trade count while
holding per-trade expectancy, since doubling the universe is the difference between a ~3.9pp and a
~2.8pp floor.

**Why this priority**: The largest single lever on the achieved floor, but it is a genuine open
question — a prior study found the edge is structurally the high-volatility alt and that expanding
the universe has no lever for EDGE. This feature wants SAMPLES, which is a different question, and
must not assume the answer.

**Independent Test**: Run the harness at the current universe and at an expanded one; compare
pooled trade count and per-trade expectancy.

**Acceptance Scenarios**:

1. **Given** an expanded universe, **When** pooled, **Then** both trade count AND per-trade
   expectancy are reported per pair-group.
2. **Given** added pairs whose expectancy is materially below the incumbents', **Then** the
   expansion is reported as NOT a valid source of power, with the measured expectancy gap.

### Edge Cases

- A fold trains successfully but produces zero trades in its test range — recorded as zero-trade,
  never as zero-return, and excluded from expectancy statistics while still counting as a
  completed fold.
- Two seeds produce identical trades in a fold. The entry gate is often tight enough for this;
  it is ONE measurement, not two, and must be detected and reported as such.
- A fold's training crashes but the run still exits successfully — the harness must verify the
  model artefact exists rather than trusting an exit code.
- Fold model directories collide, so one fold silently evaluates another's weights.
- The per-trade archive for a fold is pruned before pooling, making that fold unpoolable.
- Fold returns are strongly correlated through a shared market regime, so pooling reduces the
  floor more slowly than independence predicts.
- The earliest folds fall in a market era so different that including them changes the answer;
  the report must allow the floor to be recomputed on a recent subset.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The harness MUST accept an explicit fold plan (train range, test range per fold) and
  validate it before running anything.
- **FR-002**: The harness MUST refuse any fold whose test range intersects its own training range,
  naming the offending fold.
- **FR-003**: Each fold MUST train and store its model under a fold-unique identity so folds
  cannot overwrite each other's weights.
- **FR-004**: The harness MUST verify each fold's trained model artefact exists before evaluating
  that fold, and MUST NOT rely on process exit status.
- **FR-005**: The harness MUST evaluate each fold only on its own out-of-sample test range.
- **FR-006**: The harness MUST pool per-trade results across folds, retaining fold identity, pair,
  and entry time on every trade.
- **FR-007**: The harness MUST compute and report the bootstrap 95% CI half-width on the pooled
  trade set as its headline output.
- **FR-008**: The harness MUST report the achieved floor against the 7.633pp single-window
  baseline, and state the resulting minimum resolvable effect size.
- **FR-009**: The harness MUST report the inter-fold correlation of returns, and flag when the
  floor falls more slowly than 1/sqrt(n) predicts.
- **FR-010**: The harness MUST record per-fold trade counts and per-fold returns, not only pooled
  aggregates.
- **FR-011**: The harness MUST detect and report folds where two seeds produce identical trades.
- **FR-012**: The harness MUST record zero-trade folds as zero-trade, never as zero-return.
- **FR-013**: Pooled comparisons MUST be produced by the existing paired-bootstrap comparison
  rather than a second, divergent statistic.
- **FR-014**: The harness MUST report per-pair-group trade count and expectancy when the universe
  is expanded. **SCOPE NOTE**: this requirement is CONDITIONAL and the first run does not expand
  the universe, so it is vacuously satisfied there. It is exercised only in the deferred
  pair-expansion phase (research R10); the same applies to SC-006.
- **FR-015**: The harness MUST NOT alter any entry gate, exit parameter, label, or model
  configuration. Its only effect is on how results are measured.
- **FR-016**: The harness MUST run its work sequentially.
- **FR-017**: The harness MUST state its compute budget in advance and report actual elapsed cost.
- **FR-018**: The harness MUST NOT change the existing arm-record schema, which 480+ stored
  records depend on.
- **FR-019**: The harness MUST NOT modify shared framework mechanism code.
- **FR-020**: Any run whose fold set is incomplete MUST be reported as partial, and MUST NOT
  present a pooled floor as if it came from the full plan.
- **FR-021**: The harness MUST allow the pooled floor to be recomputed over a subset of folds, so
  era sensitivity can be inspected.
- **FR-022**: The harness MUST be reproducible — a rerun on unchanged inputs MUST produce the same
  floor and the same verdicts.

### Key Entities

- **Fold**: one train range, one disjoint later test range, a model identity, and a completion
  status.
- **Fold Plan**: the ordered set of folds, validated for train/test disjointness before execution.
- **Pooled Trade Set**: the union of per-fold trades with fold, pair, and entry time retained.
- **Power Report**: pooled trade count, bootstrap CI half-width, comparison against the 7.633pp
  baseline, minimum resolvable effect, inter-fold correlation, per-fold breakdown.
- **Pair Group**: incumbent versus added pairs, carrying trade count and per-trade expectancy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (obligation, MANDATORY)**: The pooled evaluation's bootstrap 95% CI half-width is
  **measured and reported** against the 7.633pp single-window baseline, together with the
  improvement ratio.
- **SC-001a (target, ASPIRATIONAL)**: That half-width is **at most 4.0pp** — a reduction of at
  least 45%. The chosen geometry is estimated at 3.98pp, a margin of 0.02pp, so this target may be
  missed on measurement noise alone. **Missing it is a reportable RESULT, not a feature failure**,
  and is never grounds to retune the fold geometry until the number improves (research R9).
- **SC-002**: The pooled out-of-sample trade count is **at least 350** for the single-universe
  configuration, against 99 today.
- **SC-003**: Every fold in a completed run is verifiably out-of-sample: zero folds with any
  overlap between their own train and test ranges.
- **SC-004**: The achieved floor and the minimum resolvable effect size are both stated in the
  report, so no reader can mistake what the instrument can see.
- **SC-005**: A rerun on unchanged inputs reproduces the reported floor exactly.
- **SC-006**: Where the universe is expanded, per-pair-group expectancy is reported, and any
  expansion whose added pairs materially underperform the incumbents is reported as an invalid
  source of power.
- **SC-007**: If the pooled floor falls more slowly than 1/sqrt(n) predicts, the report states the
  shortfall and the inter-fold correlation behind it.
- **SC-008**: No entry gate, exit parameter, label, or model configuration differs before and
  after the feature.

## Assumptions

- Trade rates from the current acceptance window (NNNC 99/yr, NNMT 183/yr at 11 pairs) carry
  approximately into earlier eras. If earlier folds trade at materially different rates the pooled
  count will differ from the projection; the harness reports actuals, so this self-corrects.
- The 1/sqrt(n) scaling used for the projections assumes fold returns are independent. They may
  not be — FR-009 exists precisely to measure this rather than assume it.
- Data on disk (2021-01-01 .. 2026-09-07) is the binding limit on fold count. No new data sources
  are in scope.
- The per-fold retrain cost dominates and is roughly 10-15 minutes per model on current hardware.
- The existing paired-bootstrap comparison is correct and is the statistic of record.

## Non-Goals

- Finding new edge, or improving returns in any way.
- Tuning any parameter; changing the label, gate, exits, or model.
- Hyperopt of any kind.
- New data sources or exchanges.
- **Any increase in trade count obtained by lowering the entry bar.** This is explicitly
  forbidden: it has been measured repeatedly and buys volume by accepting worse trades
  (guards-off 1,737 trades and -15.99%; rvol 2.0->0.2 gave +52 trades and -2.9pp; disabling the
  NNMT task filter gave 4.5x trades at ~15x less per-trade profit).
- Making 1-2pp parameter effects resolvable. The data cannot support it and this feature does not
  claim to.

## Constitution Interaction (flagged for the planning phase)

Principle II (Persistence Across Temporally Distant Eras, NON-NEGOTIABLE) requires an edge to hold
in a temporally distant window, and the repo has used "it flipped in a distant era" as grounds for
rejection. This feature's premise does not remove that requirement, but it **changes what a flip
means**: a flip between two underpowered windows is not evidence of regime dependence, because
two noise draws disagreeing looks identical. The planning phase must state how persistence is
demonstrated once folds are pooled — pooling is not a licence to stop checking era robustness, and
FR-021 exists so era sensitivity remains inspectable.
