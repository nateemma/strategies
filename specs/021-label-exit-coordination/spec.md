# Feature Specification: Label-Exit Coordination — One Exit Model, One Outcome Definition, Many Triggers

**Feature Branch**: `021-label-exit-coordination`

**Created**: 2026-10-08

**Status**: Draft

**Input**: Redesign the label/entry/exit scheme so a label's outcome is derived from the actual exit policy, and the only thing that varies between labels is the trigger. Enforce a single source of truth for the exit arithmetic.

## Why This Is a Redesign and Not a Patch

Measured on the 15-fold pooled walk-forward harness (3.68 year-equivalent, guards-open,
volume-matched at 1.4735% of bars, per-fold thresholds selected on each fold's own
training window):

| arm | pooled total | verdict vs incumbent |
|---|---|---|
| gbb + 24 indicators (incumbent) | +39.98pp, 15/15 folds positive | — |
| gbb + 8 raw OHLCV features | +35.44pp | −1.23pp/yr, CI spans zero — **features are not the lever** |
| forward_mae + raw | −0.83pp, loses to cash | −11.07pp/yr, p=0.002 |
| forward_mae, incumbent exit restored | −0.22pp | **the exit is not the lever** (0.6pp) |
| take-profit ladder 1.5%→5.0%, models fixed | control 4.0% vs 2.2% = +2.44pp/yr | CI spans zero — **payoff geometry is not the lever** |
| oracle gate, 11 labels | **all PASS**; gbb second from LAST on ceiling | **the gate cannot rank labels** |
| gbb's trigger + forward_mae's capped return | +21.82pp | +6.15pp/yr over forward_mae, CI [+0.64, +11.56], p=0.029 — **the trigger is the mechanism** |

**RE-MEASURED 2026-10-08, after commit 09db623.** Every figure above was measured under
the PRE-FIX exit code. That commit fixed two live exit defects (the post-grace stop sign
flip and the omission of `confirm_trade_exit`'s 1-hour veto from the offline model) plus
the additive-vs-multiplicative fee model, so all five arms were re-evaluated on ONE code
version — reusing the existing per-fold models, so only exits changed. Pre-fix totals
reproduced exactly, which is how the arms above were confirmed to be the ones named.

| arm | pre-fix | post-fix | verdict vs incumbent, POST-FIX |
|---|---|---|---|
| gbb + ind24 (incumbent) | +39.99pp | **+42.71pp** | — |
| gbb + raw8 | +35.45pp | **+43.65pp** | +0.255pp/yr, CI [−7.014, +7.630], p=0.947 — UNCHANGED |
| forward_mae + raw | −0.83pp | **+0.54pp** | −11.450pp/yr, CI [−17.019, −5.895], p=0.000 — WORSENED |
| forward_mae, incumbent exit | −0.22pp | **+0.89pp** | −11.356pp/yr, p=0.000 — WORSENED |
| hybrid gbb_fwdmae | +21.82pp | **+25.61pp** | −4.643pp/yr, CI [−10.535, +1.443], p=0.137 — UNCHANGED |

**Every conclusion survives, and the central one strengthened.** Features: still a null,
now +0.255pp/yr instead of −1.23 — the sign flipped, which is what a null looks like, and
raw8 now edges the incumbent on the point estimate. The exit: swapping it moves
+0.093pp/yr (CI [−1.096, +1.236], p=0.879), down from 0.6pp, so it is even more clearly
not the lever. The trigger: hybrid over forward_mae is now **+6.801pp/yr, CI [+1.575,
+12.066], p=0.011**, up from +6.15pp/yr at p=0.029.

One wording correction: forward_mae + raw no longer "loses to cash" — it is flat
(+0.54pp over 3.68 yr-equivalent, +0.15pp/yr, plainly null). The substantive claim is
untouched: it is −11.45pp/yr worse than the incumbent at p<0.001. The take-profit ladder
and oracle-gate rows were not re-run; the ladder's verdict is a CI-spans-zero null whose
rungs all shift together under an exit change, and the gate is training-free and
label-side.

Every patchable surface has been eliminated in turn. What remains is not a component but
a missing abstraction: labels and the exit stack are specified independently, and none of
the 28 label methods consults the exit at all.

## Clarifications

### Session 2026-10-08

- Q: Should a bar's outcome be excluded or flagged when that bar could not actually have been filled at production stake size? → A: Record a fillable flag and fillable fraction per bar; report economic value both unfiltered and filtered (Option B)
- Q: Should a "positive outcome" in the table mean profit after trading fees, or before them? → A: Net of the same fees freqtrade applies in the backtest; no extra slippage model (Option A)
- Q: What should decide whether a trigger is called "selection value" rather than "decoration"? → A: Compare each trigger's no-model economic value against the EMPTY trigger; selection value means beating it, decoration means matching it while its label is markedly more learnable (Option B). Extended during integration: a third verdict, "indeterminate", covers the case where neither condition holds, and both boundaries are set by the measurement uncertainty on the difference rather than an invented threshold.
- Q: How should the learnability of each trigger's label be measured for the second column? → A: One fixed cheap probe — same model, same features, same protocol — for every trigger, reported as MCC (Option A)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One exit model, provably unchanged (Priority: P1)

As the strategy author, I want the exit policy to exist in exactly one place, so that a
label that claims to be exit-aware is reasoning about the same rules the strategy actually
trades, and so that changing the exit cannot leave a stale copy behind.

**Why this priority**: Everything else in this feature depends on being able to simulate
the exit. Doing that against a second copy of the arithmetic would reproduce, at larger
scale, the exact failure this repository has already suffered — duplicated loss logic
across base/Enhanced/MT copies where wiring checks passed while the term never ran.

**Independent Test**: Extract the exit policy, point both the live path and an offline
simulator at it, and verify live stop decisions are identical bar-by-bar over real data
before and after.

**Acceptance Scenarios**:

1. **Given** the exit policy has been extracted, **When** the live strategy path computes a
   stop for every bar of a multi-pair, multi-year dataset, **Then** every decision matches
   the pre-extraction implementation exactly.
2. **Given** a developer adds a second implementation of an ATR multiplier, stop cap/floor,
   grace window, take-profit or hold cap anywhere in the codebase, **When** the test suite
   runs, **Then** it fails and names the duplicate.
3. **Given** `_stop_aware_labels` in the NNPredict family, **When** its implementation is
   inspected, **Then** it calls the shared exit model and contains no stop arithmetic of
   its own.

### User Story 2 - The per-bar outcome table (Priority: P1)

As the strategy author, I want a precomputed table giving, for every bar, what would have
happened to a trade entered there and closed by the real exit policy, so that evaluating a
candidate label becomes a query rather than a training run.

**Why this priority**: This is what makes the whole line of enquiry affordable. Each label
comparison currently costs about 1.5 hours of training; against the table it costs seconds.
It is also the only artefact that can measure a trigger's economic value with no model in
the loop, which User Story 4 requires.

**Independent Test**: Build the table for the production dataset, then reproduce a known
arm's trade outcomes from it without running a backtest.

**Acceptance Scenarios**:

1. **Given** a dataset and an exit configuration, **When** the table is built, **Then**
   every bar carries a realised profit fraction, a hold duration, and which rule closed the
   trade (stop / take-profit / hold cap / end of data).
2. **Given** an existing table, **When** any exit parameter changes, **Then** the table is
   treated as invalid and rebuilt rather than silently reused.
3. **Given** the same dataset and exit configuration, **When** the table is built twice,
   **Then** the results are identical.
4. **Given** the table, **When** any model training path is inspected, **Then** no outcome
   column can reach a model as an input feature.

### User Story 3 - Triggers as the only variable (Priority: P2)

As the strategy author, I want a label to be expressed as a trigger over bars combined with
the one shared outcome definition, so that comparing two labels compares their triggers
rather than conflating trigger quality with whichever thresholds each author happened to
pick.

**Why this priority**: Labels are currently not comparable to one another. Measured BUY
share across today's candidates ranges from 7% to 33%, and that spread is a property of
each labeler's private thresholds, not of the idea it encodes.

**Independent Test**: Express several existing labels as triggers, recombine each with the
shared outcome, and confirm the resulting label sets differ only where the triggers differ.

**Acceptance Scenarios**:

1. **Given** the existing label family, **When** each is examined, **Then** those that
   reduce to a trigger are expressed as one, and those that do not are listed with the
   reason.
2. **Given** a trigger and the outcome table, **When** a label is formed, **Then** it is
   the conjunction of the trigger and a positive outcome, with no label-specific outcome
   test.
3. **Given** the trigger set, **When** it is inspected, **Then** it includes at minimum
   gbb's two-threshold condition, the empty trigger (every bar, the forward_mae case), and
   the smoothed zero-lag geometric valley condition that exists today but has never been
   tested.

### User Story 4 - Separating economic value from learnability (Priority: P1)

As the strategy author, I want each trigger scored on two independent axes — what it is
worth with no model at all, and how learnable the label built from it is — so that I can
tell whether a trigger earns its keep by selecting collectable moments or merely by handing
the model an easy target.

**Why this priority**: This is the open question the feature exists to answer, and no
experiment run so far can separate the two explanations. Across four measured arms,
learnability tracks return almost perfectly (0.204 / 0.521 / 0.580 / 0.622 against −0.8 /
+21.8 / +35.4 / +40.0pp). Explanation (a) — the trigger selects moments when a forward move
is collectable — and explanation (b) — the trigger hands the model a target it can partly
read off the trigger's own inputs, making the entry rule the edge and the model decoration
— predict exactly that ordering. They imply opposite strategies.

**Independent Test**: Produce the two-column leaderboard and confirm that a deliberately
constructed trigger with known high selection value and known low learnability lands in the
expected quadrant.

**Acceptance Scenarios**:

1. **Given** a trigger and the outcome table, **When** its economic value is computed,
   **Then** the result uses no model and reports base rate, mean realised outcome and its
   distribution.
2. **Given** the same trigger, **When** the corresponding label is scored for learnability,
   **Then** that number is reported in a separate column and is never combined with the
   economic value into a single score.
3. **Given** the completed leaderboard, **When** a trigger shows high economic value with
   no model, **Then** it is identified as having selection value; **When** a trigger shows
   near-zero economic value but high learnability, **Then** it is identified as the
   decoration case.

### Edge Cases

- A bar whose forward window runs past the end of the data: the outcome must be recorded as
  closed by end-of-data and distinguishable from a genuine exit, so it can be excluded
  rather than silently counted as a loss.
- A bar inside the stop-loss grace window, where the effective stop is far wider than the
  normal one: the simulation must reproduce that, since the grace window is longer than some
  candidate label horizons.
- Gap-fill bars (volume zero, OHLC equal), which are 22–55% of 15m bars for six of eleven
  production pairs: outcomes computed across them describe the absence of trading and must
  be flagged so conclusions can be checked against them.
- A bar whose quote volume cannot support the production stake: the outcome is still recorded
  but marked unfillable, so a trigger cannot earn its value from bars that could not have been
  traded.
- A trigger that fires on almost no bars, or on almost every bar: the leaderboard must
  report base rate alongside value so a trigger cannot look good by trading nothing.
- Two triggers that fire on overlapping bars: the comparison must not assume independence.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The exit policy MUST exist in exactly one place. **AMENDED 2026-10-08 after
  T021** — the original enumeration was incomplete, which made the first outcome table
  optimistic by 10.7% on mean profit per trade. The policy comprises:
  - the ATR-adaptive stop with its cap and floor, and the static fallback
  - the stop-loss grace window
  - `metric_overbought` — `guard_metric > 0.98` while in profit (fires on 0.3% of SOL bars
    and 2.6% of ZEC bars, so it is rare but NOT inert)
  - the take-profit (`cexit_take_profit`)
  - `unclog_12h` — 12 hours held with profit > 0.5%
  - `unclog_1d` — 1 day held with profit >= 0
  - `max_hold` — `cexit_max_days`
  - the native ROI exit (`minimal_roi`, 5% at the Open operating point)
  Two exits are **inert at this operating point and MUST be documented rather than
  modelled**: the three `rsi_*` exits require an `rsi` column that the MINIMAL populator
  does not produce, and `strategy_custom_exit` is not overridden anywhere in the NNNC
  chain. `enable_exit_signal` is also false. Together these mean the Open arm's exit is
  **model-independent**, which is what makes a model-free outcome table possible at all; if
  a future arm overrides `strategy_custom_exit` or enables the exit signal, that premise
  fails and this feature's approach does not apply to it.
- **FR-002**: Every consumer of exit behaviour, live or offline, MUST call that one
  implementation. No other code may re-implement any part of it.
- **FR-003**: A test MUST fail if the exit arithmetic appears in more than one place. This
  constraint is enforced mechanically, not by convention.
- **FR-004**: BOTH existing duplicates MUST become callers of the shared exit model, with
  their arithmetic removed: `_stop_aware_labels` in the NNPredict family, AND
  `BaseStrategy._resolve_normal_stop`, which duplicates `custom_stoploss`'s ATR block
  inside the same class and says so in its own docstring. The duplication is three-way,
  not two-way; an earlier draft of this requirement named only the first, which would have
  left FR-003's test failing on an uncited copy.
- **FR-005**: Live stop behaviour MUST be unchanged by the extraction, demonstrated by a
  bar-by-bar comparison over real multi-pair, multi-year data.
- **FR-006**: The system MUST produce a per-bar outcome table recording, for a hypothetical
  entry at each bar, the realised profit fraction, the hold duration, and the closing rule.
- **FR-007**: The outcome table MUST be keyed by dataset and exit configuration, so that a
  change to the exit invalidates it rather than silently reusing stale values.
- **FR-008**: Rebuilding the table from the same inputs MUST produce identical results.
- **FR-009**: The outcome table MUST be structurally prevented from reaching a model as an
  input feature. Its lookahead is legitimate for label construction and disqualifying as a
  feature.
- **FR-006a**: A "positive outcome" MUST mean profit NET of the same trading fees freqtrade
  applies in a backtest, so the table and the walk-forward harness measure the same quantity.
  No slippage model beyond those fees is introduced, because the harness does not model one
  and a table that did would disagree with the instrument that has to confirm its winners.
- **FR-009a**: Each outcome record MUST carry whether that bar was fillable at production
  stake size, and the fillable fraction of the intended stake, derived from the same
  quote-volume cap the live path applies.
- **FR-009b**: Every economic value MUST be reported twice — over all triggered bars, and
  over fillable triggered bars only. A trigger whose value survives only unfiltered is
  reporting phantom return.
- **FR-010**: A trigger MUST be expressible as a boolean condition over bars, independent of
  any outcome test.
- **FR-011**: A label MUST be formed as the conjunction of a trigger and a positive outcome
  drawn from the shared table, with no label-specific outcome definition.
- **FR-012**: Existing label methods MUST be examined and expressed as triggers where they
  reduce to one; those that do not MUST be listed with the reason.
- **FR-013**: The trigger set MUST include gbb's two-threshold condition, the empty trigger,
  and the smoothed zero-lag geometric valley condition.
- **FR-014**: For each trigger, the system MUST report an economic value computed from the
  outcome table alone, with no model: base rate, mean realised outcome, and distribution.
- **FR-014a**: The EMPTY trigger (every bar) MUST serve as the null against which every other
  trigger is judged. It is already a required member of the trigger set, so no threshold has
  to be agreed in advance.
- **FR-014b**: Each trigger MUST be assigned exactly one verdict, by this rule:
  - **selection value** — its no-model economic value exceeds the empty trigger's by more
    than the measurement uncertainty on that difference;
  - **decoration** — its economic value does NOT exceed the empty trigger's once uncertainty
    is accounted for, while its label's learnability DOES exceed that of the empty trigger's
    label by more than the uncertainty on that difference;
  - **indeterminate** — neither condition holds, which MUST be reported as such rather than
    resolved by judgement.
  The uncertainty on each difference MUST be stated alongside the verdict.
- **FR-015**: For each trigger, the system MUST separately report the learnability of the
  corresponding label, measured by ONE fixed cheap probe — the same model, the same features
  and the same protocol for every trigger — reported as MCC. Varying the model between
  triggers would make the column incomparable and the decoration verdict unreliable.
- **FR-015a**: The learnability column is DIAGNOSTIC. It feeds the verdict rule in FR-014b
  and nothing else. It MUST NOT license promotion of any trigger, consistent with FR-018.
- **FR-016**: Economic value and learnability MUST be reported as separate columns and MUST
  NOT be combined into a single score.
- **FR-017**: Any claim about profit-and-loss MUST come from the pooled walk-forward
  harness, which resolves roughly 3–4 percentage points per year and not 1–2.
- **FR-018**: Classifier metrics MAY be reported as context and MUST NOT be used as an
  acceptance input.
- **FR-019**: The exit policy MUST be held fixed throughout. This feature redesigns the
  label around it.
- **FR-020**: Outcomes computed across gap-fill bars MUST be identified so conclusions can
  be checked against that contamination.
- **FR-021**: No change may alter the behaviour of shared runtime code that the live
  dry-run depends on.
- **FR-022**: The outcome table MUST be reconciled against a known arm's realised trades
  before any trigger verdict is accepted, and the divergence MUST be quantified. **Added
  after T021**, which found a 2.70pp win-rate and 10.7% mean-profit overstatement that no
  other check would have caught. A table that disagrees with the pooled harness is wrong,
  and the bias only cancels out of a trigger comparison if it is uniform across triggers —
  which cannot be assumed.

### Key Entities

- **Exit policy**: the single definition of how an open position is closed — adaptive stop
  with cap and floor, static fallback, grace window, take-profit, hold cap.
- **Outcome record**: for one bar, the result of a hypothetical entry closed by the exit
  policy — profit fraction net of fees, hold duration, closing rule, whether the bar was
  fillable at production stake, and the fillable fraction of that stake.
- **Outcome table**: the full set of outcome records for a dataset under one exit
  configuration, with the identity of that configuration attached.
- **Trigger**: a boolean condition over bars saying when an entry may be considered,
  carrying no outcome test.
- **Label**: a trigger combined with the shared positive-outcome condition.
- **Trigger leaderboard**: per trigger, its base rate, economic value with no model, and the
  learnability of its label, as separate columns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Live stop decisions after the extraction match the pre-extraction
  implementation on 100% of bars across a multi-pair, multi-year dataset.
- **SC-002**: The exit arithmetic exists in exactly one place, and introducing a second
  copy causes a test to fail.
- **SC-003**: Building the outcome table twice from the same dataset and exit configuration
  produces identical results.
- **SC-004**: Changing any exit parameter causes the table to be rebuilt rather than reused.
- **SC-005**: Evaluating a candidate trigger against the table — both columns, economic value
  and probe learnability — takes seconds rather than the roughly 1.5 hours a trained
  comparison currently costs.
- **SC-006**: Every trigger in the leaderboard carries both an economic-value column and a
  learnability column, neither derivable from the other, with the learnability figure produced
  by the same probe for every trigger.
- **SC-007**: The leaderboard assigns every trigger exactly one of three verdicts — selection
  value, decoration, or indeterminate — by comparison against the empty trigger, with the
  measurement uncertainty on each difference reported beside it.
- **SC-008**: At least the three named triggers are scored: gbb's two-threshold condition,
  the empty trigger, and the geometric valley condition.
- **SC-009**: The share of each trigger's outcomes computed across gap-fill bars is reported
  alongside its value.
- **SC-010**: Every trigger's economic value is reported both over all triggered bars and over
  fillable triggered bars only, and the two figures are distinguishable in the leaderboard.

## Assumptions

- The exit policy is calibrated and is held fixed. This rests on today's take-profit ladder
  (control UNCHANGED from 2.2% to 4.0%, CI spanning zero) and prior grace-level and ATR-band
  sweeps, which found no resolvable lever.
- Costs are the exchange fees freqtrade applies in a backtest, matching the harness exactly.
  No slippage is modelled, because the harness models none; revisiting that is out of scope
  and would have to change both together.
- The production pair list and the 15m timeframe are unchanged, and no new data is
  downloaded.
- A single position per bar is simulated. Position sizing, concurrency limits and
  max-open-trades are properties of the strategy, not of the label, and are left to the
  walk-forward harness when a candidate is promoted.
- The outcome table is built for long entries, matching the entry side the current family
  trades.

## Out of Scope

- Changing the exit policy itself.
- Hyperopt.
- Any live or dry-run change.
- New data downloads.
- Promoting any configuration to production. This feature produces one shared abstraction
  and the evidence to choose among triggers; it does not deploy.

## Dependencies

- The pooled walk-forward harness in `evaluation/`, which is the only acceptable source of a
  profit-and-loss claim.
- The existing label family, as the source of candidate triggers.
- The gap-fill contamination finding, which qualifies every conclusion drawn from this data.
