# Feature Specification: Indicator Screen Against Labels and Regimes

**Feature Branch**: `037-indicator-screen` (spec directory; work is committed to master)

**Created**: 2026-10-16

**Status**: Draft

**Input**: User description: "Look at the strategies in SimpleStrategies. Right now, they test the
effect of individual indicators (plus some handling of bull/bear regimes in some cases). Can you create
a variant that contains all of the indicators used, and labels from one of the forward looking training
labels [...] What I would like to see is a study of whether any of the indicators correlate to the
buy/sell signals. Maybe test with gbb_path, because we know the answer is at guard_metric and bb_width
(and probably correlated indicators). [...] another useful study would be the same thing, but correlated
to market regime (so that we can see if we can find a decent regime identification metric)." Clarifications (2026-10-16): Q1 = regime horizons 12 h, 3 d and 7 d;
Q2 = regime A reported from both the gbb-trigger outcome table and the width-book trades. Follow-ups:
all three regime definitions (strategy outcome, market direction, volatility) are in scope; no entry or
exit guards; combinations of indicators are in scope. Draft:
`research/indicator_screen/spec_037_draft.md`.

## Background *(evidence; memory hubs index_features, index_entry_gates, project_model_check_width_sort)*

- **What SimpleStrategies measures today.** It has 67 indicator strategies: 56 built on the plain base
  and 11 on the Ehlers base. Each tests one indicator by P&L, behind its own volatility, regime,
  trend-mode and slope filters. None says whether the indicator carries information about the outcomes
  NNNC is trained on.
- **What is known about the gbb_path label.**
  - Its BUY label is built partly from guard_metric and bb_width.
  - Within gbb candidate bars, the NNNC model behaves like a band-width sort: AUC 0.53–0.62, and a
    model-free width book matches every NNNC book.
  - Label (034) and new-input (035) screens found nothing that beats it.
  - The model sees 24 indicators; most SimpleStrategies indicators have never been scored against the
    label.
- **Regime.** No dependable regime indicator exists:
  - the BTC regime gate failed (feature 001);
  - the volatility-regime gate failed on NNNC entries (026, switches experiment), because trades NNNC
    takes in the "off" state are profitable.
- **Prior art reused, not rebuilt:**
  - a single-pair indicator ranking tool (Pearson, Spearman and mutual information against gbb labels,
    on the 24 model columns only, without walk-forward);
  - per-fold label construction from the label screen (034);
  - the trigger causality truncation test (021);
  - the per-bar trigger outcome table under the Open exit (021).

## User Scenarios & Testing *(mandatory)*

The user is the researcher. They want to know which indicators carry information about future
outcomes, so later features can make model inputs, entry triggers and regime gates from evidence
rather than from one indicator's backtest.

### User Story 1 - One harvest of every SimpleStrategies indicator (Priority: P1)

The researcher can see, for every pair and bar, the value of every indicator the SimpleStrategies
family computes, and each strategy's own raw buy and sell signal. "Raw" means before any guard or
filter. All columns come from one consistent, causal dataset that later screens read without
recomputing.

**Why this priority**: Every other story reads this harvest. If a column is wrong, missing or looks
ahead, every score built on it is wrong.

**Independent Test**: Build the harvest for one fold window. Check that:
- every strategy is present or listed as excluded with a reason;
- each strategy's raw signal equals what that strategy produces before its filters;
- every kept column passes the lookahead truncation test.

**Acceptance Scenarios**:

1. **Given** the 67 indicator strategies, **When** the harvest is built, **Then** each strategy
   contributes its indicator columns and its raw buy/sell signals, or appears in an exclusions list
   with the reason (error, constant output, lookahead).
2. **Given** a strategy with its own filters, **When** its raw signal is harvested, **Then** the signal
   equals that strategy's own entry/exit signal before those filters. No filter, guard or regime gate
   is applied.
3. **Given** any harvested column, **When** the data is truncated at an earlier bar and the column is
   recomputed, **Then** every value up to the cut is identical. Otherwise the column is excluded as
   lookahead and named in the report.
4. **Given** the harvest, **When** it is opened as a normal strategy in the plotting tool, **Then** the
   indicators can be plotted against price.

---

### User Story 2 - Single-indicator screen against the gbb_path labels (Priority: P1)

The researcher gets a ranked table of every harvested column and raw signal against the gbb_path BUY
and SELL labels, on two populations: all bars, and gbb candidate bars only. The table shows the
strength of each relationship and whether it is stable across the walk-forward test windows. It also
shows which indicators are guard_metric or bb_width in disguise, and which still carry information once
those two are accounted for.

**Why this priority**: This is the core question. The all-bars ranking is also the check that the
instrument works: the label is partly built from guard_metric and bb_width, so they must come out on
top.

**Independent Test**: Run the screen on the harvest. Confirm that guard_metric and bb_width rank at or
near the top for BUY on all bars before reading any other row.

**Acceptance Scenarios**:

1. **Given** the harvest and the labels, **When** the all-bars screen runs, **Then** guard_metric and
   bb_width are each in the top 5 columns for BUY by median strength. If either is not, the screen is
   marked INSTRUMENT FAIL and no other row is reported as a finding.
2. **Given** the candidate-bar population, **When** the screen runs, **Then** each column has: a median
   strength and direction over the test windows; the share of windows with the same direction; a
   non-monotone association measure; and, for raw signals, the lift (label rate when the signal fires ÷
   base rate).
3. **Given** the screen, **When** redundancy is reported, **Then** columns are grouped by rank
   correlation. Each column's group, and its correlation with guard_metric and with bb_width, are shown.
4. **Given** the candidate-bar population, **When** the residual score is computed, **Then** each column
   gets a strength after removing what bb_width and guard_metric explain. The width-sort baseline is
   shown on the same scale.

---

### User Story 3 - Combinations of indicators (Priority: P2)

The researcher learns whether any combination of indicators identifies good candidate bars better than
the band-width sort. Three levels:
- pairs of raw buy signals that must both fire;
- a shallow, readable rule model on all columns;
- a more flexible model on all columns.

Each is judged against the same model given only bb_width and guard_metric.

**Why this priority**: A single indicator rarely beats a strong sort; combinations are the user's
explicit question. They depend on Stories 1–2 and carry the highest risk of chance winners.

**Independent Test**: Run the combination screen. Check that:
- selection used only the early test windows;
- survivors were checked once on the late windows;
- every model was fitted on training data that ends at least one label horizon before its test window.

**Acceptance Scenarios**:

1. **Given** all pairs of raw buy signals, **When** the pairwise screen runs, **Then** each pair's lift
   on candidate bars is computed on the early windows. The top pairs by a pre-declared rule are then
   checked once on the late windows and reported with both halves.
2. **Given** the walk-forward folds, **When** the rule model and the flexible model run, **Then** each
   is fitted per fold on that fold's training window only and scored on its test window. Each is
   compared with: the same model given only bb_width and guard_metric; the width sort; and the current
   NNNC model's ranking where available.
3. **Given** the shallow rule model, **When** it is reported, **Then** its rules are readable (at most
   3 conditions per path), with the columns and thresholds named per fold, and with how often the same
   columns recur across folds.
4. **Given** any combination result, **When** it is reported, **Then** it counts as a candidate only if
   it beats the bb_width + guard_metric comparison in BOTH the early and the late windows.

---

### User Story 4 - Regime screen (Priority: P2)

The researcher learns whether any indicator, or combination, identifies a market regime. Three regime
definitions are tested, each forward-looking:
- **A — strategy outcome:** whether gbb-trigger trades opened in the next N days made money;
- **B — market direction:** the forward return of BTC and of the pair;
- **C — volatility:** the forward realised volatility of BTC and of the pair.

Indicators are scored on the pair's own values and on BTC's values.

**Why this priority**: The user wants a regime identification metric. Past regime gates failed, and
the reason (profitable trades in the "off" state) is what definition A targets directly. B and C
answer the more general question.

**Independent Test**: Run the regime screen for one definition and horizon. Check that the targets
are computed only from bars after the scored bar, and that the early-select / late-confirm rule is
applied.

**Acceptance Scenarios**:

1. **Given** each regime definition and horizon, **When** the target is built, **Then** each bar is
   labelled top or bottom tercile of the forward quantity. The tercile cut points are fixed on the
   training data of each fold. The target is used only as a target, never as an input.
2. **Given** the regime targets, **When** the single-indicator and combination screens run, **Then**
   they report the same measures and the same early/late rule as Stories 2–3, for both the pair's own
   and BTC's indicator values.
3. **Given** a regime result, **When** it is reported, **Then** it states how much the regime target
   overlaps with simply knowing the current trailing return or volatility. That way, "it predicts
   tomorrow's volatility because volatility persists" is visible rather than mistaken for discovery.

---

### Edge Cases

- **A strategy errors or produces a constant column** on the harvest data: excluded and listed with its
  reason. The screen still runs.
- **Two strategies produce the same indicator** (for example two Bollinger variants): both are kept,
  and the redundancy grouping shows them together.
- **An indicator is undefined during its warm-up**: those bars are excluded from that column's scores,
  not filled with a value that could look like a signal.
- **A raw signal never fires, or fires on almost every bar**, in a window: its lift is reported as not
  computable for that window, not as zero.
- **A pair has no data for part of a window** (for example XRP before 2023-07): the pair contributes
  only where it has data. Per-window pair counts are reported.
- **Few candidate bars in a window**: windows with fewer candidate bars than a pre-declared minimum are
  reported but excluded from the median and the stability share.
- **A forward regime target near the end of the data** cannot be computed. Those bars are dropped, not
  labelled.
- **Strategies with tuned parameters**: their raw signals use the strategy's current parameters, so
  thresholds were hyperopted on some past window. This is stated as a limitation for raw-signal lift.
  Indicator VALUES do not depend on thresholds.

## Requirements *(mandatory)*

### Functional Requirements

**Harvest**
- **FR-001**: The harvest MUST include every indicator strategy in the SimpleStrategies family, on both
  the plain and the Ehlers base. Trigger-only and breakout trigger arms are excluded, since they wrap
  registered triggers already scored in feature 021. Each strategy's indicator columns and its raw buy
  and sell signals are included.
- **FR-002**: Indicator values MUST be produced by calling each strategy's own indicator code, not by
  re-implementing its formula.
- **FR-003**: Raw signals MUST be taken before every filter the family applies: volatility, market
  state, regime, trend mode, slope, and entry/exit guards. No guard, filter or gate is applied anywhere
  in this feature.
- **FR-004**: Every harvested column MUST pass a truncation lookahead test on at least three pairs and
  three distant eras. A failing column is excluded and named in the report.
- **FR-005**: Strategies that error, or that produce a constant or all-missing column, MUST be listed
  with the reason. A column whose value never varies is not scored.
- **FR-006**: The harvest MUST also include the 24 model input columns and the gbb trigger flag, so
  that guard_metric, bb_width and the candidate population are defined exactly as for NNNC.
- **FR-007**: The harvest MUST be usable as a normal strategy for plotting, and MUST be cached per pair
  so that the screens never recompute it.

**Targets and windows**
- **FR-008**: The label targets MUST be the gbb_path BUY and SELL training labels with the ARO
  control's label parameters, built per pair exactly as for training.
- **FR-009**: Scores MUST be computed on each of the 15 walk-forward fold TEST windows. Medians and
  same-direction shares are reported over those windows; early = f00–f07, late = f08–f14.
- **FR-010**: The candidate population MUST be the bars where the gbb trigger fires.
  - The trigger is the registered one, including its own conditions.
  - It is not a guard: it defines which bars are asked about, and Story 2 also scores all bars.
- **FR-011**: Regime targets MUST be defined for definitions A, B and C at the horizons N = 12 h
  (48 bars, equal to the gbb_path label horizon), 3 days and 7 days.
  - Each target is the top vs bottom tercile of the forward quantity, with tercile cut points fixed
    per fold on training data.
  - Definition A is reported from two outcome sources, side by side:
    - (A1) every gbb-trigger bar's realised outcome from the per-bar outcome table (Open exit);
    - (A2) the recorded width-book backtest trades. A period with no width-book trade is "not
      computable" for A2, and per-window computable counts are reported.

**Scores**
- **FR-012**: For every column and target, the screen MUST report: rank strength and direction (AUC);
  a non-monotone association measure (mutual information); the per-window values; the median; and the
  share of windows agreeing in direction with the median.
- **FR-013**: For every raw signal, the screen MUST report the fire rate and the lift for each target.
- **FR-014**: The screen MUST group columns by rank correlation. It MUST report each column's
  correlation with guard_metric and with bb_width.
- **FR-015**: On candidate bars, the screen MUST report each column's residual strength after removing
  what bb_width and guard_metric explain, with the width-sort baseline on the same scale.

**Combinations**
- **FR-016**: The pairwise screen MUST score every pair of raw buy signals (combined with AND) by lift
  on candidate bars.
  - Selection is made on the early windows only, under a selection rule and survivor count fixed
    before running.
  - Survivors are scored once on the late windows.
- **FR-017**: The rule model (at most 3 conditions per path) and the flexible model MUST each be fitted
  per fold on that fold's training window only. A purge gap at least as long as the target's look-ahead
  horizon MUST separate training from test. Any scaling or selection MUST be fitted on the training
  window only.
- **FR-018**: Each multivariate model MUST be compared with the same model restricted to bb_width +
  guard_metric, and with the width sort. A combination counts as a candidate only if it beats the
  restricted model in both the early and the late windows.
- **FR-019**: The combination screens MUST run for the label targets and for every regime target.

**Regime-specific**
- **FR-020**: Regime screens MUST score each indicator on both the pair's own values and BTC's values
  at the same bar.
- **FR-021**: Each regime result MUST report a persistence baseline: the score of the trailing return
  (B) or trailing volatility (C) over the same N. A candidate counts only if it beats that baseline in
  both halves.

**Outputs and conduct**
- **FR-022**: All outputs MUST be written under the feature's research directory: one machine-readable
  file per screen, plus a human-readable report. A findings section is added to the project findings
  document.
- **FR-023**: The selection rules, survivor counts, horizons and minimum candidate counts MUST be
  recorded in a pre-registration committed before any late-window score is computed.
- **FR-024**: The feature MUST NOT change any strategy, framework code, default, or either live
  dry-run bot.
  - It runs no backtests.
  - It trains no neural model.
  - Multivariate fits run one fold per process.
- **FR-025**: Every result MUST be labelled as a screen (descriptive). The report MUST state that a
  candidate needs a backtest in a later feature before any P&L claim.

### Key Entities

- **Harvested column**: an indicator value or raw signal per pair per bar. Attributes: source strategy,
  kind (value / raw buy / raw sell), parameters used, causality status, exclusion reason if any.
- **Target**: the gbb_path BUY or SELL label, or a regime target (definition, horizon, BTC or pair).
  Per bar, it is the class, or "not computable".
- **Population**: all bars, or gbb candidate bars.
- **Window score**: one column × target × population × fold test window. Fields: strength, direction,
  association, lift, row count.
- **Screen result**: the aggregate of window scores (median, same-direction share, early/late halves),
  plus redundancy group and residual strength.
- **Combination**: a signal pair, or a fitted model per fold. Fields: its score, the restricted
  comparison, early/late halves, and its readable rules.
- **Pre-registration**: the fixed selection rules and thresholds, committed before late-window scoring.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the indicator strategies in scope are either harvested or listed with a reason.
  Every harvested column has a recorded causality result.
- **SC-002**: The instrument check passes: guard_metric and bb_width are each in the top 5 columns for
  the BUY label on all bars, by median strength over the 15 test windows. If it fails, the feature
  stops and reports the defect instead of findings.
- **SC-003**: Every column has a score on both populations and both label targets for every test
  window where it is computable. Every exclusion from a median is explained.
- **SC-004**: The report answers each question below with a named list. An empty list is a valid
  answer.
  - (a) Which columns beat the width sort on candidate bars in both halves?
  - (b) Which columns retain residual strength beyond bb_width + guard_metric in both halves?
  - (c) Which signal pairs survive late confirmation?
  - (d) Does any multivariate model beat the restricted model in both halves?
  - (e) For each regime definition and horizon, which columns or models beat the persistence baseline
    in both halves?
- **SC-005**: No late-window score is computed before the pre-registration commit (verifiable from
  commit order).
- **SC-006**: Neither live dry-run bot is interrupted; both are alive at the end of every run step.
- **SC-007**: A rerun on the same data reproduces every reported number exactly.

## Assumptions

- **Data:** the gbb_path labels use the ARO control's label parameters (the label NNNC is trained on).
  The data is the same walk-forward fold plan and pair whitelist as the NNNC controls.
- **Raw-signal parameters:** raw signals use each strategy's current parameters, without re-tuning.
  Indicator values are the primary object; raw-signal lift is secondary.
- **Population:** "gbb candidate bars" means bars where the registered gbb trigger fires. This is the
  population on which the model behaves like a width sort.
- **Persistence:** the early/late fold split (f00–f07 vs f08–f14) is the persistence check required by
  the constitution's Principle II. It spans 2021–2026, so the halves are temporally distant.
- **No P&L claim:** this is a screen. Under Principle III, association with a label or regime decides
  nothing about acceptance. It only nominates candidates for later backtested features.
- **Combination model sizes:** the rule model is limited to at most 3 conditions per path, for
  readability. The flexible model uses standard settings fixed in the pre-registration, not tuned per
  fold beyond what the training window alone supports.
- **Regime horizon vs label horizon:** the regime horizon is separate from the label horizon
  (`TrainingConfig.HORIZON` = 48 bars = 12 h). The 12 h regime horizon is a bridge: regime A at 12 h
  should behave like a smoothed gbb_path label. The 3 d and 7 d horizons are the regime question proper.
- **Regime targets:** they use the same 15m bars, aggregated forward over N days. BTC values come from
  the BTC/USDT series on the same exchange.
- **Load:** the work is CPU-light compared with training, so it can run while the live bots record.
