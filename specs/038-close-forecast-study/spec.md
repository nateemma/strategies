# Feature Specification: Close-Price Forecast Study

**Feature Branch**: `038-close-forecast-study` (spec directory; work is committed to master)

**Created**: 2026-10-19

**Status**: Draft

**Input**: User description: "New study. I want to investigate time series prediction, specifically the
close column. Review the code in TSPredict/ and the associated utilities and test code in utils/. The
goal is to find the best method (if there is one) that can predict the close data N candles ahead, and
the most viable value for N (where N cannot be so small that we cannot trade on it)." Follow-up
(2026-10-19): restore the test data so it can be used for exploration, then develop the study through
the spec process. Question raised: should the predicted value be gain or close? Answer (Assumptions):
the N-candle return of close, from which a close level is derived.

## Background *(evidence; review of TSPredict/ and utils/ on 2026-10-19)*

- **What TSPredict predicts.** Its `gain` is the percentage change in close over the last `lookahead`
  candles (6 by default), and the model is trained on that value shifted forward, i.e. the return over
  the next 6 candles. Close is never predicted directly. Strategy outcomes were judged only by backtest,
  which mixes forecast quality with entry and exit thresholds; forecast quality itself was never
  measured.
- **The forecaster library** offers about 25 forecasters behind one interface: statistical models fitted
  to each short window, and learned regressors trained on features. The review found defects that would
  distort any comparison:
  - five statistical forecasters (Exponential, SimpleExp, ETS, StateSpace, ARIMA) return the value one
    candle further ahead than requested; Holt and AR are correct;
  - ARIMA hides every failure and returns zeros, which looks like a genuine "no move" forecast;
  - SimpleExponential crashes outright on the installed statistics library (found while planning);
  - with detrending on, the learned regressors add back the trend of the training window to a
    prediction for a different window;
  - the FFT-extrapolation forecaster (the one the test script enables) passes frequency coefficients to
    a regressor trained on returns, so its output has no meaning;
  - two TS_Wavelet methods call a helper with the wrong number of arguments and would fail if reached.
- **The test script** plots one forecaster against shifted data from 993 normalised gain values of one
  pair. It reports no score and no baseline. At 993 rows the standard error of a hit rate is about
  1.6 percentage points, so it cannot separate a weak forecast from a coin flip. The data file was
  deleted and has been restored from history for exploration.
- **Related evidence.**
  - 037: the best indicator model reached about 0.6 AUC on the buy label, a much easier target than a
    return.
  - 035: no new price-derived input beat the existing sizer.
  - Lesson in project memory: an unlearnable target is a data problem; more model types on the same
    inputs do not move the ceiling.

## User Scenarios & Testing *(mandatory)*

The user is the researcher. They want to know whether any forecasting method predicts where close will
be N candles ahead well enough to trade after costs, and which N is most usable. The outcome is a
verdict per (method, N), not a strategy.

### User Story 1 - A forecast library that does what it claims (Priority: P1)

The researcher can trust that every forecaster in the library forecasts exactly the horizon requested,
fails loudly rather than silently, and does not mix data from one window into another.

**Why this priority**: Every comparison in the study reads these forecasters. A forecaster that answers
a different horizon, or returns zeros on failure, makes any ranking meaningless.

**Independent Test**: Feed each forecaster a synthetic series whose future is known exactly (a straight
line, a sine wave, a constant) and check the returned value against the known value at the requested
horizon.

**Acceptance Scenarios**:

1. **Given** a straight-line series and a horizon of h candles, **When** any trend-capable forecaster is
   asked for h, **Then** it returns the line's value h candles ahead, not h + 1.
2. **Given** a series that makes a forecaster fail to fit, **When** it is asked to forecast, **Then** the
   failure is reported and counted. It is never returned as a zero forecast.
3. **Given** a forecaster with detrending on, **When** it forecasts a window, **Then** only that
   window's own trend is restored, never another window's.
4. **Given** the FFT-extrapolation forecaster and the TS_Wavelet helper calls, **When** they are used,
   **Then** they either produce a correct forecast under the same synthetic checks or are marked
   unavailable with the reason.
5. **Given** the restored test data, **When** the researcher runs the existing plot script, **Then** it
   runs and plots without error.

---

### User Story 2 - Baselines and the cost line per horizon (Priority: P1)

The researcher sees, for each horizon N, how far close typically moves, what a trade costs, and how the
two simplest forecasts score: "no change" (random walk) and the historical mean return.

**Why this priority**: These set the bar. A method that does not beat "no change" has no information. A
horizon whose typical move is smaller than the round-trip cost cannot be traded even with a good
forecast. This story alone answers which values of N are worth testing.

**Independent Test**: Run the baselines on all pairs and folds and confirm that "no change" scores
exactly zero skill by construction. Confirm that the typical move grows with N.

**Acceptance Scenarios**:

1. **Given** the 10 pairs and every fold, **When** the baseline report runs for each N in
   {1, 2, 4, 8, 16, 32, 48, 96}, **Then** it shows the typical absolute N-candle move, the round-trip
   cost at each slippage level, and the ratio between them, per pair and pooled.
2. **Given** the "no change" forecast, **When** it is scored, **Then** its skill against itself is
   exactly zero. Any other value fails the instrument.
3. **Given** the cost line, **When** a horizon's typical move is below the round-trip cost, **Then** the
   report marks it as untradeable by construction, before any model is read.

---

### User Story 3 - Statistical forecasters scored against the baselines (Priority: P2)

The researcher sees the corrected statistical forecasters (AR, Holt, ETS, exponential smoothing,
ARIMA, Theta), each refitted on a short trailing window, scored per N and per half against "no change"
and against the cost line.

**Why this priority**: These are the methods TSPredict was built on and are cheap to run. A clear result
here either confirms or rules out the original approach without any training.

**Independent Test**: Run one forecaster over one fold and confirm that every forecast used only candles
before the forecast time, and that the number of forecasts equals the number of eligible candles.

**Acceptance Scenarios**:

1. **Given** a corrected forecaster and a horizon N, **When** it is scored, **Then** the report shows
   skill against "no change", rank correlation with the realised return, hit rate, and the cost check
   for each half and each pair.
2. **Given** a forecaster that fails on some windows, **When** it is scored, **Then** the failure rate is
   reported, and the forecaster is not ranked if more than the pre-registered share of windows failed.

---

### User Story 4 - Learned forecasters on lagged returns and indicators (Priority: P2)

The researcher sees learned models (boosted trees) trained walk-forward to predict the N-candle return
from lagged returns and from the 037 indicator set, scored exactly like the statistical tier.

**Why this priority**: This is the strongest price-only method available here and the version
TSPredict's main path uses. If it fails at every N, the limit is the information in the inputs, not the
method.

**Independent Test**: Train on one fold and confirm that the gap between the end of the training labels
and the first scored candle is at least N candles. Confirm that a model trained on shuffled targets
scores no better than "no change".

**Acceptance Scenarios**:

1. **Given** a fold, **When** a learned model is trained, **Then** its training labels end at least N
   candles before the fold's first scored candle, and all scaling or selection is fitted on the training
   span only.
2. **Given** shuffled training targets, **When** the model is scored, **Then** its skill is
   indistinguishable from zero. Otherwise the instrument fails and no learned result is reported.

---

### User Story 5 - Decomposition features (Priority: P3)

The researcher sees whether wavelet and frequency decompositions of the trailing price window, added as
inputs to the learned tier, improve on Story 4.

**Why this priority**: This is the idea behind the TS_Coeff and TS_Wavelet strategies, tested in its
sound form. It only matters if Story 4 shows some skill or comes close.

**Independent Test**: Truncate the data at an earlier candle and recompute the decomposition inputs.
Every value up to the cut must be identical.

**Acceptance Scenarios**:

1. **Given** decomposition inputs, **When** the truncation test runs, **Then** any input that changes
   at or before the cut is excluded as lookahead and named.
2. **Given** the learned model with and without decomposition inputs, **When** both are scored, **Then**
   the report shows the paired difference per N and per half.

---

### User Story 6 - Verdict per method and horizon (Priority: P1)

The researcher gets one table: for each (method, N), PASS, FAIL or INCONCLUSIVE under a rule written
down and committed before any result is read. It also names the most viable N, or states that none
qualifies.

**Why this priority**: This is the question the study exists to answer.

**Independent Test**: Confirm that the pass rule's commit predates every result file, and that the
verdict tool refuses to run otherwise.

**Acceptance Scenarios**:

1. **Given** the committed pass rule, **When** the verdict runs, **Then** each (method, N) receives
   exactly one verdict with the numbers that decided it.
2. **Given** results newer than the rule, **When** the rule file has uncommitted changes, **Then** the
   verdict refuses to run.
3. **Given** no (method, N) passes, **When** the report is written, **Then** it says so plainly and
   records it as a completed negative result.

### Edge Cases

- **Zero-volume candles.** On binance.us thin pairs have many candles with no trades, and their close
  repeats the previous one. These candles are still scored, but the cost check is also reported with
  them excluded, because no trade could have filled there.
- **Gaps in the data.** A window that spans a missing stretch of candles is not forecast. Excluded
  windows are counted.
- **Pairs that start late.** XRP has no data from 2021-01 to 2023-07. Pooled results weight pairs by the
  candles they actually have, and per-pair results are shown as well.
- **Overlapping targets.** For N > 1, consecutive N-candle returns overlap, which inflates naive
  significance. Uncertainty is computed with a method that respects this, such as resampling blocks of
  at least N candles or one sample every N candles.
- **Outlier moves.** A few extreme candles can dominate squared-error skill. Rank-based metrics and the
  cost check are reported alongside it, and the pass rule does not rely on squared error alone.
- **Forecaster failures.** These are counted per (method, N). A method that fails often is reported, not
  silently scored on the windows it survived.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST predict the N-candle return of close (close N candles ahead divided by close
  now). Any predicted close level MUST be derived from that return and MUST NOT be modelled directly.
- **FR-002**: Horizons MUST be N ∈ {1, 2, 4, 8, 16, 32, 48, 96} candles of 15 minutes.
- **FR-003**: Data MUST be the 10 NNNC pairs at 15m from binance.us, 2021 onward, read from the stored
  market data, using the existing 15 walk-forward folds and their early and late halves.
- **FR-004**: Every forecast MUST use only data available at the forecast time. Learned models MUST
  leave a gap of at least N candles between the end of their training labels and the first scored
  candle. All fitted preprocessing MUST be fitted on the training span only.
- **FR-005**: The forecaster defects listed in Background MUST be fixed or the forecaster marked
  unavailable, each verified by a known-answer test, before any of its results are read.
- **FR-006**: The study MUST score two baselines, "no change" and the trailing historical mean return,
  with the same procedure as every other method.
- **FR-007**: For each (method, N, half, pair) the study MUST report:
  - skill against "no change";
  - rank correlation between forecast and realised return;
  - hit rate on direction;
  - the cost check: long-only trades taken when a forecast is in the top decile of that method's
    recent forecasts, entered at the next candle's open and held N candles, one position per pair at a
    time, against round-trip cost at 0, 2.5, 5 and 10 bps of slippage per side on top of exchange fees,
    and against random entry at the same cost. Shorts cannot be traded on the venue, so short-side
    figures are descriptive only.
- **FR-008**: The study MUST report the break-even horizon: the smallest N at which the cost check is
  positive in both halves at the pre-registered slippage level.
- **FR-009**: The pass rule MUST be written and committed before any non-baseline result is read. The
  verdict tool MUST refuse to run if the rule file has uncommitted changes or is newer than the
  results.
- **FR-010**: Acceptance MUST rest on the cost check in both halves. Skill, rank correlation and hit rate
  are reported as descriptions and MUST NOT decide a PASS on their own.
- **FR-011**: Each learned tier MUST pass a shuffled-target null check before its results are reported.
- **FR-012**: Decomposition inputs MUST pass a truncation test. Any input that fails is excluded and
  named.
- **FR-013**: The existing plot script MUST run on the restored test data and remain an exploration tool
  only. No verdict may be drawn from it.
- **FR-014**: The study MUST NOT run strategy backtests. A passing forecast is a candidate for a later
  feature that tests it as an NNNC entry filter or sizer.
- **FR-015**: No pair-specific settings are allowed. Every method uses the same configuration on every
  pair.

### Key Entities

- **Horizon (N)**: A forecast distance in 15-minute candles. Each horizon has a typical move size and a
  cost ratio.
- **Forecast record**: For one pair, candle, method and horizon: the forecast return, the realised
  return, the fold, the half, and whether the candle traded.
- **Method**: A forecaster with its tier (baseline, statistical, learned, learned with decomposition), its
  inputs and its failure count.
- **Pass rule**: The committed thresholds and decision logic that turn scores into verdicts.
- **Verdict**: PASS, FAIL or INCONCLUSIVE per (method, N), with the numbers that decided it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every forecaster that is used passes its known-answer test at every requested horizon: the
  value it returns for horizon h equals its underlying model's own h-step forecast to numerical
  precision (a one-candle shift fails), it forecasts a constant series as that constant, and the
  trend models forecast a straight line exactly.
- **SC-002**: "No change" scores exactly zero skill against itself, and shuffled-target learned models
  score within noise of zero. If either fails, the study stops and nothing further is reported.
- **SC-003**: The baseline report covers all 10 pairs, all 15 folds and all 8 horizons, with fewer than 1%
  of eligible candles unscored for reasons other than data gaps.
- **SC-004**: Every (method, N) in scope receives exactly one verdict under a rule committed before its
  results existed.
- **SC-005**: The final report names the most viable horizon with its cost-check margin in both halves,
  or states that no horizon qualifies. Either outcome is recorded as a completed result.
- **SC-006**: The researcher can rerun the full baseline and statistical tiers in under one hour, so
  later variants can be screened cheaply.

## Assumptions

- **Gain or close.** The predicted quantity is the N-candle return of close. Close levels differ by orders
  of magnitude across pairs, drift over time, and are almost perfectly predicted by "same as now", so
  error on the level says nothing about tradeable skill. TSPredict's `gain` is this same return, measured
  backward and in percent. Shifting it forward N candles gives the target. Close-level error can be
  derived from the return forecast for display.
- **Costs.** Exchange fees follow the existing backtest configuration. Slippage uses the repo's standard
  grid of 0, 2.5, 5 and 10 bps per side. The pass rule names one slippage level.
- **Timeframe.** 15m only. Longer timeframes are covered by the larger N values (96 candles = 1 day).
- **Pairs and folds.** These are the same 10 pairs and 15 folds as the NNNC studies, so results are
  comparable with them. Data after the last fold (2026-09-07 to the present, about 6 weeks) is scored
  as a descriptive forward window, too short for a verdict.
- **Persistence.** There is no distant-era data (the binance.com data directory is empty), so the early
  and late halves are the persistence test and every PASS is PROVISIONAL, as in all studies since 023.
- **Test data.** `utils/test_data.npy` (993 normalised gain values from one pair, restored from git) is for
  exploration only.
- **Out of scope.** Neural sequence models (LSTM, transformer), non-price inputs (funding, order flow),
  and any strategy change. If price-only methods fail at every horizon, the next lever is new inputs, in
  a later feature.
- **Compute.** Runs are sequential, one training job at a time, as for all studies here.
