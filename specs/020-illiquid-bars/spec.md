# Feature Specification: Illiquid Bars — What Data This Project Trains and Trades On

**Feature Branch**: `020-illiquid-bars` (spec directory; commits to master)

**Created**: 2026-09-20

**Status**: Draft

**Input**: While reviewing the study's scaling approach we found that a raw feature was passing
through unscaled because its pooled interquartile range was exactly zero. The cause is that a
large fraction of the 15m binanceus bars are gap-fill: freqtrade writes a candle with
open=high=low=close(previous) and volume=0 when no trades occurred. Measured zero-volume share
since 2022-01-01: AAVE 55.0%, ZEC 54.1%, BCH 53.7%, NEAR 47.6%, DOT 39.4%, SUI 39.4%,
LINK 25.3%, LTC 23.0%, AVAX 22.2%, XRP 6.7%, SOL 3.8%. The author independently observes the
same flat plateaus in a LIVE dry run (ARB, INJ, ONDO, HYPE), which establishes that this is
genuine venue illiquidity at this timeframe, not a download artefact, and therefore cannot be
fixed by re-downloading.

## Scope Statement

This feature decides **what data the project trains and trades on**, supported by measurement.
It is deliberately small and gated: three strands, one measurement phase each, and an explicit
kill criterion that stops it early if the contamination turns out not to matter. It is NOT an
open-ended data-quality programme.

Strand B touches live behaviour and is the only part with money attached; it is diagnosis and
reporting only. No strategy change ships under this feature without a separate decision.

## Prior context

- Feature 018/019 results are all measured on this data. The per-pair conclusions that lean on
  the thin pairs (notably "the gbb edge is structurally high-vol-alt (ZEC)" and "the universe
  has no lever") were measured on a universe where ZEC is 54% synthetic bars.
- The execution-side relative is already recorded: a volume filter unmasked phantom returns in
  earlier work. That finding was about fills; this one is about the training data and the
  exit path.
- Class-balance and label autocorrelation on every forward-looking label are very high
  (persistence MCC 0.76-0.87), which long runs of identical filled bars would produce.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Does contamination change any conclusion? (Priority: P1)

As the strategy author, I want a sample-validity rule that refuses to emit a training or
evaluation sample whose feature window plus label horizon is mostly gap-fill, and I want the
current best configuration re-run with and without it, so I know whether the study's
conclusions rest on synthetic bars.

**Why this priority**: it is the cheapest strand, it gates the other two, and it either
invalidates or confirms everything measured so far.

**Independent Test**: the rule is implemented behind a config field that enters the config id;
the current best configuration is run both ways on the same splits and seeds; the report states
whether the headline metric moves by more than the seed noise and whether the architecture
ranking survives.

**Acceptance Scenarios**:

1. **Given** a validity threshold, **When** samples are built, **Then** every emitted sample
   has at least that fraction of real (volume > 0) bars across its window and horizon, and the
   count of dropped samples is recorded per pair and per window on the run row.
2. **Given** runs with and without the rule, **When** they are compared, **Then** the report
   gives the delta in headline MCC per window with seed standard deviations, and the
   architecture ranking under both.
3. **Given** the rule is active, **When** the training set is summarised, **Then** the report
   shows how the per-pair composition of the training set changes (measured: an 80%-real
   threshold keeps 95% of SOL's samples, 47% of LINK's and 19% of ZEC's).

---

### User Story 2 - Is the live exit path evaluating stale candles? (Priority: P1)

As the strategy author, I want to know whether entries, the ATR stop, the stop-loss grace and
the exit signals behave correctly when the current candle is a zero-volume fill, because a stop
evaluated against a price that has not updated fires late or not at all.

**Why this priority**: this is the only strand with direct money consequences, and the author's
running dry run can answer it with data that already exists.

**Independent Test**: a written trace, for each decision point in the live path, of what it
reads and what a zero-volume candle makes it read; plus a count from the dry-run database of
decisions taken on zero-volume candles and of exits whose trigger bar was a fill.

**Acceptance Scenarios**:

1. **Given** the live code path, **When** each decision point is traced, **Then** the report
   states for entries, custom stoploss, stoploss grace, trailing behaviour and exit signals
   whether a zero-volume candle can cause a wrong or delayed decision, citing file and line.
2. **Given** the dry-run trade record, **When** exits are attributed to their trigger bar,
   **Then** the report gives the share of exits whose trigger bar had volume 0, by pair.

---

### User Story 3 - Is 15m on this venue the right data at all? (Priority: P2, conditional)

As the strategy author, I want the same classifier screen run on 1h and on a liquid-only subset
so I can decide whether to change timeframe or universe rather than filter samples.

**Why this priority**: it is the most expensive strand and the most disruptive to production,
so it runs only if Strand A shows contamination matters.

**Independent Test**: the screen's verdict table for 15m (all pairs), 15m (liquid subset), and
1h (all pairs), on the same label family and with the same promotion rule.

**Acceptance Scenarios**:

1. **Given** the three arms, **When** they are screened, **Then** the report compares them on
   the same metric and states the sample count and zero-volume share of each.

### Edge Cases

- A pair whose zero-volume share makes it contribute almost nothing after filtering: recorded,
  not silently dropped; the report names it and its surviving sample count.
- A window that straddles a long fill run at a window boundary: the rule is evaluated on the
  union of the feature window and the label horizon, so such samples are refused.
- Filled bars must NOT be deleted from the frames: windowing and forward-looking labels need
  contiguous bars. The rule selects samples, it does not remove rows.

## Requirements *(mandatory)*

- **FR-001**: A sample-validity rule MUST be available as an experiment-config field that
  enters the config id, with the default preserving today's behaviour so existing rows stay
  valid and comparable.
- **FR-002**: The rule MUST be evaluated over the union of the feature window and the label
  horizon of each candidate sample, and MUST refuse samples below the configured fraction of
  real bars.
- **FR-003**: Filled bars MUST remain in the frames; only sample selection changes.
- **FR-004**: Every run row MUST record the zero-volume share of its train and test rows, the
  validity threshold in force, and the number of samples dropped by it per pair.
- **FR-005**: Strand A MUST compare the current best configuration with and without the rule on
  identical splits and seeds, and report per-window deltas against seed standard deviation.
- **FR-006**: Strand B MUST produce a written trace of the live decision points with file and
  line references, and a count of dry-run exits whose trigger bar had volume 0.
- **FR-007**: Strand B MUST NOT change strategy behaviour; any fix is a separate decision.
- **FR-008**: Strand C runs only if the Strand A gate opens (see below), and MUST use the same
  screen and promotion rule as feature 019.
- **FR-009**: No artefact of this feature may compute or report P&L; Strand B reports counts and
  code behaviour, not returns.
- **FR-010**: The report MUST state explicitly which earlier conclusions are re-opened by the
  result, in particular the per-pair findings that rest on the thin pairs.

## Success Criteria *(mandatory)*

- **SC-001**: The validity rule is implemented, tested, and recorded on every run row.
- **SC-002**: Strand A answers, with numbers, whether the headline metric and the architecture
  ranking survive the rule.
- **SC-003**: Strand B answers, for each live decision point, whether a zero-volume candle can
  cause a wrong or delayed decision.
- **SC-004**: A recorded decision on whether Strand C runs, with its reason.
- **SC-005**: The report names every earlier conclusion that the result re-opens.

## Kill criterion (gate after Strand A and B)

If the validity rule moves the headline metric by less than the seed standard deviation AND
leaves the architecture ranking unchanged, AND the live trace shows the exit path already
ignores or correctly handles stale candles, then the feature closes with "apply the rule,
document it, carry on" and Strand C does NOT run.

## Out of scope

- Changing any strategy's live behaviour (diagnosis only).
- Downloading new data or adding a venue.
- Re-running features 018/019 in full; only the current best configuration is re-measured.
- The scaler-scope question (pooled versus per-pair normalisation), which is recorded as a
  separate open question.

## Assumptions

- The immediate fixes (degenerate-IQR scaler guard, zero-volume share on every row and in the
  data inventory) land before this feature starts and are not part of it.
- "Real bar" means volume > 0. Bars with volume > 0 but OHLC all equal are treated as real.
- The dry-run database is available for Strand B; if it is not, Strand B reports the code trace
  alone and says so.
