# Feature Specification: New-Input Screen for NNNC

**Feature Branch**: `035-new-input-screen` (spec directory; work is committed to master)

**Created**: 2026-10-15

**Status**: Draft

**Input**: User description: "New-input screen for NNNC (feature 035). [...] Goal: find whether
information the model has never seen (non-OHLCV, or cross-pair/market state) improves the ranking
of width-book trades beyond bb_width and the current model's p_buy, screening cheaply before any
training. [...] Candidate input families (confirm in spec): (1) order flow [...] (2) perp
positioning [...] (3) market state as model input [...] (4) spot-perp basis [...] S0 data [...] S1
no-training screen on the early width-book trades [...] S2 retrain ARO with each surviving input
added [...] S3 backtest the S2 winner as the cs_aro-style sizer on all 15 folds [...]" (full text
in the conversation of 2026-10-15)

## Background *(evidence; research/control_sizer/report.md, research/label_screen/report.md, research/model_calibration/findings.md)*

**The model's measured value is as a sizer inside the width book.**
- A model-free width book matches every NNNC entry book.
- With the current ARO weights as a capital-neutral sizer on that book (`cs_aro`), the result is
  59.95 pp over 15 folds against the width book's 51.44 on the identical 930 trades. The paired
  test is IMPROVED (p 0.0007), and the gain is positive in both halves. This is PASS
  (PROVISIONAL), and the late half was not blind.
- The ranking behind it shows as a 0.100 agree − disagree gap per trade on the 412 early-half
  width-book trades (034's control at the 0.5 share).

**Changing what the model learns from has not improved that ranking:**
- 032 (training rows) and 033 (a width-relative label) failed;
- 034 tried seven other label types, and all of them ranked width-book trades worse.

**Changing how the model reads the same inputs has not improved it either.** Six architecture
variants, an 85-indicator set, a CNN on raw candles, and session and Fibonacci features all added
nothing. The model hit an information ceiling: the remaining lever is information it has never
seen.

**Inputs already tested, and how:**

| input | test | result |
|---|---|---|
| funding rate | feature screen at 12 h; entry filter | no signal (rank corr −0.004); the sign flips at short holds; not a feature (`regime/FUNDING_GATE.md`) |
| BTC regime | entry gate | over-filters dip entries; helps neither family |
| BTC vol / ADX | alt-return model (btc_corr Exp 2) | small information gain (IC +0.012); never fed to NNNC |
| market phase / breadth | entry gate (001) | no gate passes; reverting states rank best in every window (a sizing signal) |
| cross-pair volatility regime | reversion gate (026) | cut reversion's late loss by 63%; not adopted as a gate; never a model input |

**Inputs never tested** (data availability checked 2026-10-15):
- **Order flow.** Binance's 15-minute spot candles carry trade count and taker-buy volume.
  - Binance.US serves them live, and its history starts at each pair's listing: the same coverage
    as the price data the model already trains on.
  - Binance global has them from 2021-01 for every pair, and its volume also covers Binance.US's
    gap-filled bars (about 54% of ZEC's 15-minute bars are filled, with zero volume). Its live API
    is blocked from this location (HTTP 451), so it is historical only.
- **Perpetual-futures positioning:** open interest, top-trader and global long/short ratios, and
  taker long/short volume, at 5-minute resolution. It is historical only, and only from
  2021-12-01. A live source would be another venue.
- **Spot–perp basis** (premium index, 15 minutes, from 2021-01): close to funding, which carried
  no signal.

**Harness lessons carried from 032–034 and the control-sizer study:**
- fold strategies scored outside a backtest need their saved parameters loaded, one fold per
  process;
- trade identity is (pair, open date, close date);
- the liquidity cap absorbs stake multipliers: a 2.0 ratio realised about 1.6;
- the fold runner runs from the repo root, and skips folds that already have a record;
- the control's records predate the held-pair add default (they are add-off);
- per-fold scalers are mandatory (constitution XI). Planning resolved this for new inputs by
  giving them no fitted artefact (FR-013).

## Clarifications

### Session 2026-10-15

- Q: Which input families enter the S1 screen? → A: All four: order flow, perp positioning,
  market state and basis (17 inputs, FR-002).
- Q: What role does live availability play? → A: It is recorded only. Each input's live source
  (or its absence) is recorded and reported, and a winner without one is flagged. It does not
  stop an input at S1 or S2.
- Q: How are positioning inputs trained when the training window starts before 2021-12-01? → A:
  With a neutral value plus a missing flag. Bars without positioning data carry the input's
  neutral value and a missing indicator. Training rows stay identical to the control's, so the
  change is one variable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Screen new inputs without training (Priority: P1)

The researcher builds each candidate input causally for every pair and bar, and scores it against
the early-half width-book trades. The question is whether each input ranks those trades beyond
what bb_width and the current model's buy probability already capture. No model is trained and
no backtest is run.

**Why this priority**: It covers every candidate input for the cost of data preparation alone. It
decides which inputs deserve training, and it answers the request on its own.

**Independent Test**: Run the screen. The result is a table with one row per input:
- its coverage;
- its correlation with bb_width;
- its correlation with the control's buy probability;
- its incremental rank correlation with trade return (after bb_width and buy probability), per
  fold and pooled;
- a pre-registered survivor list.

**Acceptance Scenarios**:

1. **Given** the committed pre-registration, **When** S0 runs, **Then** every input has a recorded
   coverage per pair and fold, and passes a truncation test. The truncation test computes the
   input from history cut at a sample of bars and checks that the value at each cut bar equals the
   full-history value.
2. **Given** S0, **When** S1 runs, **Then** every input is scored on the same 412 early width-book
   trades that 033, 034 and the control sizer used. The trade set and the control's 0.100 gap must
   reproduce.
3. **Given** S1 results, **When** survivors are selected, **Then** the pre-registered rule is
   applied mechanically, and its output is committed before any training.

---

### User Story 2 - Train the survivors into the model (Priority: P2)

Each surviving input is added on its own to the control model's inputs. Everything else is
unchanged: architecture, label, sequence length, exits and seed. The model is trained on the early
folds and judged with the same separation instrument as the control sizer.

**Why this priority**: The model learns interactions a univariate screen cannot see, and an input
that ranks on its own may add nothing once the model has it. Training is the real test.

**Independent Test**: Per surviving input, the early-half agree − disagree gap at the 0.5 share,
next to the control's 0.100.

**Acceptance Scenarios**:

1. **Given** a survivor, **When** it is trained, **Then** its weights live under their own model
   key on f00–f07, its new input has no fitted artefact (causal, pre-normalised) while the
   original columns use the control's scaler unchanged, and no existing key, scaler, weight file
   or record changes.
2. **Given** trained survivors, **When** S2 is judged, **Then** an input passes only if its early
   gap beats the control-at-share 0.100. The single best passer goes to S3.
3. **Given** no S2 pass, **Then** nothing is backtested.

---

### User Story 3 - Backtest the winner as the sizer (Priority: P3)

The S2 winner is trained on all 15 folds and backtested as the control-sizer arm with only its
weights changed. It is judged against the current sizer `cs_aro`, the width book `w_aro` and the
control `wf_aro_g24`.

**Why this priority**: Better separation is not edge. Only a backtest under the wallet and
liquidity limits shows return, and the late half is read only here.

**Independent Test**: A judge_one verdict against each comparator, with early and late gains.

**Acceptance Scenarios**:

1. **Given** an S2 winner, **When** S3 runs, **Then** no late-half P&L of any other input is ever
   read.
2. **Given** S3 results, **Then** the winner passes only if judge_one passes against `cs_aro`
   (the sizer it would replace) and against `w_aro`. A margin under 1 pp needs the paired seed-2
   repeat.
3. **Given** any pass, **Then** it is labelled PASS (PROVISIONAL), and the report records whether
   the input's live data is reachable. Nothing is adopted.

### Edge Cases

- **Gap-filled bars** (zero volume, flat OHLC on Binance.US): venue-native order-flow inputs are
  undefined there. They are marked missing by a declared rule, never read as a real zero.
- **Missing source data** (a pair absent from a source, a day missing from the archive, or
  positioning before 2021-12-01): coverage is reported per pair and fold. An input with under 90%
  coverage on the early width-book trades is INCONCLUSIVE at S1.
- **Timestamp alignment:** an archive bar stamped at its open covers up to its close. An input
  for signal bar t may use only source bars that closed at or before t's close. 5-minute data is
  aggregated so the last sub-bar used closes at t's close.
- **Market-state inputs pool several pairs.** Each pair's value at t uses only bars at or before
  t, and a pair with a filled bar is excluded from that bar's pool.
- **Multiple testing:** with 17 inputs scored, roughly one may pass a 2-standard-error
  cutoff by chance. S1 only filters; the S2 training test is the real check.
- **Downloaded archives are untrusted:** stored in their own directory, parsed as data, never
  executed, and never committed.
- **Low-memory kill mid-chain:** the chain is resumable and is not restarted without the user's
  say-so.

## Requirements *(mandatory)*

### Functional Requirements

**Inputs and pre-registration**

- **FR-001**: Before any S1 score is computed, a pre-registration under `research/new_inputs/`
  MUST be committed. It covers every input's exact definition, source, alignment rule, window
  length, missing-value rule, the S1 statistics, cutoffs and survivor rule, the S2 and S3 rules,
  and the fold sets. Later changes are amendments with a reason, made before the affected results
  are read.
- **FR-002**: The candidate inputs are fixed in the pre-registration, from the families confirmed
  in Clarifications. Their window lengths are declared once and never tuned on screen results.
  - **Order flow (6):** from Binance global, taker-buy share, trade-count ratio to its trailing
    median, and average trade size ratio to its trailing median; from Binance.US, taker-buy share
    and trade-count ratio; and the global ÷ US volume ratio.
  - **Positioning (4):** open-interest change, open-interest change combined with price change (the
    flush proxy), the top-trader long/short ratio change, and the taker long/short volume ratio.
  - **Market state (5):** cross-pair median volatility ratio (026's volreg), breadth (026's: the
    share of pairs with close above their 768-bar EMA), BTC 24 h volatility, BTC ADX, and the
    pair's 24 h return relative to the basket.
  - **Basis (2):** the premium index level and its change.

  That makes 17 inputs, defined exactly in research R6.
- **FR-003**: Funding as a feature, any entry gate, label changes, architecture changes, new pairs
  and paid data are out of scope.

**Data (S0)**

- **FR-004**: Source data MUST be fetched from free public archives (and, for the live-feasibility
  record, the venue's public endpoint). It is cached outside the repository in its own directory,
  parsed as data only, and never committed.
- **FR-005**: Every input MUST be causal. Its value at signal bar t uses only source data closed
  at or before t's close. This MUST be proven by a truncation test on a sample of at least 200
  bars per pair: the value from history cut at t equals the full-history value.
- **FR-006**: Coverage MUST be reported per input, pair and fold, both as the share of bars and
  as the share of early width-book trades with a valid value. The missing-value rule (gap-filled
  bars, absent days, pre-history) is declared in the pre-registration.
- **FR-007**: For each input, the live-feasibility record MUST state whether a live source is
  reachable from this location, and which one. The record is informational: it does not stop an
  input at any stage, and the report flags any winner without a live source.

**Screen (S1, early half only)**

- **FR-008**: S1 MUST use the identical early width-book trades as 033, 034 and the control
  sizer (f00–f07, 24 h-grace simulator). Before any input is read, it MUST reproduce the trade
  count (412) and the control-at-share gap (0.100 ± 0.002).
- **FR-009**: Per input, S1 MUST record:
  - Spearman correlation with bb_width and with the control's buy probability, at each trade's
    signal bar;
  - the partial Spearman correlation with trade return given bb_width and the control's buy
    probability, pooled and per fold;
  - the number of folds whose sign matches the pooled sign;
  - the split gap: agree − disagree pp after splitting at the input's median within each fold.
- **FR-010**: An input survives S1 only if all of these hold:
  - coverage ≥ 90% of the trades (an input below this is INCONCLUSIVE, not FAIL: it was not
    tested);
  - |Spearman with bb_width| ≤ 0.5 (orthogonality, constitution IX);
  - |pooled partial correlation| ≥ 0.10 (about 2 standard errors at 412 trades);
  - the same sign in at least 6 of 8 folds.

  Survivors are the top 3 by |partial correlation|. The output is committed before any training.
- **FR-011**: S1 reads no late-half (f08–f14) data for any purpose.

**Training (S2, early half only)**

- **FR-012**: Each survivor MUST be a one-variable change from the control ARO model: the same
  architecture, label, sequence length, training rows, exits and seed, plus that one input in the
  model's inputs. Inputs are added one at a time (constitution VIII), never combined at this stage.
  An input whose training span lacks source data (positioning before 2021-12-01) enters as its
  declared neutral value plus a missing-flag column. The pair counts as one input, and training
  rows stay identical to the control's.
- **FR-013** *(amended in planning, research R1)*: New inputs MUST carry no fitted artefact. Each
  is normalised by a causal trailing transform, proven by the truncation test (FR-005), and enters
  the model as a pre-normalised column. The control's existing columns MUST use the control's own
  scaler, unchanged and not overwritten, so the survivor differs from the control in that one input
  alone (constitution VIII). The control's shared scaler is a pre-existing constitution XI exposure
  that applies identically to both. It is recorded in the report, and fixing it is out of scope.
  The original wording required per-fold scalers for the survivor. That would have been a second
  variable against the control.
- **FR-014**: Each survivor's weights MUST live under their own model key. Every existing model
  key, fingerprint, scaler, weight file and per-fold trades hash MUST be unchanged, and existing
  classes behave byte-identically.
- **FR-015**: S2 MUST score each survivor with the control sizer's S1 instrument: the threshold
  at the 0.5 agree share on the validation tail, and agree on buy probability alone. An input
  passes only if its early gap exceeds 0.100, with at most 2 of 8 folds degenerate. The S2 winner
  is the passer with the largest gap. Each passer's margin over 0.100 is recorded. S2 selects and
  S3 decides: a small S2 margin is not evidence by itself, because retraining moves the gap.
- **FR-016**: Training MUST run as one sequential, resumable chain, refusing to start while
  another trainer or backtest chain runs.

**Backtest (S3)**

- **FR-017**: The S2 winner MUST be trained on f08–f14, have thresholds picked per fold as the
  control sizer did, and be backtested as the control-sizer arm with only its weights changed, on
  all 15 folds.
- **FR-018**: S3 passes only if judge_one passes against `cs_aro` AND against `w_aro`. The
  comparison with `wf_aro_g24` is reported. A margin under 1 pp needs a paired seed-2 repeat in
  which every margin is positive on both seeds.
- **FR-019**: The S3 instrument repeats the control sizer's checks:
  - the trade set equals the width book's on (pair, open, close), with at most 2 folds differing
    per half;
  - 0 entry violations and 0 tag mismatches;
  - thresholds loaded at 4 decimals;
  - mean capital within ±10%;
  - the realised stake ratio reported.
- **FR-020**: A failed stage stops that input; later stages are recorded as not reached and their
  records are not read. Outcomes are PASS (PROVISIONAL), FAIL or INCONCLUSIVE. Nothing is adopted
  or promoted by this feature.
- **FR-021**: The judge MUST refuse to run if the pre-registration is uncommitted or dirty, or if
  any arm record predates its commit.

**Constraints**

- **FR-022**: Everything new is default OFF:
  - Framework/ stays strategy-agnostic, and input builders are registered from the strategy side;
  - parameter dicts merge from the parent;
  - there is no pair-specific code;
  - the exit-parameter guard test keeps passing;
  - the live dry-run bot is not touched.

### Key Entities

- **Input**: a named, causally aligned per-pair, per-bar series, with its definition, source,
  window, missing rule and live-feasibility record.
- **Source cache**: the downloaded raw archives, outside the repository, with a manifest
  recording what was fetched and when.
- **Screen table**: per input, the coverage, orthogonality, partial correlations, sign stability,
  split gap and survivor status.
- **Survivor model**: the control architecture plus one input, with its own model key and
  per-fold scalers.
- **Verdict**: per input, the stage reached, the decision and the evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every declared input has a recorded S0 coverage and truncation result, and an S1
  row or a stated INCONCLUSIVE reason.
- **SC-002**: S1 reproduces the 412-trade set and the control-at-share gap 0.100 within ±0.002
  before any input is scored.
- **SC-003**: Existing classes produce byte-identical trades on at least 2 control folds, with
  the held-pair add declared as in those records, and the full test suite passes.
- **SC-004**: No late-half P&L is read for any input except the S2 winner.
- **SC-005**: The report answers three questions for every input: does it carry ranking
  information beyond bb_width and the current model, does training it in improve separation, and
  (for a winner) does it improve return over the current sizer. Each answer comes with its
  live-feasibility record.
- **SC-006**: Compute stays within one sequential chain: S1 needs no training; S2 is at most
  3 inputs × 8 folds of training; S3 is one winner on 7 more folds plus 15 backtests (plus a
  seed-2 repeat only if a margin is small).

## Assumptions

- The control is ARO with 24 h grace (`wf_aro_g24`); the sizer to beat is `cs_aro`, and the width
  book is `w_aro`. Their records are on disk.
- The early width-book trade set and the simulator are 033's, reused unchanged.
- Window lengths default to multiples of the 15-minute bar matching the existing indicators (4,
  16, 96 bars), declared once.
- The input is supplied to the model as one more feature column. Its sequence handling is the
  same as the existing features.
- ARO only; GBR is out of scope unless a later feature repeats a winner.
- One era (2023–26) is available, so any result is PROVISIONAL. The confirmation route is
  post-2026-09-07 data or a dry-run.
