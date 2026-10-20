# Feature Specification: NNMT Parity and Head Study

**Feature Branch**: `040-nnmt-parity-heads` (spec directory; work is committed to master)

**Created**: 2026-10-20

**Status**: Draft

**Input**: User description: "ok, let's switch to NNMT. The goals are: 1. bring NNMT up to par with NNNC in terms of framework 2. add model variants for raw data and raw LSTM, and LSTM + raw data + indicators, and evaluate performance 3. evaluate if any of the other prediction heads (e.g. regime) are good enough to use in addition to (or instead of) trading. For example, an upcoming transition from bear to bull seems like a valuable signal. Draft a specification for use with the speckit process"

## Background *(evidence; review of NNMT/ on 2026-10-20 and memory)*

- **What NNMT does.** NNMT is the multi-task family. One network (`NNMT_MLX`, an MLX LSTM) trains a
  trading head (sell / hold / buy) and five auxiliary heads:
  - regime (bull / bear / sideways);
  - risk (low / normal / high volatility);
  - flow and momentum (down / neutral / up);
  - profit (loss / neutral / profit).

  The aim of the auxiliary heads is to regularise the shared network so the trading head
  generalises.
- **It has never run on the common harness.** NNMT has no record on the 15-fold walk-forward harness
  that NNNC, the width book, SimpleTrigger and NNPredict use. None of its results compare with:
  - the NNNC reference book `cs_aro` (59.95);
  - the width book `w_aro` (51.44);
  - the reversion trigger `st_g24_rev` (71.5).
- **What earlier single-window studies found** (`docs/NNMT_TASK_AUDIT.md`,
  `regime/NNMT_OPERATING_POINT.md`, memory):
  - **The guards, not the model, pick the entries.** The guard chain rejected about 93% of
    candidates. A no-model run matched the real model to +0.02 pp, and a lookahead-perfect model
    did worse (negative headroom).
  - **Task weights cannot be tuned.** Head-weight changes are not resolvable above seed noise, and
    several arms produced byte-identical trades from different weights.
  - **The regime and profit heads are never trained** (task weight 0).
  - **The trading label is NNNC's label.** An older audit recorded the trading label as filtered by
    the ground-truth auxiliary targets. Since 2026-09-01 the filter writes only the debug columns, and
    the head trains on the unfiltered gbb_path label (H = 48, ±2%), the same label as NNNC's models.
    So the auxiliary heads act only as regularisers, and a trading-head-only NNMT is directly
    comparable with NNNC.
  - **Gating entries on the auxiliary predictions** (`apply_task_filters`) cost 14 pp.
  - **NNMT's best rvol guard is 8× looser than NNNC's,** so the families need separate tuning.
- **What the regime head predicts today.** The regime label compares the close with a CENTRED
  200-bar mean (±1.5%), shifted 5 bars ahead. That is a local trend state of about 2 days that uses
  future prices. It is not a market bear or bull regime, and a "transition" in it is mostly that
  centred-mean artifact.
- **Prior evidence on regime prediction (037):** regime-style targets beat a persistence baseline
  only at 48- and 288-bar horizons (12 h and 3 days). Price-level and calendar columns produced
  spurious "winners" there.
- **Where NNNC stands.** NNNC carries the entry framework: the model or a trigger as the entry
  source, per-family exits, the held-pair add, and an operating point with guards open and a 0.80
  prediction threshold (ARO's). Its adopted defaults are:
  - 24 h stop-loss grace (NNMT has 3 h);
  - in-profit held-pair add.

  The ATR take-profit (C2) exists as a default-off switch. Its models earn their keep as a stake
  sizer inside the width book (`cs_aro`), not as an entry source.

## User Scenarios & Testing *(mandatory)*

The user is the researcher. They want three things:
- NNMT on the same framework and harness as NNNC;
- to know whether NNNC's simpler raw-input LSTMs work better inside NNMT's multi-task setup;
- to know whether any auxiliary head carries a signal worth trading, above all an early warning of a
  bear-to-bull (or bull-to-bear) change.

### User Story 1 - The NNNC framework at the NNMT family base, proven to change nothing (Priority: P1)

NNMT gains the NNNC entry framework with every feature off by default, and the first evaluation shows
it changes no trade.

**Why this priority**: Every later arm runs through this framework. An identity proof is the only
way to know that a later difference comes from the feature under test, not the port.

**Independent Test**: Run current NNMT on the 15 folds before and after the port, with nothing
declared. The trades are identical on every fold.

**Acceptance Scenarios**:

1. **Given** the framework-enabled NNMT base with nothing declared, **When** the NNMT baseline arm is
   re-run, **Then** every fold's trades are identical to the pre-port baseline (the same hash).
2. **Given** two NNMT classes that differ only in a post-model setting (a threshold, guard, gate or
   exit setting), **When** their model keys are computed, **Then** the keys are identical. A
   setting that shapes the weights (a label parameter, task weight, input set or head) changes the
   key.
3. **Given** an NNMT base class with no pinned training window, **When** its weights are located
   under a dynamic pairlist, **Then** it uses mainline naming like every other family.

---

### User Story 2 - NNMT baseline and operating point on the common harness (Priority: P1)

The researcher sees current NNMT scored on the 15 folds and 10 pairs, both at its own defaults and at
an operating point where the model rather than the guards decides the entries. Both appear beside
the NNNC, width-book and trigger references.

**Why this priority**: Without a baseline nothing can be judged. Without an operating point where
the model decides, no later model or head change can show up in P&L, as the earlier task studies
found.

**Independent Test**: Both arms have a record for all 15 folds. The report gives the share of
entries that the model rather than the guards decided.

**Acceptance Scenarios**:

1. **Given** the 15-fold plan, **When** the baseline runs, **Then** each fold trains on its own
   training window only and is evaluated on its test window.
2. **Given** the operating-point arm (guards open and ARO's prediction threshold of 0.80, as
   NNNC's), **When** it is reported, **Then** it shows per-half totals, trades and the share of model-decided entries,
   against the baseline and the references.

---

### User Story 3 - NNNC's adopted features as pre-registered arms (Priority: P2)

Each NNNC framework feature is tested on NNMT as its own arm:
- 24 h stop-loss grace;
- in-profit held-pair add;
- gbb trigger gate (model AND trigger);
- C2 ATR take-profit.

The features that pass become candidates for NNMT's defaults.

**Why this priority**: This is the parity the user asked for. Families tune independently, so
NNNC's verdicts cannot simply be copied.

**Independent Test**: Each arm gets one verdict against its comparator under a rule committed before
its results exist.

**Acceptance Scenarios**:

1. **Given** each feature arm, **When** it is judged, **Then** it receives PASS, FAIL or INCONCLUSIVE
   per the pre-registered rule, with per-half numbers.
2. **Given** an arm whose trades are unchanged on every fold, **When** it is judged, **Then** it is
   INERT (a defect) and gets no verdict.

---

### User Story 4 - Raw-input and plain-LSTM model variants (Priority: P2)

The researcher sees whether NNNC's model designs work inside NNMT's multi-task setup. The heads stay
the same and only the inputs and architecture change:
- **raw:** raw OHLCV-derived inputs only, in NNMT's current network;
- **plain LSTM:** inputs straight into a plain LSTM, then the heads, with no resize or convolution
  stage. This is ARO's design. It is run on three input sets:
  - NNNC's 24 indicators (ARO's own inputs);
  - raw inputs only;
  - raw inputs plus the 24 indicators.

A trading-head-only twin of the best variant measures whether multi-task training helps at all.

**Why this priority**: The user asked for it. In NNNC, the raw-input plain LSTM (ARO) is the primary
model.

**Independent Test**: Each variant trains on one fold, produces predictions for every head, and
resolves to its own model key.

**Acceptance Scenarios**:

1. **Given** each variant, **When** it runs on all 15 folds at the selected operating point,
   **Then** it receives one verdict against current NNMT at that operating point. It is also reported against ARO and the
   width book.
2. **Given** the best variant and its trading-head-only twin, **When** they are compared, **Then**
   the report says whether the auxiliary heads help, hurt or do nothing, in each half.
3. **Given** the best variant, **When** its buy probability is screened as a width-book stake sizer
   (the `cs_aro` pattern), **Then** the screen reports its ranking against the bar `cs_aro` passed.
   A sizer arm runs only if the screen passes, and is judged against the width book and `cs_aro`.

---

### User Story 5 - Is any auxiliary head worth trading? (Priority: P2)

For each trained auxiliary head (regime, risk, momentum, flow, profit), the researcher sees a cheap
screen with two questions:
- does the head beat a persistence or trailing baseline on held-out folds?
- does it rank the outcomes of width-book or NNNC trades?

Only heads that pass get a pre-registered backtest arm in a use that fits the head, for example:
- regime as an entry gate or exposure scaler;
- risk as a stake sizer;
- profit as an exit or bail-out.

**Why this priority**: The heads exist and cost nothing to read. Using them in trading has failed
before (the task filter cost 14 pp), so a head must earn a backtest by passing the screen first.

**Independent Test**: The screen produces a per-head, per-half table with a pass or fail. A head
that fails gets no backtest.

**Acceptance Scenarios**:

1. **Given** a head's held-out predictions, **When** the screen runs, **Then** it reports the head's
   skill against the persistence or trailing baseline, and its ranking of trade outcomes, per half.
2. **Given** a passing head, **When** its arm runs, **Then** the arm is judged on walk-forward P&L
   against the arm without the head.
3. **Given** a failing head, **When** the study is reported, **Then** the failure is recorded as a
   result and no arm is run.

---

### User Story 6 - Regime transition: an early warning of bear-to-bull and bull-to-bear (Priority: P1)

The researcher sees whether an upcoming regime change can be predicted early enough to trade. The
change is defined causally, from information available at the predicted time, at horizons of 1 to 7
days. It is not defined from the centred-mean label. Up-transitions (bear to bull) are tested as an
entry or exposure signal, and down-transitions (bull to bear) as an exit or risk-off signal.

**Why this priority**: The user named it as the most valuable signal. It is also the only head
whose target must be newly defined, so it carries the most risk of an artifact.

**Independent Test**: The transition target has a written causal definition and a base rate per
fold. A truncation test shows that no input or label at time t uses data after t, except the label's
own forward horizon.

**Acceptance Scenarios**:

1. **Given** the transition definition, **When** the labels are built, **Then** the regime at both
   ends of the horizon is computed only from data up to that time, and the share of transitions per
   fold and per half is reported.
2. **Given** a model of the transition (an NNMT head, or a cheap model trained on the same inputs),
   **When** it is screened, **Then** its held-out skill is reported against two baselines: the base
   rate, and a trailing-trend rule that flags a transition when the current trend has already
   turned.
3. **Given** a transition signal that passes the screen, **When** it is traded, **Then** each use is
   its own pre-registered arm and is judged on walk-forward P&L. The uses are:
   - a gate on an existing book;
   - exposure scaling of an existing book;
   - a standalone long entry at a predicted bear-to-bull change;
   - a risk-off exit at a predicted bull-to-bear change.
4. **Given** a signal that fails the screen, **When** the study is reported, **Then** it is recorded
   as not predictable at that horizon, with its numbers.

### Edge Cases

- **Guards still choose the entries.** If, at the operating point, the model decides fewer entries
  than the pre-registered minimum share, that is reported. Model and head arms there are
  INCONCLUSIVE rather than null.
- **Transitions are rare.** A 1–7-day regime change may occur only a few times per pair per fold. The
  screen requires a pre-registered minimum count of transitions per half. Below it the result is
  INCONCLUSIVE.
- **A transition label that is just trend-following.** If the trailing-trend baseline matches the
  model's skill, the signal adds nothing beyond "the trend already turned". That is a FAIL, however
  good the fit.
- **Market-wide and per-pair regimes differ.** Alts follow BTC, so a per-pair "bull" can be the
  market's. Both are defined and reported, and a transition that holds only for one of them is
  reported as such.
- **Heads that are never trained.** Regime and profit have task weight 0 today, so their outputs are
  untrained noise. They are screened only from a model that trains them, and the trading head of
  that model is checked against the default so a weight change does not silently move the baseline.
- **Auxiliary labels use future prices.** Every auxiliary label is a centred rolling state shifted
  5 bars ahead, so half its window is future data. This is allowed in a label, but a head screen
  must compare against a baseline built from past data only. The head is not credited with
  "predicting" what is mostly the current state.
- **Pairs with gaps or zero-volume candles** (XRP 2021–2023). These are handled by the shared
  harness. There is no pair-specific code.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The NNNC entry framework MUST be available at the NNMT family base with every feature
  off by default. With nothing declared, the framework-enabled base MUST reproduce current NNMT's
  trades on all 15 folds.
- **FR-002**: Every NNMT setting that acts only after the model MUST be classified as not shaping
  the weights, and every setting that shapes them MUST stay weight-shaping. Tests MUST prove both.
  NNMT base classes MUST work with mainline model naming.
- **FR-003**: A baseline of current NNMT (`NNMT_MLX` at its defaults) MUST be scored on the 15-fold
  harness, with one model per fold trained on that fold's training window only.
- **FR-004**: An operating-point arm (guards open, ARO's prediction threshold 0.80) MUST be scored. The report MUST
  give the share of entries the model decides under the baseline and the operating point.
- **FR-005**: Each NNNC framework feature MUST be tested on NNMT as its own arm: 24 h grace,
  in-profit held-pair add, gbb trigger gate and C2 ATR take-profit. An arm whose trades are
  unchanged on every fold MUST be reported as INERT, not given a verdict.
- **FR-006**: Four model variants MUST exist and be scored, all keeping the multi-task heads:
  - raw inputs in the current network;
  - a plain LSTM on NNNC's 24 indicators (ARO's design);
  - a plain LSTM on raw inputs;
  - a plain LSTM on raw inputs plus the 24 indicators.
- **FR-007**: A trading-head-only twin of the best variant MUST be scored, to measure the effect of
  multi-task training.
- **FR-008**: The best variant's buy probability MUST be screened as a width-book stake sizer before
  any sizer arm. A sizer arm runs only if the screen passes, and is then judged against `w_aro` and
  `cs_aro`.
- **FR-009**: Every auxiliary head MUST be screened on held-out folds for two things:
  - skill against a persistence or trailing baseline;
  - ranking of width-book trade outcomes. The NNNC reference book `cs_aro` trades exactly the width
    book's trades, so this one set covers both.

  A head MUST get a backtest arm only if it passes.
- **FR-010**: A regime-transition target MUST be defined causally. The regime at each end of the
  horizon uses only data up to that time, and the centred-mean label is not used. It MUST be defined
  for horizons from 1 to 7 days, at the pair level and at the market (BTC) level, with base rates
  reported per fold.
- **FR-011**: The transition signal MUST be screened against the base rate and a trailing-trend
  baseline before any trading arm. Each trading use that passes MUST be its own arm:
  - a gate on an existing book;
  - exposure scaling;
  - a standalone long entry on bear to bull;
  - a risk-off exit on bull to bear.
- **FR-012**: Every rule that decides a verdict or screen MUST be written and committed before the
  results it judges exist.
- **FR-013**: Verdicts MUST rest on walk-forward P&L in both halves (early f00–f07, late f08–f14),
  with paired and risk checks. Fit metrics (MCC, accuracy, AUC on labels) MUST NOT decide a verdict.
  They may decide only whether a screen passes. Any claimed win MUST be repeated on a second seed.
- **FR-014**: Every new or changed setting MUST merge from its parent's parameters. No pair-specific
  code is allowed. Shared framework code MUST stay family-agnostic. Training MUST be sequential, and
  the running bots MUST NOT be touched.
- **FR-015**: The study MUST end with three things:
  - a recommendation on NNMT's default configuration (framework features, model variant, operating
    point);
  - a verdict per auxiliary head (use or don't use, and how);
  - a verdict on the regime-transition signal.

### Key Entities

- **Arm**: a strategy configuration run over the 15 folds, with its comparator and its verdict.
- **Fold record**: one fold's trades, totals, model-decided entry share and trained-model reference.
- **Head screen result**: for one head and horizon, held-out skill against its baseline and trade
  outcome ranking, per half, with pass or fail.
- **Regime-transition target**: a causal label marking a change of regime between now and a horizon
  of 1–7 days. It exists at the pair and market level, with its base rate per fold.
- **Pre-registration**: the committed rule set for a group of arms or screens.
- **Verdict**: PASS, FAIL, INCONCLUSIVE or INERT per arm, with the numbers that decided it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The framework-enabled NNMT base with nothing declared matches current NNMT's trades on
  15 of 15 folds.
- **SC-002**: The baseline, the operating point, every feature arm and every model variant have a
  record for all 15 folds. No fold trains inside its own evaluation window.
- **SC-003**: Every arm and screen in scope receives exactly one verdict under a rule committed
  before its results existed. Every claimed win is repeated on a second seed.
- **SC-004**: Every auxiliary head, and the transition signal at each defined horizon, has a
  screen result per half. Every trading use of a passing signal has a P&L verdict.
- **SC-005**: The transition target passes a truncation test: no input or label at time t changes
  when data after t (beyond the label's own horizon) is removed.
- **SC-006**: The final report gives, in one table, every arm's per-half result against its
  comparator and against `cs_aro`, `w_aro` and `st_g24_rev`. It ends with the three verdicts of
  FR-015.
- **SC-007**: All existing NNMT, NNNC and framework test suites still pass, apart from failures that
  already existed before the study.

## Assumptions

- **Harness.** This uses the existing 15-fold plan:
  - anchored training from 2021-01-01, then 90-day test windows f00–f14;
  - the `config_wf` pairs;
  - the comparators' existing records (`cs_aro`, `w_aro`, `st_g24_rev`, ARO's `wf_aro_g24`).

  Every PASS is PROVISIONAL (no distant era), as since 023.
- **Baseline model.** The baseline is `NNMT_MLX` at its current defaults, including its task weights
  (regime and profit at 0). Other NNMT classes (CGP, DDPM, WGAN, Transformer, Attention) are out of
  scope. GAN augmentation is out of scope; memory records it as not helping P&L.
- **Order of work.** The stages run in this order:
  1. port and identity proof;
  2. baseline and operating point;
  3. feature arms and model variants;
  4. sizer and head screens;
  5. transition target and screen;
  6. trading arms for whatever passed.

  A stage proceeds only if its prerequisites hold (identity proven, screen passed).
- **Heads that are not trained today.** The regime and profit heads are screened from a model that
  trains them (non-zero task weight), and that model's trading head is reported against the default.
  Task-weight tuning is not reopened; memory records it as unresolvable.
- **Transition definition.** The regime at a time is computed from trailing data only. A candidate
  is the sign of a trailing multi-day trend with a dead band. Its exact form and dead band are fixed
  in the pre-registration before any transition result is seen. Horizons are a small fixed set
  within 1–7 days (e.g. 1, 3 and 7 days). The market regime is BTC's.
- **The transition model.** It is screened first with a cheap per-fold model trained on the
  deployed model's inputs (as 037's screens used), so a dead target does not cost a 15-fold retrain.
  Only if that passes is it trained as an NNMT head (the idle regime head, retargeted) and screened
  the same way. Trading arms read the passing model's per-fold predictions. Wiring a passing signal
  into a live strategy is a follow-up.
- **Standalone transition entries** use the shared exit stack. Designing new exits for them is out
  of scope.
- **Out of scope:** new data sources (order flow, funding), more network capacity, label smoothing,
  and changing how the trading label is built from the auxiliaries. The last would be a separate
  feature if the head screens point to it.
