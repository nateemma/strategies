# Feature Specification: NNPredict Improvement Study

**Feature Branch**: `039-nnpredict-improvement` (spec directory; work is committed to master)

**Created**: 2026-10-19

**Status**: Draft

**Input**: User description: "OK, let's move on to the NNPredict folder. Can you review the contents and give an opinion on what we can do to improve these strategies (one of which is likely to adopt the recent NNNC framework changes)." After the review: "start a new study and use the speckit process. Start with fixing what is here and testing, then proceed with your suggestions (and anything else that comes up). For models, maybe use an NNNC-like approach (straight LSTM) maybe with/without indicators (and/or with the additional indicators such as ret1). If you do not need input from me, run through the whole process."

## Background *(evidence; review of NNPredict/ on 2026-10-19 and memory)*

- **What NNPredict does.** NNPredict is the regression family: it predicts the return over the next H
  candles in ATR units. Three targets exist:
  - "point": the endpoint return (the default);
  - "excursion": the dominant capturable move;
  - "stop_aware": the realised stop if the path hits it first.

  It enters when the prediction's rolling z-score reaches `entry_z` and the prediction clears a
  magnitude floor. Backends are Ridge, Keras LSTM, MLX hybrid LSTM, and wavelet-coefficient variants
  (`NNPredict_Coeff*`).
- **Its evaluation has never been comparable.** The best recorded result is `NNPredict_Coeff_Exc`:
  8 of 8 windows positive, +4.1% over 14 months, 134 trades. It came from a separate 8-window
  walk-forward on a different pair set, and was only 3 of 8 windows positive on held-out pairs.
  NNPredict has never run on the 15-fold harness that NNNC, the width book and SimpleTrigger use.
- **Defects found in the review:**
  - eight post-model settings are declared as weight-shaping in the model key: `entry_z`,
    `min_magnitude`, `rolling_window`, `magnitude_floor_mode`, `magnitude_pctile`, `mag_std_mult`,
    `entry_quantile` and `exit_quantile`;
  - the README describes rolling-quantile signals the code no longer uses;
  - Ridge saves `.joblib` weights while the fold runner's artifact check looks for `.safetensors`;
  - stop-loss grace is 2.6 h, inherited from the old base value; NNNC adopted 24 h after testing;
  - the `stop_aware` target has never been tested;
  - the Coeff features depend on `utils/Wavelets.py`, which has uncommitted local edits.
- **Prior evidence:**
  - 038 found that forecasting short-horizon returns from price and indicators gives about 0 edge
    over random entry after costs;
  - NNPredict's own history agrees: only cutting turnover and changing the target improved P&L;
    fit-quality levers did not;
  - NNNC found the model's value is as a ranker or sizer inside the width book: `cs_aro` beat the
    width book by 8.5 pp.

## User Scenarios & Testing *(mandatory)*

The user is the researcher. They want NNPredict fixed, evaluated on the same footing as every other
family, and then improved where improvement is plausible. The end result is a clear recommendation on
which configuration, if any, adopts the NNNC framework by default.

### User Story 1 - Fixes, proven by tests (Priority: P1)

The researcher can trust NNPredict's plumbing. Changing a post-model setting reuses the same trained
weights, Ridge models pass the fold runner's artifact check, and the README matches the code.

**Why this priority**: Every later story trains and evaluates through this plumbing. A misclassified
model key makes gate arms retrain or fail. A failed artifact check makes Ridge folds unrunnable.

**Independent Test**: Run the family's test suite. Two strategies that differ only in a post-model
setting resolve to the same model, and a trained Ridge fold passes the artifact check.

**Acceptance Scenarios**:

1. **Given** two NNPredict classes that differ only in `entry_z` (or any other post-model gate
   setting), **When** their model keys are computed, **Then** the keys are identical.
2. **Given** a setting that does shape the weights (the target mode, the target caps, `atr_floor`,
   the label smoothing), **When** it changes, **Then** the key changes.
3. **Given** a Ridge fold that has been trained, **When** the fold runner verifies its artifact,
   **Then** the check passes.
4. **Given** the README, **When** it is read against the code, **Then** it describes the z-score and
   magnitude-floor gate, the targets and the family's current status.

---

### User Story 2 - Baseline on the common harness (Priority: P1)

The researcher sees NNPredict's existing configuration scored on the 15 folds and 10 pairs that every
other family uses, against the same comparators.

**Why this priority**: Without a common baseline, no later change can be judged. The baseline is
NNPredict with no change in behaviour.

**Independent Test**: The baseline arm has a record for every fold and a pooled score, and it is
compared with the width book and the NNNC reference book.

**Acceptance Scenarios**:

1. **Given** the 15-fold plan, **When** the baseline arm runs, **Then** each fold trains its own
   model on that fold's training window only and is evaluated on the fold's test window.
2. **Given** the baseline results, **When** they are reported, **Then** they appear beside the width
   book (`w_aro`), the NNNC reference book (`cs_aro`) and the momentum reference (`st_g24_mom`),
   with per-half totals and trade counts.

---

### User Story 3 - Plain-LSTM regressors with three input sets (Priority: P2)

The researcher sees whether an NNNC-style plain LSTM (inputs straight into an LSTM, then a linear
output) predicts NNPredict's target better than the existing models, using three input sets:
- **raw:** OHLCV-derived inputs only;
- **raw + indicators:** NNNC's 24 indicators added;
- **raw + lagged returns:** returns over several horizons, realised volatility and volume ratio.

**Why this priority**: The user asked for it, and it answers whether the architecture that works as
NNNC's ranker also helps NNPredict. Its weight in decisions is bounded by 038's finding.

**Independent Test**: Each variant trains on one fold, produces finite predictions, and resolves to a
model key distinct from the others.

**Acceptance Scenarios**:

1. **Given** each input set, **When** the arm runs on all 15 folds, **Then** it receives one verdict
   against the baseline under the pre-registered rule.
2. **Given** the three variants, **When** they are compared, **Then** the report says which input set
   (if any) helps, in each half.

---

### User Story 4 - The NNNC framework at the family base (Priority: P1)

NNPredict gains the NNNC framework (the model as the entry source, with per-family exits, the trigger
gate, the held-pair add and the framework's stop rules) with every feature off by default, proven to
change nothing. Each feature is then tested as its own pre-registered arm.

**Why this priority**: This is the change the user expects NNPredict to adopt, and it is required by
Story 5 (the width book and sizer need the framework).

**Independent Test**: The baseline arm, re-run on the framework-enabled base with all features off,
produces identical trades on every fold.

**Acceptance Scenarios**:

1. **Given** the framework-enabled base with nothing declared, **When** the baseline is re-run,
   **Then** every fold's trades are identical to the baseline's (the same hash).
2. **Given** each feature arm (24 h grace, in-profit held-pair add, family exits, the gbb trigger
   gate), **When** it is judged against the framework baseline, **Then** it receives one verdict, and
   features that pass become candidates for the family default.

---

### User Story 5 - NNPredict as a sizer inside the width book (Priority: P1)

The researcher sees whether NNPredict's predicted move ranks the width book's trades. It is screened
cheaply first and, if the screen passes, run as a stake-sizer arm judged against the width book and
`cs_aro`.

**Why this priority**: This is where NNNC found the model's value, and a regressor's predicted move is
a natural ranking score.

**Independent Test**: The screen reports, for the width book's trades, how well the prediction ranks
trade outcomes in each half, against the same bar `cs_aro`'s screen used.

**Acceptance Scenarios**:

1. **Given** the width book's trades and NNPredict's predictions at their entries, **When** the screen
   runs, **Then** it reports the ranking quality per half and a pre-registered pass or fail.
2. **Given** a passing screen, **When** the sizer arm runs, **Then** it is judged against the width
   book (it must beat it) and against `cs_aro` (whether it beats it is reported).
3. **Given** a failing screen, **When** the study is reported, **Then** no sizer backtest is run and
   the failure is recorded as a result.

### Edge Cases

- **A model that predicts almost constant values.** The z-score gate then fires on noise (the
  label-smoothing failure in memory). The selected model's prediction spread per fold is reported
  from its signal tables. For every arm, trades per fold are reported, and an arm with no trades in
  a half fails its instrument check (INCONCLUSIVE).
- **Folds with too few entries.** These are reported per fold. An arm with fewer trades than the
  pre-registered minimum is INCONCLUSIVE, never a zero return.
- **Training that silently fails.** The artifact check is the only reliable sign. A fold without
  weights stops the arm rather than training inside its own backtest.
- **Pairs with long gaps (XRP) or many zero-volume candles.** Handled by the shared harness, the same
  as for every other family; there are no pair-specific settings.
- **The Coeff family's dependence on uncommitted code.** It is excluded unless that file is clean, so
  no result depends on unrecorded edits.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every NNPredict setting that acts only after the model MUST be classified as not
  shaping the weights. Every setting that shapes the weights MUST stay classified as weight-shaping.
  Tests MUST prove both.
- **FR-002**: Ridge NNPredict folds MUST pass the fold runner's trained-model check.
- **FR-003**: The NNPredict README MUST describe the current code and the family's status.
- **FR-004**: A baseline of NNPredict's existing configuration MUST be scored on the 15-fold harness
  (f00–f14, the 10 `config_wf` pairs), with a model trained per fold on that fold's training window
  only.
- **FR-005**: Plain-LSTM regressor variants MUST exist for three input sets (raw; raw + NNNC's 24
  indicators; raw + lagged returns), and each MUST be scored on the same harness.
- **FR-006**: The NNNC framework MUST be available at the NNPredict family base with every feature
  off by default. With nothing declared, the framework-enabled base MUST reproduce the baseline's
  trades exactly.
- **FR-007**: Each framework feature (24 h grace, in-profit held-pair add, family exits, gbb trigger
  gate) MUST be tested as its own arm against the framework baseline. For family exits, only the
  momentum set is tested: the framework's reversion set equals the shared exits, so a reversion arm
  would be identical to the framework baseline. An arm whose trades are unchanged on every fold is
  INERT, a defect, and gets no verdict.
- **FR-008**: NNPredict's predicted move MUST be screened as a ranker of the width book's trades
  before any sizer backtest. A sizer arm MUST run only if the screen passes, and MUST then be judged
  against the width book and `cs_aro`.
- **FR-009**: Every rule that decides a verdict MUST be written and committed before the results it
  judges exist.
- **FR-010**: Verdicts MUST rest on walk-forward P&L in both halves (early f00–f07, late f08–f14),
  with paired and risk checks. Fit metrics MUST NOT decide a verdict.
- **FR-011**: Every new or changed setting MUST merge from its parent's parameters. No pair-specific
  code is allowed. Shared framework code MUST stay family-agnostic.
- **FR-012**: The study MUST end with a recommendation: which NNPredict configuration (if any) adopts
  the NNNC framework by default, and which framework features it turns on.

### Key Entities

- **Arm**: A strategy configuration run over the 15 folds, with its comparator and its verdict.
- **Fold record**: One fold's trades, totals and trained-model reference.
- **Pre-registration**: The committed rule set for a group of arms.
- **Screen result**: The ranking quality of predictions on the width book's trades, per half, with a
  pass or fail.
- **Verdict**: PASS, FAIL or INCONCLUSIVE per arm, with the numbers that decided it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All fixes are covered by tests, and the existing NNPredict and framework test suites
  still pass.
- **SC-002**: The baseline and every model variant have a record for all 15 folds. No fold trains
  inside its own evaluation window.
- **SC-003**: The framework-enabled version of the selected model, with nothing declared, matches
  that model's trades on 15 of 15 folds.
- **SC-004**: Every arm in scope receives exactly one verdict under a rule committed before its
  results existed.
- **SC-005**: The final report states, in one table, every arm's per-half result against its
  comparator, plus a single recommendation on NNPredict's default configuration.
- **SC-006**: Each training run is sequential (one trainer at a time), and the two running bots are
  never touched.

## Assumptions

- **Harness.** This uses the existing 15-fold plan shape: anchored training from 2021-01-01, 90-day
  test windows f00–f14, the `config_wf` pairs, and the comparators' existing records (`w_aro`,
  `cs_aro`, `st_g24_mom`). Every PASS is PROVISIONAL (no distant era), as since 023.
- **Gate and target.** The baseline uses NNPredict's current defaults (point target, H = 4, z-score
  plus magnitude-floor gate) on `NNPredict_Ridge`. Changing them is a separate arm, not a silent
  change.
- **Coeff family.** `NNPredict_Coeff*` is excluded from scoring while `utils/Wavelets.py` has
  uncommitted edits. 038 also found wavelet features add nothing once boundary effects are handled.
- **`stop_aware` target.** Tested as one arm against the point target on the best baseline model. A
  plain MSE fit of its bimodal label is the configuration as built.
- **Order of work.** Fixes → baseline → framework port (identity proof) → model variants and feature
  arms → screen → sizer. A later stage proceeds only if its prerequisites hold (identity proven,
  screen passed).
- **Out of scope:** multi-horizon heads, more LSTM capacity, and label smoothing. Memory records each
  of these as failed or decoupled from P&L.
