# Feature Specification: Label-Type Screen for NNNC

**Feature Branch**: `034-label-type-screen` (spec directory; work is committed to master)

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Label-type screen for NNNC (feature 034). [...] Goal: find whether any
other label type teaches the model information bb_width does not already carry, screening cheaply
first and backtesting only survivors. Candidates (proposed, confirm in spec): gbb_path (control),
gbb_fwdmae, forward_mae, triple_barrier, mfe_mae_ratio, future_expectancy or risk_adj_future,
quantile_future, local_min_followthrough, breakout_gbb [...] Model: one architecture first, ARO or
GBR -- decide in spec; early folds f00-f07 only for the screen [...] S0 label sanity; S1
no-backtest separation inside the width book [...] S2 backtest survivors on the early half in
role(s) -- sizer and/or entry source -- decide in spec [...] S3 late-half confirmation of the S2
winner only [...]" (full text in the conversation of 2026-10-14)

## Background *(evidence; research/model_calibration/findings.md, research/candidate_training/report.md, research/width_relative/report.md)*

**Every NNNC model judged against the width book was trained on one label family.** ARO, GBR and
MLX use gbb_path; GFM uses gbb_fwdmae (gbb's trigger plus a forward adverse-excursion cap). On the
gbb rule's candidates these models score 0.53–0.62 AUC, and what they rank by is mostly band width.
A model-free width book matches ARO, GBR and MLX and beats GFM by 24 pp in real 15-fold backtests.

**Inside the width book, the current models still separate trades** (agree − disagree pp per
width-book trade, early / late): ARO 0.072 / 0.085, GBR 0.063 / 0.072. That separation is the one
place the model adds something width cannot.

**Two attempts to sharpen it by changing what the model is trained on both failed:**
- 032 trained on rule-eligible rows only: within-candidate AUC rose, but the books lost 10–15 pp
  in the early half and the sizer separated worse;
- 033 removed width from the label (outcome ranked within width deciles): the label was
  width-neutral as designed, but the separation gap fell to 0.043 (ARO) and 0.005 (GBR).

**Most label types have never been tried under the current framework.** The framework offers 29
label methods. The ones judged before were judged on old exits and model-only books:
- breakout: learnable but loses as an entry source on every timeframe and exit;
- forward_mae: −0.83 pp on 15 folds as an entry source, against gbb's +39.98 pp (2026-10-08);
- triple-barrier: no feature set moved its ceiling (a data problem).

None of them has been asked the question the width book now poses: **does a model trained on this
label rank width-book trades better than the gbb_path model does?** The oracle gate (lookahead
ceilings) passed all 11 labels it screened, so ceilings do not separate them.

**Harness lessons carried from 032 and 033:**
- scoring a fold strategy outside a backtest needs its saved fold parameters loaded explicitly, and
  parameter values leak between loads in one process;
- the entry threshold rounds to 2 decimals unless redefined (the sizer uses 4);
- trade identity is (pair, open date, close date), not the trades hash, when stakes differ;
- regenerating fold classes drops their thresholds, and the fold runner skips folds that already
  have a record, so smoke records must be set aside;
- the liquidity cap absorbs stake multipliers, so realised stake ratios are measured;
- threshold picks run one fold per process (low-memory kills).

## Clarifications

### Session 2026-10-14

- Q: Which candidate labels? → A: gbb_path (control, existing weights), gbb_fwdmae (existing GFM
  weights), forward_mae, triple_barrier, mfe_mae_ratio, future_expectancy, quantile_future,
  local_min_followthrough, breakout_gbb. Seven are trained; risk_adj_future, future_sharpe and
  future_sortino are left out.
- Q: Which model architecture goes first? → A: ARO. GBR only repeats an S3 winner.
- Q: Which roles are backtested at S2? → A: Both the sizer on the width book and the model as an
  UNGATED entry source (no gbb trigger gate: the control's own entry configuration with only the
  label changed).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Screen label types without a backtest (Priority: P1)

The researcher trains the same model architecture on each candidate label type, on the early folds
only, and learns for each one: is the label sane, how strongly is it tied to band width, and does
the model trained on it separate good from bad width-book trades better than the gbb_path model?
The answer comes from scoring the trained models against width-book trades that have already been
backtested, so no new backtest is needed.

**Why this priority**: It covers every candidate label at the cost of training alone, and it
decides which labels deserve a backtest. It answers the user's request (cover different label
types) on its own.

**Independent Test**: Run the screen for the declared candidates. A ranked table exists with, per
label: label sanity figures, label–width correlation, early-half separation gap and within-candidate
AUC, each next to the gbb_path control's, and a pre-registered survivor list.

**Acceptance Scenarios**:

1. **Given** the committed pre-registration, **When** the screen runs, **Then** every candidate has
   trained weights for all 8 early folds under its own model key, and no existing model key, weight
   file or record is changed.
2. **Given** trained candidates, **When** S0 runs, **Then** each label's per-fold positive rate,
   class balance and Spearman correlation with bb_width on rule-eligible bars are recorded, and a
   label with no positive rows in any fold is marked INCONCLUSIVE rather than scored.
3. **Given** S0 passes, **When** S1 runs, **Then** each label's agree − disagree gap is computed on
   the identical width-book trades used for the control, with the control's gap reproducing the
   pre-registered 0.072 (ARO) or 0.063 (GBR) within rounding.
4. **Given** S1 results, **When** survivors are selected, **Then** the selection follows the
   pre-registered rule exactly and is written out before any survivor is backtested.

---

### User Story 2 - Backtest the survivors (Priority: P2)

The surviving labels (at most 3) are backtested on the early half as a sizer on the width book and
as an ungated entry source, and compared with the width book and the 24 h control.

**Why this priority**: Separation is necessary but not sufficient (learnability is not edge). Only
a real backtest shows whether better ranking turns into return.

**Independent Test**: For each survivor, early-half pooled pp for the arm, the width book and the
control, with a staged verdict per arm.

**Acceptance Scenarios**:

1. **Given** a survivor in the sizer role, **When** it is backtested, **Then** its trades are
   identical to the width book's in the 8 early folds (at most 2 folds may differ, as in 033), its
   capital is within ±10% of the width book's, and every trade's agree or
   disagree tag matches the model's decision on its entry bar.
2. **Given** a survivor in the entry-source role, **When** it is backtested, **Then** its entries
   come only from its own model decision under the control's guards (no gbb trigger gate), and its
   picked entry-signal bar rate on the validation tail is within ±5% of the control's 0.014735.
3. **Given** the early-half results, **When** S2 is judged, **Then** an arm passes only if it beats
   both the width book and the control; a margin under 1 pp triggers the paired seed-2 repeat.
4. **Given** no survivor, **When** the screen ends, **Then** no backtest is run and the report
   records the S1 table as the result.

---

### User Story 3 - Confirm the winner (Priority: P3)

The S2 winner alone is trained on the late folds and judged on the late half against both
comparators; if it passes, the second model architecture repeats it.

**Why this priority**: Confirmation on unread data is what makes a screen result trustworthy, but
it only matters if something wins S2.

**Independent Test**: A late-half verdict (judge_one against the control, late pp against the
width book) and, on a pass, the second model's repeat verdict.

**Acceptance Scenarios**:

1. **Given** an S2 winner, **When** S3 runs, **Then** no late-half P&L of any other candidate is
   ever read.
2. **Given** S3 passes on the first model, **When** the second model repeats it, **Then** the
   repeat runs all 15 folds with the same pre-registered rules.
3. **Given** any pass, **Then** it is labelled PASS (PROVISIONAL), nothing is adopted, and the
   confirmation route (folds after 2026-09-07, or dry-run) is named.

### Edge Cases

- **A label fires almost everywhere or almost nowhere** (forward_mae marked about a third of all
  bars as buys at H48; gbb_fwdmae about 7%). The model's threshold is picked per fold on the
  validation tail to a fixed agree share, so the sizer is comparable regardless of base rate; the
  base rate is reported, not bounded.
- **A label's model never predicts BUY as argmax on width-book entry bars** in a fold: the agree
  share is then 0, the fold is reported as degenerate, and the label's S1 is INCONCLUSIVE if more
  than 2 of 8 early folds are degenerate.
- **A labeler rejects a declared parameter**: the run must fail loudly (unsupported label
  parameters raise), never train on silent defaults.
- **A candidate's weights already exist** (gbb_path = the control; gbb_fwdmae = GFM's weights on
  the same architecture): they are reused, not retrained, and recorded as reused.
- **Lookahead**: every label is forward-looking by definition; the check is that no label column
  or forward value reaches the model's features, and that validation-tail rows used for picking are
  never used for fitting.
- **Low-memory kill mid-chain**: the chain is resumable per fold, and is not restarted without the
  user's say-so.

## Requirements *(mandatory)*

### Functional Requirements

**Candidates and pre-registration**

- **FR-001**: The candidate list, each candidate's label parameters, the model architecture, the
  fold set, the agree share, the survivor rule and every stage cutoff MUST be committed in a
  pre-registration under `research/label_screen/` BEFORE any candidate is trained. Changes after
  that are amendments with a reason, made before the affected results are read.
- **FR-002**: Candidates are gbb_path (control, existing weights), gbb_fwdmae (existing GFM
  weights, reused), and seven trained labels: forward_mae, triple_barrier, mfe_mae_ratio,
  future_expectancy, quantile_future, local_min_followthrough and breakout_gbb. The rule-derived
  labels (technical_indicators, indicators2–4, trends), risk_adj_future, future_sharpe and
  future_sortino are excluded.
- **FR-003**: Each candidate's parameters MUST be declared once, from its labeler defaults or the
  existing arm and oracle classes, at the shared horizon of 48 bars and the shared ±2% thresholds
  where the labeler uses them. Parameters are never tuned on screen results (one shot per label).
- **FR-004**: Each candidate MUST be a one-variable change from the control: same architecture,
  features, sequence length, training rows, sell-side handling (each labeler's own sell variant),
  seed and exits; only the label differs.

**Model and folds**

- **FR-005**: The screen MUST use one model architecture, ARO (plain LSTM on raw data; the
  gbb_fwdmae weights already exist on it). GBR is used only to repeat an S3 winner.
- **FR-006**: The screen (S0–S2) MUST train and read early folds f00–f07 only. Late folds f08–f14
  are trained and read only for the S2 winner at S3.
- **FR-007**: Training MUST run as one sequential, resumable chain, one fold at a time, refusing to
  start while another trainer or backtest chain runs.
- **FR-008**: Each candidate's weights MUST live under their own model key. Every existing model
  key, fingerprint, weight file and per-fold trades hash MUST be unchanged, and the classes that
  exist today MUST behave byte-identically.

**Stages**

- **FR-008a (S0a, label audit before training)**: Before the pre-registration is committed, each
  candidate's labels (no model) are computed on every early fold's training window: BUY / SELL /
  HOLD shares after the peak filter and conflict resolution, raw BUY–SELL overlap, whether each
  declared parameter changes the labels, and whether the feature columns equal the control's. A
  candidate whose kept BUY share is under 1% of training rows in any early fold is INCONCLUSIVE and
  is not trained. (Planning found mfe_mae_ratio has no sell variant, so its raw SELL labels equal
  its BUY labels; this audit puts its trainability on record.)
- **FR-009 (S0, label and instrument)**: Per candidate and fold, record the BUY and SELL positive
  rates, class balance of kept training rows, and Spearman(label, bb_width) on rule-eligible bars.
  Verify the label parameters actually reached the labeler (a declared parameter that changes
  nothing is a failure). S0 fails a candidate (INCONCLUSIVE) only for missing weights, a parameter
  that did not take effect, or no positive rows.
- **FR-010 (S1, separation)**: On the identical early-half width-book trades, compute each
  candidate's agree − disagree pp per trade at the pre-registered agree share (threshold picked per
  fold on the first bar of each width-book candidate run in the validation tail), and its
  within-candidate AUC against each trade's realised outcome. Report the control's figures beside
  each.
- **FR-011 (survivors)**: A candidate survives S1 only if its early gap exceeds both the control's
  pre-registered gap and the control's gap at the same agree share (the control scored with the
  same picker), with at most 2 of 8 folds degenerate;
  survivors are the top 3 by gap among those. The rule is applied mechanically and its output is
  committed before any survivor backtest.
- **FR-012 (S2, early backtest)**: Each survivor is backtested in two roles, each against the
  width book and the 24 h control:
  - **sizer**: the width book's entries, staked by the model's agree / disagree decision;
  - **entry source (ungated)**: the control's own entry configuration (model decision plus the
    control's guards, no gbb trigger gate) with only the label changed. Its threshold is picked per
    fold on the validation tail so that its entry-signal bar rate equals the control's pre-registered
    rate (0.014735, as in 032). breakout_gbb is scored in the sizer role only.

  A role passes only if its early pp beats both comparators; a margin under 1 pp is resolved by a
  paired seed-2 repeat (pass only if every margin is positive on both seeds).
- **FR-013 (S3, late confirmation)**: The single best S2 arm (largest minimum margin) is trained on
  f08–f14 and judged with judge_one against the control plus late pp against the width book, with
  the small-margin and seed-2 rules of 032/033. A pass is repeated on the second architecture over
  all 15 folds.
- **FR-014**: A failed stage stops that candidate or arm; later stages are recorded as not reached
  and their records are not read. Outcomes are PASS (PROVISIONAL), FAIL or INCONCLUSIVE; nothing is
  adopted or promoted by this feature.

**Instrument safeguards**

- **FR-015**: Sizer arms MUST keep exactly the width book's entries (≤ 2 of the read folds may
  differ), tag every entry agree or disagree from the model's own decision, deploy capital within
  ±10% of the width book, and report realised stake ratios and wallet shortfalls.
- **FR-016**: Any script scoring fold strategies outside a backtest MUST load the fold's saved
  parameters and run one fold per process.
- **FR-017**: Thresholds MUST be stored and loaded at 4 decimals on the new arm classes only.
- **FR-018**: The judge MUST refuse to run if the pre-registration is uncommitted or any arm record
  predates its latest commit (smoke records set aside are exempt by name).

**Constraints**

- **FR-019**: Everything new is default OFF; Framework/ stays strategy-agnostic; parameter dicts
  merge from the parent; no pair-specific code; the exit-parameter guard test keeps passing; the
  live dry-run bot is not touched.

### Key Entities

- **Candidate label**: a label method plus its declared parameters; owns one model key per fold.
- **Pre-registration**: the committed record of candidates, parameters, folds, cutoffs and rules.
- **Screen table**: per candidate, the S0 figures, S1 gap and AUC, and survivor status.
- **Arm**: a survivor in one role on one model, with its per-fold backtest records and verdict.
- **Verdict**: per candidate and arm, the stage reached, the decision and the evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every declared candidate has a recorded S0 and S1 result (or a stated reason it is
  INCONCLUSIVE) on all 8 early folds.
- **SC-002**: The control's S1 gap reproduces the pre-registered value within ±0.002 pp per trade,
  showing the instrument is the one 033 used.
- **SC-003**: Existing classes produce byte-identical trades on a sample of existing fold records
  (at least 2 folds of the control), and the full test suite passes.
- **SC-004**: No late-half (f08–f14) P&L is read for any candidate except the S2 winner.
- **SC-005**: The report states, for each candidate label, whether it separates width-book trades
  better than gbb_path, and whether any survivor beats the width book and the control on return.
- **SC-006**: Total compute stays within one sequential chain: at most 8 candidates × 8 folds of
  training for the screen, plus at most 3 survivors × 2 roles × 8 folds of backtests.

## Assumptions

- The control is the existing 24 h-grace model of the chosen architecture (ARO: `wf_aro_g24`; GBR:
  `fs_gbr_gr`) and its width book (`w_aro` / `w_gbr`); their early-half records are already on disk.
- Candidates train on the same rows as the control (all rows), not the 032 eligible-row filter:
  032 showed that filter changes the book, and this feature varies the label only.
- Model exits stay off for sizer arms (033 found they add nothing on the width book).
- The agree share is 0.5 and stakes are 1.333 / 0.667, as in 033.
- future_expectancy is preferred over risk_adj_future (risk_adj_future's oracle ceiling was the
  lowest of those screened, +681% against 3,100–6,800% for the rest); future_sharpe and
  future_sortino are left out as near-duplicates of future_expectancy.
- breakout_gbb is scored only as a ranker (sizer), never as an entry source, because breakout
  labels lose as an entry source on every tested exit.
- The ungated entry-source role is the configuration the control already runs (NNNC's trigger gate
  is off by default), so it is a one-variable comparison with the control.
- One era (2023–26) is available, so any result is PROVISIONAL.
