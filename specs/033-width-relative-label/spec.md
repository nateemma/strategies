# Feature Specification: Width-Relative Label for NNNC Models

**Feature Branch**: `033-width-relative-label` (spec directory; work is committed to master)

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Width-relative label for NNNC (feature 033). [...] Goal: a label asking 'did this candidate do better than its width would predict?', so the model learns information orthogonal to width, then is used where width cannot help: ranking/filtering inside the width book. [...] Arms (proposed): (A) width-book filter [...] (B) width-book sizer [...] optional (C) model as entry source [...] Evaluation, staged and pre-registered BEFORE any model is trained: S0 instrument [...] S1 within-width-book separation [...] S2 early half [...] S3 late half [...]" (full text in research/candidate_training/spec_033_draft.md and the conversation of 2026-10-14)

## Background *(diagnostic evidence; research/model_calibration/findings.md §1–7, research/candidate_training/report.md)*

**On the rule's candidates, the model mostly sorts by band width.** About 20% of bars pass the gbb
rule (guard_metric < 0.3 AND bb_width > 0.035), and about half of those carry a positive label. On
those bars the NNNC models score only 0.53–0.62 AUC.

A model-free **width book** takes the rule's candidates whose band width is at or above the previous
fold's count-matched cut. In real 15-fold backtests it matches or beats every model book: the width
minus model difference is ARO +1.48, GBR +0.22, GFM +24.31 and MLX −1.09 pp, all paired UNCHANGED.

**But inside the width book, the current model still separates trades.** Width-book trades the
model agrees with earn far more than those it rejects:

| | agree − disagree pp per trade, early / late |
|---|---|
| ARO | 0.072 / 0.085 |
| GBR | 0.063 / 0.072 |

**Feature 032 removed the rule from training, but not width.** The candidate-only models were
trained on eligible rows only:
- their within-candidate AUC rose (GBR +0.031 in both halves), and their scores became less
  width-correlated (GBR Spearman with bb_width 0.58 → 0.30);
- their books failed the early half (−10 to −15 pp against the control and the width book);
- as a sizer on the width book they separated worse than the current model;
- run-matched thresholds recovered ARO's trade count, but the extra trades were poor (late 5.5 vs
  14.7 pp).

**The label still rewards width.** The gbb_path label is more often positive on wide-band
candidates, so part of what any model learns from it is width, which the width book already uses
directly.

**Lessons from 032 for the instrument:**
- the entry threshold only takes 2-decimal values, so picked thresholds are rounded when loaded;
- scoring a fold strategy outside a backtest needs the fold's saved parameters loaded explicitly,
  and parameter values can leak from one load to the next in the same process;
- a whole-window threshold pick in one process was stopped by the host for low memory.

## Clarifications

### Session 2026-10-14

- Q: Which forward outcome does the label measure? → A: The realised return of the gbb_path trade:
  +2% target or −5% stop, whichever is hit first, else the close return at 48 bars.
- Q: Which arm is judged in the walk-forward stages? → A: The sizer. Every width-book trade is
  kept; the stake is larger when the model agrees and smaller when it does not (1.0 : 0.5), scaled
  so the arm deploys the same average capital as the width book.
- Q: Threshold precision? → A: 4 decimals, on the new arm classes only. Existing classes and
  records are untouched.
- Q (raised after planning: earlier studies found model exits helped at open gates, +10.4 pp on
  NNMT; on NNNC they were +0.03 pp at the tight gate and −5.15 pp with the exit gate also opened)
  → A: The label covers exits by mirroring the BUY side. Exit-eligible bars (the gbb_path_sell
  rule) are ranked within their own width buckets by the sell-path outcome, and the worst half is
  SELL. One three-class model, trained on entry- plus exit-eligible rows, serves both arms.
- Q: How is the exit-on variant judged? → A: Co-primary. The sizer with model exits off and the
  sizer with model exits on each go through the pre-registered stages and get their own verdict.
  The exit-on arm is judged against the exit-off arm and the width book.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A default-off width-relative label (Priority: P1)

A researcher declares, on a new model class, a width-relative label. Each rule-eligible training
bar gets a forward outcome. The outcome is compared with what bars of similar band width achieved
in the same fold's training window. An entry-eligible bar is BUY if it did better than half of the
entry bars in its width bucket. An exit-eligible bar is SELL if holding through it did worse than
half of the exit bars in its width bucket. Without the declaration, labelling is unchanged.

**Why this priority**: Nothing can be trained until the label exists, and it must leave every
existing model, weight file and recorded result untouched.

**Independent Test**:
- With no width-relative label declared, the training labels for a scripted multi-pair frame are
  identical to the build captured before the change, and every existing model key and strategy
  fingerprint is unchanged.
- With it declared, on a scripted frame with known outcomes and widths, every bar's label equals
  the hand-computed "in the top half of its bucket's outcomes" (with the declared tie-break).

**Acceptance Scenarios**:

1. **Given** no width-relative label declared, **When** training labels and data are built,
   **Then** they are identical to the pre-change capture. All existing model keys and fingerprints
   are unchanged.
2. **Given** the label declared, **When** labels are built for one fold, **Then** the bucket
   boundaries and rankings are computed from that fold's training window only, pooled across
   all its pairs. No test-window bar contributes.
3. **Given** the label declared, **When** labels are built, **Then** a bar is positive exactly
   when its forward outcome ranks in the top declared share (default half) of its bucket. Ties are
   broken by speed: a target hit in fewer bars ranks higher, a stop hit in more bars ranks higher,
   then by position. Otherwise the bar is HOLD.
3b. **Given** the label declared, **When** labels are built, **Then** an exit-eligible bar
   (guard_metric > 0.3 AND bb_width > 0.035, the gbb_path_sell rule) is SELL exactly when its
   sell-path outcome ranks in the bottom declared share (default half) of its exit-side width
   bucket. Ties are broken by speed: a fall reached in fewer bars ranks lower, a take-profit reached
   in more bars ranks higher, then by position. Otherwise the bar is HOLD. Entry- and exit-eligible
   bars are disjoint, so no bar can be both BUY and SELL.
4. **Given** the label declared, **When** labels are built, **Then**, on each side (BUY among
   entry-eligible bars, SELL among exit-eligible bars), the labelled share in every width bucket is
   between 40% and 60%. The rank correlation of each side's label with band width on its eligible
   bars is within ±0.05.
5. **Given** the label declared, **When** the model key is computed, **Then** it differs from
   every existing key and changes with each label setting (outcome, bucket count, share).
6. **Given** a bar whose forward outcome cannot be computed (the horizon runs past the end of the
   data), **When** labels are built, **Then** it is excluded from training, never labelled
   negative.
7. **Given** the label declared, **When** training runs, **Then** it trains on entry- or
   exit-eligible rows with a valid outcome only (the 032 training row filter, extended to the
   union), with the validation split filtered the same way.

---

### User Story 2 - Measure whether the new model separates trades inside the width book (Priority: P2)

The researcher trains the width-relative versions of ARO and GBR over the 15 walk-forward folds.
The researcher then measures, on the identical width-book trades, whether the new model separates
winners from losers better than the current model.

**Why this priority**: This is the cheap mechanism check, read before any book's P&L. If the new
model separates no better than the current one inside the width book, the arms cannot do better
than the width book either.

**Independent Test**: A report with, per model and half:
- the label sanity numbers per fold;
- the new and the current model's agree/disagree split on the identical width-book trades;
- the instrument checks.

**Acceptance Scenarios**:

1. **Given** the trained folds, **When** the label is audited, **Then** each fold reports, per side,
   its labelled share per width bucket and the label's rank correlation with width, against the
   US1 scenario 4 bounds.
2. **Given** the new model's per-bar scores on the test folds, **When** the width-book trades are
   split by whether the model fires, **Then** the agree − disagree pp per trade gap is reported
   next to the current model's (ARO 0.072 / 0.085, GBR 0.063 / 0.072) on the same trades.
3. **Given** any book is read, **When** the instrument is checked, **Then**:
   - the exit-off arm reproduces the width book's trades fold by fold (same pairs, dates and
     profit ratios); any difference is reported as a wallet or stake effect;
   - 0 entries in either arm violate the width book's entry rule;
   - the exit-on arm records model exits in every fold where its exit gate passes (liveness), with
     the count per fold reported;
   - every fold's saved parameters are verified as loaded.

---

### User Story 3 - Pre-registered walk-forward judgement (Priority: P3)

The researcher commits a pre-registration of every stage rule and operating point BEFORE any model
is trained. Each of the two co-primary arms' walk-forward records is then judged, early half
first, late half confirmatory:
- the exit-off sizer against the width book and the model control;
- the exit-on sizer against the exit-off sizer and the width book.

**Why this priority**: Only the confirmatory stage decides the verdict, and it is read only if the
earlier stages pass.

**Independent Test**: The committed pre-registration, the arm records (15 folds per model), and the
verdicts.

**Acceptance Scenarios**:

1. **Given** a committed pre-registration, **When** the judge runs, **Then** it refuses if any arm
   record predates the commit or the pre-registration has uncommitted changes.
2. **Given** the stages, **When** they are read, **Then**:
   - per arm, S0 (instrument and label sanity), S1 (separation inside the width book; exit-off arm
     only), S2 (early half f00–f07, the arm's pp against its two comparators) and S3 (late half
     f08–f14, judged against both) are read in that order;
   - a failed stage leaves that arm's later stages recorded as not reached, and their records
     unread;
   - the arms are judged independently. An S1 failure stops the exit-off arm only, because the
     exit-on arm's question (do model exits add value on top of the sizer?) does not depend on it.
3. **Given** a stage passes by less than 1 pp, **When** it is judged, **Then** it is INCONCLUSIVE
   until a paired seed-2 repeat.
4. **Given** the run ends, **When** the report is written, **Then**:
   - each model and arm gets PASS (PROVISIONAL), FAIL or INCONCLUSIVE, naming the stage that stopped
     it;
   - a pass names its forward-confirmation route: folds after 2026-09-07, or dry-run;
   - a negative result is recorded with full evidence.

---

### Edge Cases

- **Sparse width buckets.** A bucket with too few eligible training bars to rank reliably (fewer than 500 bars) is merged with its neighbour, and this is reported.
  A fold where this happens is not dropped.
- **Pairs that enter mid-span.** A pair's bars count only where it has data. Buckets are pooled
  over whatever pairs are present in the fold's training window.
- **Ties at the cut.** The outcome is capped at the +2% target, so many bars tie. Ties are broken
  deterministically by speed to exit, then by position (US1 scenario 3). Plan research R3 shows why
  a "beats the bucket mean" rule would keep the label width-correlated: the hit rate rises with
  width, from 0.45 to 0.59 across deciles.
- **Overlapping outcomes.** Consecutive eligible bars on one pair have overlapping forward windows.
  Their outcomes are correlated. This is accepted, as for every forward label today, and is not
  treated as independent evidence in any stage.
- **Sizing inside the width book.** The arm takes exactly the width book's trades, so trade count and
  timing match the width book. Only the stakes differ. The agree share is pre-registered, not tuned
  on the books.
- **Stake limits.** An agree trade's stake is above the width book's base stake (about 1.33× at a
  50% agree share). If the wallet or a liquidity cap cannot fill it, the shortfall is counted and
  reported per fold. A fold where it binds on more than 5% of trades is flagged.
- **Agree share drifts out of sample.** The threshold is set on training data, so the test-window
  agree share can differ from the target. It is reported per fold, and the capital normalisation
  uses the pre-registered share, not the realised one. Any capital difference against the width
  book is therefore reported, not hidden.
- **Held-pair add.** The held-pair add is off in the arms and in all comparators, as in 032.
- **Exit gate.** Model exits pass through the controls' existing exit gate (rvol > 2.0 AND
  guard_metric > −0.1, active because entry guards are enabled), unchanged. That is the gate NNNC_MLX
  already trades with exits on. Tuning or opening it is out of scope (one parameter space at a time;
  opening it made exits worse in the COCONUT test).
- **Earlier exits change the trade set.** In the exit-on arm, an earlier exit frees the pair sooner,
  so its entries can differ from the width book's. Its instrument checks are entry-rule violations
  and exit liveness, not trade identity.
- **Sparse SELL opportunities.** The exit gate's rvol > 2.0 passes few bars, so model exits may fire
  rarely. The count per fold is reported. A fold with exit-gate-passing bars but zero model exits
  is flagged as a liveness failure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The shared labelling path MUST accept an optional, strategy-declared width-relative
  label. When none is declared, training labels, data, model keys, fingerprints and every recorded
  result MUST be unchanged.
- **FR-002**: The label's forward outcome per entry-eligible bar MUST be the realised return of
  the gbb_path trade from that bar: the +2% target or the −5% stop, whichever the path reaches
  first, else the close return at 48 bars (Clarification Q1). Per exit-eligible bar, it MUST be the
  mirror, the gbb_path_sell path: a 2% fall (NNNC's MIN_SELL_LOSS_THRESHOLD) or the 2.2% take-profit, whichever comes first, else
  the close return at 48 bars. The targets, stops and horizon are the controls' label parameters.
- **FR-003**: The width reference MUST be the bar's band-width quantile bucket: its outcomes
  ranked with the declared tie-break. It uses 10 equal-count buckets, computed per fold from that fold's training window only, pooled
  across pairs, separately for entry-eligible and exit-eligible bars.
- **FR-004**: An entry-eligible bar MUST be labelled BUY if and only if it ranks in the top declared
  share (default 0.5) of its bucket, by outcome, then the speed tie-break, then position. An
  exit-eligible bar MUST be labelled SELL if and only if it ranks in the bottom declared share
  (default 0.5) of its exit-side bucket. Every other kept bar is HOLD. (Refined at planning from
  "mean plus margin": plan research R3.)
- **FR-005**: The outcome measure, bucket count, share and eligibility rule MUST be declared by
  the strategy (the NNNC side), not written into the shared code, so the shared code stays
  strategy-agnostic. Pooling across pairs MUST be applied where the training window's pairs are
  assembled, not inside a per-pair label function.
- **FR-006**: Training MUST use the 032 training row filter, keeping entry- or exit-eligible rows
  with a valid outcome, with the validation split filtered the same way.
- **FR-007**: Declaring the label MUST change the model key of that class only, and the key MUST
  reflect each label setting. Existing classes' model keys MUST be unchanged. The model-key audit
  MUST classify every new setting.
- **FR-008**: The arms MUST be the sizer (Clarification Q2), in two co-primary variants: model
  exits off and model exits on (the controls' exit gate unchanged). Both use the same weights,
  thresholds and stakes. The sizer:
  - it takes exactly the width book's trades;
  - each trade's stake is 1.0 when the model agrees at the trade's signal bar and 0.5 when it does
    not, scaled by a per-fold constant;
  - the constant is chosen so the expected mean stake equals the width book's, at the
    pre-registered agree share.

  This keeps total pp directly comparable with the width book. The agree share is pre-registered
  (default 50%, matching the label's ~50% positive rate) and set on the training window only. The
  achieved agree share and mean stake are reported per fold.
- **FR-009**: Each fold's threshold MUST be chosen from that fold's training data only, with the
  fold's saved parameters loaded as a backtest loads them. The new arm classes MUST carry the
  threshold at 4 decimals (Clarification Q3). Existing classes keep 2 decimals, so their keys,
  fingerprints and records are unchanged. The threshold the arm loads MUST be verified equal to the
  picked value.
- **FR-010**: Arm classes MUST be built for ARO and GBR on all 15 folds, at 24 h grace with the
  held-pair add off. Everything except the label, the training filter, the entry rule and the
  threshold MUST match the 24 h controls (wf_aro_g24, fs_gbr_gr).
- **FR-011**: A label audit MUST report, per fold, the positive rate per width bucket, the label's
  rank correlation with width, and the kept training rows and class counts.
- **FR-011b**: The exit-on arm MUST report, per fold, the count of model exits, and the count of
  exit-gate-passing bars during open trades.
- **FR-012**: The S1 measurement MUST compare the new and the current model's agree − disagree pp
  per trade on the identical width-book trades, per model and half.
- **FR-013**: All stage rules and operating points MUST be committed in one pre-registration
  before any model is trained. The stages MUST be read in order; a failed stage leaves the later
  stages recorded as not reached.
- **FR-014**: The walk-forward arms MUST be judged with the existing judge:
  - the exit-off arm against the model control and the width book;
  - the exit-on arm against the exit-off arm and the width book. The judge MUST refuse records that predate the pre-registration.
- **FR-015**: Training and threshold picking MUST run one fold per process, one chain at a time.
- **FR-016**: Promoting any model or arm to a default MUST NOT be part of this feature.

### Key Entities

- **Width-relative label**: the declared label setting:
  - the outcome measure;
  - the bucket count;
  - the share;
  - the eligibility rule;
  - the per-fold bucket boundaries and positive shares it produces.
- **Width bucket**: a band-width quantile range on a fold's eligible training bars, with its count,
  median outcome and positive share.
- **Width-relative model**: a model class with the label declared. It has its own model key and
  weights per fold.
- **Sizer arm**: the per-fold strategy classes that take the width book's entries and set each stake
  from the model's agreement, with the pre-registered agree share and capital normalisation. There
  are two variants: model exits off, and model exits on.
- **Stage rule**: the pre-registered pass condition for one stage (S0–S3).
- **Width book**: the existing model-free baseline (w_aro, w_gbr) that every arm must beat.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With no width-relative label declared:
  - the scripted label and training-data build is identical to the pre-change capture in 100% of
    cases;
  - all model keys and fingerprints pinned in tests are unchanged.
- **SC-002**: On a scripted frame with known outcomes and widths, 100% of labels equal the
  hand-computed value, and no bucket uses a test-window bar.
- **SC-003**: On every trained fold, for each side, the labelled share in every width bucket is
  between 40% and 60%, and the label's rank correlation with width on that side's eligible bars is
  within ±0.05.
- **SC-004**: Weights exist for all 15 folds of ARO and GBR (30 trained folds).
- **SC-005**: The S1 separation (agree − disagree pp per trade inside the width book) is reported
  per model and half next to the current model's, on the identical trades.
- **SC-006**: The instrument holds for both arms:
  - the exit-off arm reproduces the width book's trades per fold, with differences reported as stake
    effects;
  - 0 entries in either arm violate the entry rule;
  - the exit-on arm shows model exits in every fold with gate-passing bars;
  - every fold's saved parameters are verified as loaded.
- **SC-007**: Each stage's outcome (pass, fail or not reached) is recorded against its committed
  rule. The run ends with a verdict per model and arm: PASS (PROVISIONAL), FAIL or INCONCLUSIVE, naming the
  stage that stopped it. A PASS is a provisional result, not a finding (constitution II: one era),
  and names its forward-confirmation route.

## Assumptions

- The entry-eligibility rule is the gbb rule as in 032: guard_metric < 0.3 AND bb_width > 0.035.
  The exit-eligibility rule is the gbb_path_sell rule: guard_metric > 0.3 AND bb_width > 0.035.
- Features, architecture, seed, training spans and grace are unchanged from the controls, so no
  scaler regeneration is needed. The scaler-leak rule (constitution XI) applies as for the controls.
- The width book's own definition and cut are unchanged; it is the baseline, not a variable.
- Evaluation is the 2023–26 walk-forward span. Its early half selects and its late half confirms.
  There is no distant-era window for NNNC models (training needs 730 days from 2021-01-01), so the
  verdict is PROVISIONAL.
- 032's early-half results for other models have been seen. This feature's models and arms are new
  and unseen, and their rules are committed before training.
- GFM and MLX are out of scope here, but can be added later with the same label.
- Out of scope:
  - a filter arm and the model as entry source (032-style); they can be added later on the same
    weights;
  - changing the gbb rule or the width book;
  - regime-conditional labels;
  - architecture changes;
  - promotion to a default.
