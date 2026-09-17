# Feature Specification: NNC Classifier Architecture & Input-Data Study

**Feature Branch**: `018-nnc-classifier-study` (spec directory only; work is committed to master per constitution)

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "NNC classifier architecture & input-data study — systematically evaluate the neural classifiers used by the NNC family of strategies, judged on their inputs/outputs against the generated labels, so architectures and inputs are chosen on evidence. No backtests, no P&L."

## Scope Statement

This feature measures **classifiers**, not strategies. Every number it produces is a
classification metric computed from model outputs against the fixed, pre-existing ternary
labels (SELL=0, HOLD=1, BUY=2). Nothing in this feature computes, reports, or gates on
backtest return, drawdown, trade counts, or any trade-level quantity. Constitution
Principle III (learnability is not edge) is acknowledged: the study's outputs are a
**screen** that selects candidate architectures and inputs for later, separate, P&L-based
evaluation; they do not by themselves promote anything to production.

## Clarifications

### Session 2026-09-16

- Q: What total machine-time budget should the whole study fit in? → A: ~40 machine-hours: 3 evaluation windows, 2 seeds for sweeps and 3 seeds for baselines/finalists, epoch cap 40 with patience 10.
- Q: Should models be trained pooled across the 11 production pairs, or per pair? → A: Pooled across the 11 production pairs; no single-pair runs.
- Q: How should the "viable" margin over the ridge baseline be set? → A: Adaptive floor: margin = max(0.02, 2 × seed-std of MCC measured on the ridge baseline in phase 2); stability bound seed-std of MCC ≤ 0.03 per window; majority of windows.
- Q: Where should the study's unit tests live? → A: In a tests/ directory beside the harness, run with plain pytest from the repo root like the existing Framework tests; no new runner.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Harness, baselines and existing classifiers scored on one table (Priority: P1)

As the strategy author, I run one command that trains a named classifier configuration on a
fixed, versioned, leakage-safe walk-forward split of the existing 15m data, evaluates it,
and appends a row of standard metrics to a results table. I run a second command that does
this for a whole phase of configurations and can be interrupted and resumed. With that in
place I score (i) trivial baselines, (ii) a linear ridge baseline, (iii) a strong cheap
non-neural baseline, and (iv) every classifier type that already exists, all at their
current default configuration, and read a first leaderboard that tells me how large the gap
between the baselines and the existing neural classifiers actually is.

**Why this priority**: Everything else depends on the harness, and the baseline-vs-existing
gap is itself the most decision-relevant single number in the study: if existing
classifiers do not clear the ridge baseline, the later phases change shape.

**Independent Test**: Run the phase-1..3 command on the default split set with the default
seed count and confirm the results table contains one row per (configuration, evaluation
window, seed) with every mandatory metric populated, and that the leaderboard ranks
baselines and existing types together.

**Acceptance Scenarios**:

1. **Given** the default data set, split set and a baseline configuration, **When** the
   single-experiment command is run, **Then** exactly one results row per (window, seed) is
   appended, each containing every mandatory metric, the parameter count, training
   wall-clock and inference latency.
2. **Given** a results table already containing rows for some configurations of a phase,
   **When** the run-phase command is invoked, **Then** only configurations whose
   (configuration, window, seed) keys are absent are trained, and the rest are reported as
   skipped.
3. **Given** the ridge baseline is registered as a first-class classifier type, **When** it
   is selected by an existing NNC strategy's classifier-type attribute, **Then** it is
   built, trained and scored through the same path as every neural type with no strategy
   code changes.
4. **Given** any evaluation window, **When** the scaler and class weights used for that
   window are inspected, **Then** they depend only on bars from that window's training span,
   and the training span ends at least one embargo gap before the window starts.
5. **Given** the completed phase-3 results, **When** the leaderboard is produced,
   **Then** it shows, for every configuration, the seed mean and seed standard deviation of
   each headline metric on each window separately and pooled across windows.

---

### User Story 2 - Family sweeps and compositions on the viable set (Priority: P2)

As the strategy author, I sweep each core architecture family in isolation on a coarse size
grid, gate the families on the phase-3/phase-4 leaderboard against the stated viability rule,
and then evaluate common compositions built only from viable families, so I learn which
size and structural choices actually move the metrics and which are habit.

**Why this priority**: This is the core of the architecture question and produces the
recommendation, but it is only meaningful once User Story 1 has established the bar.

**Independent Test**: Run the phase-4 command, apply the gate rule to its leaderboard,
record the viable set, run phase 5 on that set, and confirm each family's sweep rows carry
parameter count and timing so cost/benefit can be read off the table.

**Acceptance Scenarios**:

1. **Given** phase-4 results, **When** the gate is evaluated, **Then** the viable set and
   the per-configuration pass/fail against each clause of the viability rule are recorded in
   the report, and no phase-5 configuration references a non-viable family.
2. **Given** a family sweep, **When** its rows are listed, **Then** every row records the
   model parameter count, training wall-clock and inference latency alongside the metrics.

---

### User Story 3 - Literature-derived candidates (Priority: P3)

As the strategy author, I get a short, cited literature review of recent time-series
classification architectures and LLM-era transformer internals, a ranking of candidates by
expected benefit against implementation cost, and results for the top few, so I can decide
whether any recent idea deserves adoption rather than guessing.

**Why this priority**: Highest novelty, highest implementation cost, and only worth
running once the harness and the incumbent architectures have been measured.

**Independent Test**: The research document lists every candidate with a source, an
applicability verdict, a rank, and either results rows or a "considered, not tested" reason.

**Acceptance Scenarios**:

1. **Given** the candidate list, **When** the research document is read, **Then** each
   candidate has at least one citation, an explicit statement of whether it applies to a
   16-to-128-bar, roughly 24-feature, 3-class problem, and a rank.
2. **Given** an implemented candidate, **When** it is scored, **Then** it appears on the
   same leaderboard, on the same splits and seeds, as every other configuration.

---

### User Story 4 - Input-data study (Priority: P3)

As the strategy author, I learn whether the hand-built indicators are necessary, which
indicator groups and individual indicators matter, whether a learned per-feature importance
agrees with ablation, and how sensitive the best models are to lookback length.

**Why this priority**: Directly answers the "are indicators needed" question, but must be
run only on architectures already shown viable to keep compute bounded.

**Independent Test**: The report contains, for each viable architecture, a raw-vs-engineered
table, a per-group ablation table, a per-indicator permutation-importance table for the best
model, a learned-importance table, an agreement statement between learned and ablation
importance, and a window-length sensitivity table.

**Acceptance Scenarios**:

1. **Given** the three input sets (raw normalised OHLCV only, full indicator set, both),
   **When** each viable architecture is scored on each, **Then** the leaderboard shows all
   three on the same splits and seeds.
2. **Given** the best model, **When** each input feature is permuted in the evaluation
   window, **Then** the drop in each headline metric is recorded per feature and per
   indicator group.
3. **Given** a learned-selection model, **When** it is trained once, **Then** it yields a
   per-feature importance score, and the report states the rank correlation between that
   score and the permutation-importance ranking.

---

### User Story 5 - Results report (Priority: P2)

As the strategy author, I receive a written report with a ranked leaderboard, per-phase
findings, recorded gate decisions, the recommended architecture(s) and input set for the NNC
strategies, concerns about the label definition (noted, not acted on), and a short list of
open questions.

**Why this priority**: The report is the decision artefact; without it the results table is
data, not evidence.

**Independent Test**: The report file exists and contains each required section with
numbers traceable to results-table rows.

**Acceptance Scenarios**:

1. **Given** the completed phases, **When** the report is generated, **Then** every
   leaderboard number can be traced to a results-table row by configuration id, window id
   and seed.

---

### Edge Cases

- A pair has under 365 days of history before a window's validation slice (e.g. a pair
  listed in 2023): that pair is excluded from that window only, and the exclusion is recorded
  in the run row.
- A class is absent from an evaluation window: per-class metrics for that class are recorded
  as undefined (not zero), the run is flagged, and the window remains in the report.
- Training diverges (non-finite loss) or a model collapses to a single predicted class: the
  row is recorded with a failure/collapse flag rather than silently as a result, and a
  collapsed model is never "viable".
- The harness is interrupted mid-phase: re-running the phase command resumes without
  re-training completed (configuration, window, seed) keys.
- Two configurations differ only in a field that does not change behaviour: they must still
  receive distinct configuration ids so nothing is silently skipped.
- A configuration's metrics are undefined because the model predicts no positive for a class:
  AUROC and log-loss are still defined from probabilities and are reported.

## Requirements *(mandatory)*

### Functional Requirements

**Harness**

- **FR-001**: The harness MUST accept one model configuration and one data configuration and
  perform train → evaluate → append-row as a single command invocation.
- **FR-002**: The harness MUST provide a "run whole phase" command that takes a list of
  configurations, runs each, and skips any (configuration id, window id, seed) already
  present in the results table.
- **FR-003**: Every configuration MUST be identified by a deterministic id derived from its
  full content (architecture, hyperparameters, input set, window length, split-set id), so
  identical configurations collide and different ones never do.
- **FR-004**: Runs MUST be seeded and the seed recorded on every row. Baselines (phase 2)
  and finalists (the top configurations of phase 7 and the final recommended configurations)
  use N=3 seeds; the existing-type census (phase 3) and the sweep phases (4, 5, 6) use N=2
  seeds. Across-seed standard deviations are computed with ddof=1 and the seed count is
  shown beside every std. A configuration MAY be re-run with
  extra seeds when a gate decision is within the margin, and the seed count is always
  recorded.
- **FR-005**: All experiment code, configurations, results and reports MUST live under
  `user_data/`. Nothing is written to the upstream freqtrade package.

**Data and splits**

- **FR-006**: The study MUST use only data already present locally: the binanceus 15m
  candles for the 11 pairs in the production static pair list (ZEC, XRP, SOL, LINK, NEAR,
  AAVE, SUI, AVAX, LTC, BCH, DOT, all against USDT). No data download is performed.
- **FR-007**: Labels MUST be the existing ternary gbb labels at the production settings
  (horizon 48 bars, ±2% thresholds, conflict resolution on), held fixed for the entire
  study. Concerns about the label definition are recorded in the report, not acted on.
- **FR-008**: Splits MUST be time-ordered with exactly three sequential, non-overlapping
  evaluation windows in the default split set, at least one of which is temporally distant
  from the others (constitution Principle II). Split sets are versioned by id and stored
  with the results.
- **FR-009**: Between the end of any training span and the start of its evaluation window
  there MUST be an embargo gap of at least the label horizon (48 bars). The validation slice
  used for early stopping MUST be the tail of the training span, separated from the
  remainder by the same embargo.
- **FR-010**: Feature scalers, class weights and any feature selection or importance MUST be
  fitted on the training span of the window in question only, and refitted for every
  window. No cached production scaler is reused.

**Baselines and classifier types**

- **FR-011**: Trivial baselines MUST include majority-class and
  "label persists from previous bar", scored through the same metric code.
- **FR-012**: A ridge (ridge-logistic on the flattened window) classifier MUST be added as a
  first-class type in the existing MLX classifier registry so it is built, trained and
  scored through the identical path as every neural type.
- **FR-013**: A strong cheap non-neural baseline (random-convolution-kernel features with
  a ridge read-out, MiniRocket-style) MUST be scored as the bar to beat.
- **FR-014**: Every classifier type already registered in the MLX classifier registry
  (24 at time of writing) MUST be scored at its current default configuration.
- **FR-015**: Every new classifier type added by this feature MUST be selectable by an
  existing NNC strategy through its existing classifier-type attribute without strategy
  code changes.

**Metrics** (all computed on the evaluation window, class order SELL, HOLD, BUY;
hard predictions are the argmax of the model's class probabilities)

- **FR-016**: Balanced accuracy MUST be the unweighted mean over the three classes of
  per-class recall.
- **FR-017**: Macro-F1 MUST be the unweighted mean over classes of per-class F1, where a
  class with zero predicted and zero true positives contributes F1 = 0.
- **FR-018**: MCC MUST be the multiclass Matthews correlation coefficient (Gorodkin
  generalisation) computed from the 3×3 confusion matrix.
- **FR-019**: Per-class precision and recall MUST be reported for each of the three classes.
- **FR-020**: Log-loss MUST be the mean negative log probability of the true class, with
  probabilities clipped to [1e-15, 1−1e-15].
- **FR-021**: AUROC MUST be one-vs-rest per class and its unweighted macro mean.
- **FR-022**: Calibration MUST be reported as Expected Calibration Error using 15
  equal-width confidence bins on the maximum class probability, together with the per-bin
  (confidence, accuracy, count) table.
- **FR-023**: The 3×3 confusion matrix (rows = true, columns = predicted) MUST be stored on
  every row.
- **FR-024**: Every row MUST also record: trainable parameter count, training wall-clock
  seconds, epochs run, inference latency per window (mean milliseconds per input window at
  the harness batch size, and separately at batch size 1), and the counts of each predicted
  class.
- **FR-025**: As a secondary, decision-relevant view, every row MUST record BUY-class
  precision and coverage (fraction of windows predicted BUY) at the production decision
  threshold of 0.8 on the BUY probability.
- **FR-026**: Aggregations MUST report, for every configuration, the mean and standard
  deviation over seeds of each metric on each window separately, and a pooled figure
  defined as the mean over windows of the per-window seed means.

**Phases and gates**

- **FR-027**: The study MUST proceed in the seven phases listed in the feature description.
  Phase 3 (every existing type) runs unconditionally once the phase-2 baselines exist, and
  its results are reviewed with the author at a recorded checkpoint before any sweep. Each
  of phases 4, 5, 6 and 7 MUST be preceded by a recorded gate decision naming the viable set
  carried forward, the rule applied and the margin value used.
- **FR-028**: A configuration is **viable** when, on a majority of evaluation windows, its
  seed-mean MCC exceeds the ridge baseline's seed-mean MCC by at least the viability margin
  AND its seed-mean macro-F1 exceeds the ridge baseline's by at least the same margin, AND
  its across-seed standard deviation of MCC on every window is at most the stability bound
  (0.03), AND it is not flagged collapsed or failed on any window. The viability margin is
  adaptive: margin = max(0.02, 2 × the pooled across-seed standard deviation of MCC measured
  on the ridge baseline in phase 2). The margin value actually used MUST be recorded in every
  gate decision. Both the floor (0.02) and the stability bound are parameters of the gate
  step.
- **FR-029**: Phase 4 sweeps MUST cover, per family, at least three sizes on a coarse
  logarithmic grid, and for each family at least one structural axis (depth,
  bidirectionality, kernel/dilation, heads/layers, positional encoding as applicable).
- **FR-030**: Phase 5 compositions MUST be built only from families viable after phase 4.
- **FR-031**: The literature review MUST cite sources for every candidate, verify
  applicability to this problem size, rank candidates by expected benefit against
  implementation cost, and record every un-implemented candidate as "considered, not tested"
  with the reason.
- **FR-032**: The input-data study MUST evaluate raw normalised OHLCV only, the full
  indicator set, and both; per-indicator-group ablation; per-indicator permutation
  importance on the best model; at least one learned single-run feature-importance method
  with its agreement to permutation importance reported; and window-length sensitivity on
  at least three lengths including the production length (16).

**Deliverables and tests**

- **FR-033**: Deliverables MUST include the harness and configurations, the results table,
  a Markdown report with tables and plots, and the literature review, all under `user_data/`.
- **FR-034**: Unit tests MUST cover: split construction (no overlap, correct embargo),
  the leakage guard (scaler and class weights fitted on train only), metric computation
  against known values, and that every registered classifier type can be built, trained and
  scored on a tiny synthetic data set.
- **FR-035**: The report MUST contain a ranked leaderboard, per-phase findings, every gate
  decision, the recommended architecture(s) and input set, label-definition concerns, and a
  short list of open questions.
- **FR-036**: No artefact of this feature MAY compute, report or gate on backtest return,
  P&L, drawdown, trade counts or any trade-level metric.
- **FR-037**: Neural training in the study MUST cap epochs at 40 with early-stopping
  patience 10 (instead of the production 100 / 20), applied identically to every type, so
  the whole study fits within roughly 40 machine-hours. The cap and patience MUST be recorded
  on every row.

### Key Entities

- **ExperimentConfig**: A fully specified experiment — architecture type, its
  hyperparameters, input set id, window length, split-set id — with a deterministic id.
- **SplitSet**: A versioned list of evaluation windows, each with its training span,
  validation slice, embargo and evaluation range; identified by id.
- **RunResult**: One row: config id, split-set id, window id, seed, all metrics, timing,
  parameter count, status flags (ok / failed / collapsed / class-absent), pair exclusions.
- **Phase**: A named, ordered list of ExperimentConfigs plus the gate rule that admits the
  next phase.
- **GateDecision**: The recorded outcome of applying the viability rule to a phase's
  aggregated results: viable set, per-config clause results, date.
- **Leaderboard**: The aggregated view of RunResults (seed mean/std per window, pooled).
- **Report**: The Markdown decision document referencing Leaderboard and GateDecisions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single command scores any configuration end to end, and a second command
  runs a full phase and can be resumed with zero re-training of completed keys.
- **SC-002**: 100% of results rows contain every mandatory metric, timing field and
  parameter count, or an explicit status flag explaining the absence.
- **SC-003**: Automated tests demonstrate that perturbing evaluation-window bars leaves the
  fitted scaler and class weights unchanged, and that every window's training span ends at
  least 48 bars before the window begins.
- **SC-004**: Every classifier type in the registry, including those added here, passes the
  synthetic build/train/score test.
- **SC-005**: The ridge type is selectable from an existing NNC strategy with no strategy
  code change, verified by a test.
- **SC-006**: Each of phases 4 to 7 has a recorded gate decision in the report before its
  results appear, and the phase-3 checkpoint review is recorded in the report.
- **SC-007**: The report presents per-window and pooled seed-mean ± std for every
  configuration, and every leaderboard number is traceable to results rows.
- **SC-008**: The literature review has a citation, applicability verdict and rank for
  every candidate listed in the feature description.
- **SC-009**: A text search of every deliverable for backtest, P&L, drawdown or trade-level
  terms returns only the scope statement and label-concern notes.
- **SC-010**: The report names a recommended architecture set and input set with the
  evidence rows that support each recommendation.

## Assumptions

- **Pooled multi-pair training**: one model is trained on the 11 production pairs
  concatenated, matching how the production classifier is trained. Single-pair training is
  not part of this study (confirmed in clarification).
- **Data span** starts 2022-01-01 (the production training start). A pair is included in an
  evaluation window only if it has at least 365 days of bars before that window's validation
  slice; otherwise it is excluded from that window entirely and the exclusion is recorded.
- **Default split set (v1)**: three evaluation windows of six months each — one temporally
  distant window in 2023 and two later windows inside 2025-01 to 2026-06 — with an anchored
  expanding training span from 2022-01-01, embargo of 96 bars (2× horizon) as a safety
  margin over the 48-bar minimum, and a validation slice equal to the final 10% of each
  training span. Exact dates are fixed in the plan.
- **Seeds**: N=3 for baselines and finalists, N=2 for sweep phases (FR-004); raised for
  close calls at the author's discretion, and the seed count is always recorded.
- **Training procedure** for neural types is the existing one (optimizer, focal loss with
  inverse-frequency class weights, early stopping on validation MCC, checkpoint restore),
  applied identically to every type so that architecture is the only variable.
- **Compute budget**: about 40 machine-hours for the whole study (confirmed in
  clarification). Grounding: one pooled training at production settings costs roughly 5 to
  7 minutes; with the 40-epoch cap, 3 windows and the seed policy in FR-004, phases 1 to 3
  complete in one overnight run and the sweep phases in the remainder.
- **Tests** live in a tests/ directory beside the study harness and run with plain pytest
  from the repo root, like the existing Framework tests (confirmed in clarification).
- The Keras classifier sibling is out of scope; only the MLX path is studied.
- Existing evaluation tooling (fold planner, learnability debugger) is reused where it fits
  rather than duplicated.
