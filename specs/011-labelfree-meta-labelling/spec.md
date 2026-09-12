# Feature Specification: labelfree-meta-labelling

**Feature Branch**: `011-labelfree-meta-labelling`

**Created**: 2026-09-12

**Status**: Draft

**Input**: User description: "Feature: labelfree-meta-labelling. This is research/label-free-study brief 03; the feature must be specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented. Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md and briefs/03-meta-labelling-from-trade-archive.md. Build research/labelfree/metalabel/ that (1) extracts entry-time features and realised outcomes for every trade in the pooled walk-forward archive for NNNC_MLX (regime/wf/, regime/arms/), (2) runs the leakage audit, (3) trains the nested walk-forward veto model, and (4) wires the accepted model into a thin NNNC_MLX subclass via confirm_trade_entry with the primary model path pinned. Never let a feature read data after the trade's open_date. Log trials to research/labelfree/trials.jsonl. Write briefs/results/03-<date>.md per protocol §6. Success criteria: the acceptance and kill gates of research/label-free-study/00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md."

## Context

The primary strategy (`NNNC_MLX` + guards + ATR stop) already selects trades with a positive
but concentrated edge. This feature does not touch it. It asks a narrower question: **given
only what was knowable at the moment a trade opened, can a secondary model tell which of the
primary's trades to skip?**

The training set already exists and is out-of-sample by construction: the pooled walk-forward
archive holds **378 trades across 15 folds**, each with its realised `profit_abs` under real
exits, fees and fills. The target is therefore money the strategy actually made — not a
forward-excursion stamp on a bar.

**Measured facts that shape this spec** (from `regime/wf/pooled_nnnc.json`, 2026-09-12):

| quantity | value |
|---|---|
| pooled trades / folds | 378 / 15 |
| warm-up span (folds 0–4) | 91 trades |
| **evaluation span (folds 5–14)** | **287 trades, 10 folds, 895 days = 2.45 yr** |
| base win rate on the evaluation span | **73.2%** (210/287) |
| ZEC share of evaluation trades | 16.0% |
| ZEC share of evaluation P&L | 27.9% |
| trades in the thinnest fold (f14) | 3 |

Two of these are load-bearing and are why several requirements below exist at all:

- **The base rate is 73.2%.** A veto model must beat a majority-class baseline that accepts
  every trade. Accuracy is therefore meaningless here and is excluded as a reported metric.
- **Fold 5's model trains on 91 trades.** The nested design is genuinely thin at the start,
  and a per-fold count must accompany every per-fold figure.

The brief cites ZEC as ≈70% of P&L from an earlier study; on this archive's evaluation span it
is 27.9%. The pair-concentration check (US4) uses the measured figure.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build the trade table and prove it cannot see the future (Priority: P1) 🎯 MVP

A researcher needs one row per primary trade carrying features known at `open_date` and the
realised outcome, with positive evidence that no feature reads past `open_date`.

**Why this priority**: Every later story consumes this table. A leak here does not fail
loudly — it produces an excellent model and a worthless result. This story is also a genuine
standalone deliverable: a leak-audited entry-time feature table over 378 archived trades is
reusable by briefs 5 and 8 whether or not any veto model works.

**Independent Test**: Build the table, run the leakage audit, and confirm every feature passes
its causality check and the audit's planted-leak control is *detected*.

**Acceptance Scenarios**:

1. **Given** the pooled archive, **When** the extractor runs, **Then** it emits exactly one row
   per archived trade, each carrying `fold_id`, `pair`, `open_date`, realised `profit_abs`, and
   the entry-time feature vector.
2. **Given** any feature column, **When** the causality check runs, **Then** it proves the
   feature's value is unchanged when all bars strictly after `open_date` are deleted.
3. **Given** a deliberately planted leaky feature (the trade's own realised outcome), **When**
   the audit runs, **Then** the audit **flags it** — an audit that cannot detect a known leak
   certifies nothing.
4. **Given** a feature that is causal but uninformative, **When** it is permuted across trades,
   **Then** the model's discrimination does not move materially.

---

### User Story 2 - Nested walk-forward veto model (Priority: P2)

A researcher trains a secondary model to predict whether a primary trade will close profitable,
where the model for fold *k* sees only trades from folds `< k`, and measures its discrimination
on folds 5–14.

**Why this priority**: This is the brief's hypothesis. It is separable from US1 (which produces
the data) and from US3 (which only acts on the output).

**Independent Test**: Train the nested model and report per-fold discrimination against the
majority-class baseline, with per-fold trade counts.

**Acceptance Scenarios**:

1. **Given** the nested design, **When** the fold-*k* model is trained, **Then** its training
   rows contain no trade whose `open_date` falls in fold *k* or later.
2. **Given** the evaluation span, **When** discrimination is reported, **Then** it is reported
   against the 73.2% base rate, with the per-fold trade count beside every per-fold figure.
3. **Given** five or more seeds, **When** results are reported, **Then** they are an
   interquartile mean with a bootstrap interval, never a best seed.

---

### User Story 3 - Apply the veto and measure the P&L consequence (Priority: P3)

A researcher applies the veto at a threshold and measures the resulting change in realised
return on the evaluation span, against controls and baselines.

**Why this priority**: Discrimination is not edge. A model can rank trades correctly and still
lose money after it removes winners along with losers.

**Independent Test**: Replay the archived trades with vetoed trades removed, and compare the
realised return to the unvetoed archive over the identical span.

**Acceptance Scenarios**:

1. **Given** a veto threshold, **When** the filtered return is computed, **Then** it is
   reported with its interval against the recomputed floor for the evaluation span.
2. **Given** any reported return, **When** it is published, **Then** trades per year, mean
   holding time and retained-trade count appear beside it.
3. **Given** a veto that removes most trades, **When** results are reported, **Then** a
   configuration retaining too few trades to resolve is reported as inconclusive, not as a
   return.

---

### User Story 4 - Establish whether the veto is pair selection (Priority: P4)

A researcher determines whether the veto's gain is discrimination *within* pairs or simply the
removal of weak pairs — the "implicit pair selector" finding this repo has already recorded.

**Why this priority**: A gain that disappears when the dominant pair is excluded is a
restatement of a known result, not meta-labelling. This story decides which of the two the
result is, and the brief's own kill criterion depends on it.

**Independent Test**: Recompute the veto's gain with ZEC excluded, and separately within each
pair, and compare.

**Acceptance Scenarios**:

1. **Given** the accepted veto, **When** ZEC trades are excluded, **Then** the gain is
   recomputed and reported alongside the all-pairs figure.
2. **Given** the accepted veto, **When** the gain is decomposed within pairs, **Then** the
   report states whether discrimination survives inside pairs.

---

### User Story 5 - Wire the accepted model into the strategy (Priority: P5)

If and only if a configuration is accepted, it is exposed as a thin `NNNC_MLX` subclass that
consults the secondary model at entry, with the primary model path pinned so nothing retrains.

**Why this priority**: Deployment is worthless before acceptance, and this repo has repeatedly
found interventions that look good offline and vanish in the backtest. Gated last, deliberately.

**Independent Test**: Run the subclass over the evaluation span and confirm its trade set
matches the offline veto replay.

**Acceptance Scenarios**:

1. **Given** the subclass, **When** it runs, **Then** the primary model is loaded from the
   pinned path and is not retrained.
2. **Given** the subclass, **When** its trades are compared to the offline replay, **Then**
   any divergence is reported and explained rather than averaged away.
3. **Given** no accepted configuration, **When** the feature closes, **Then** the subclass is
   not written and the report records the kill.

### Edge Cases

- **A fold contains too few trades to evaluate.** f14 has 3. Per-fold figures on such folds
  are reported as counts, never as rates presented without them.
- **The veto removes every trade in a fold.** That fold's return is zero by construction, which
  is not a loss avoided and must not be reported as one.
- **A pair appears in evaluation but never in training.** The model must produce a decision
  without a pair-specific prior, and this case must be counted.
- **A trade's entry bar is missing from the feature source.** The row is excluded and counted,
  never imputed.
- **Two trades open on the same pair and bar.** They are one decision, not two measurements.
- **The veto threshold accepts everything.** This is the null configuration and must appear in
  the results table as the control, not be omitted.

## Requirements *(mandatory)*

### Functional Requirements

**Data extraction and causality**

- **FR-001**: The system MUST emit one row per trade in the pooled walk-forward archive,
  carrying fold, pair, open date, realised profit, and the entry-time feature vector.
- **FR-002**: No feature MAY depend on any data timestamped after that trade's `open_date`.
- **FR-003**: The system MUST prove FR-002 per feature by recomputing it with all later bars
  removed and asserting the value is unchanged — not by inspection or by argument.
- **FR-004**: The leakage audit MUST include a planted-leak control that it is required to
  detect, and the audit fails if the control passes.
- **FR-005**: The system MUST reuse the repo's existing entry-time feature computations rather
  than reimplementing them, so the audited features are the ones the strategy actually sees.
- **FR-006**: Rows that cannot be built MUST be excluded and counted, never imputed.

**Model and evaluation**

- **FR-007**: The secondary model for fold *k* MUST train only on trades from folds `< k`.
- **FR-008**: Evaluation MUST be confined to folds 5–14; folds 0–4 are warm-up.
- **FR-009**: Discrimination MUST be reported against the measured base rate (73.2%), and
  accuracy MUST NOT be reported as a success metric.
- **FR-010**: Every configuration MUST be trained at five or more seeds and reported as an
  interquartile mean with a bootstrap interval, never a best seed.
- **FR-011**: Every configuration MUST be appended to the trial log **before** its result is
  known.
- **FR-012**: Every swept parameter MUST be proven live before compute is spent on the sweep.
- **FR-013**: The floor MUST be recomputed for the folds 5–14 span rather than inheriting the
  full-archive figure, and the recomputed value reported wherever a delta is quoted.
- **FR-014**: Every reported return MUST carry retained-trade count, trades per year and mean
  holding time.
- **FR-015**: A configuration retaining too few trades to resolve MUST be reported as
  inconclusive, and its return withheld.

**Attribution**

- **FR-016**: The report MUST state the veto's gain both across all pairs and with ZEC
  excluded.
- **FR-017**: The report MUST state whether discrimination survives *within* pairs.

**Integration**

- **FR-018**: The strategy subclass MUST load the primary model from a pinned path and MUST NOT
  retrain it.
- **FR-019**: The subclass MUST be written only if a configuration is accepted.
- **FR-020**: Any divergence between the subclass's trades and the offline replay MUST be
  reported.

**Provenance and process**

- **FR-021**: All runs MUST execute sequentially.
- **FR-022**: Only files this feature creates or modifies MAY be staged.
- **FR-023**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.
- **FR-024**: A delta below the recomputed floor MUST be reported as "not resolved", never as
  "no effect".
- **FR-025**: The result report MUST be written even on a kill; the directional finding is the
  deliverable.

### Key Entities

- **Archived trade**: one realised primary trade — fold, pair, open and close dates, realised
  profit, exit reason.
- **Entry-time feature vector**: quantities knowable at `open_date` only.
- **Trade row**: an archived trade joined to its feature vector; the unit of training.
- **Nested fold assignment**: which trades may train the model evaluated on a given fold.
- **Veto decision**: accept or skip, at a threshold.
- **Trial record**: a configuration written to the log before its result exists.

## Success Criteria *(mandatory)*

### Protocol §5 acceptance gates

- **SC-001**: Annualised return on the evaluation span is at least the control plus the
  **recomputed floor for that span**, with a bootstrap interval excluding zero.
- **SC-002**: The configuration is positive in at least 7 of the 10 evaluation folds, and in
  each of W1/W2/W3 when run single-window.
- **SC-003**: The configuration beats all four mandatory baselines net of full costs.
- **SC-004**: Backtest-overfitting probability is at most 0.2 and the deflated risk-adjusted
  return is above zero, computed over the complete trial log.
- **SC-005**: The return delta remains positive at 1.5× fees, and the fee level at which the
  edge crosses zero is reported.
- **SC-006**: A separate process reproduces the accepted result from its saved configuration
  and seed.

### Protocol §5 kill gates

- **SC-007**: If no configuration clears SC-001 within the budgeted work, the verdict is
  "not resolved" and the report records it as such, never as "no effect".
- **SC-008**: If any configuration's in-sample advantage exceeds its held-out advantage by more
  than the recomputed floor, it is recorded as overfitting and stopped.

### Brief-specific gates

- **SC-009**: If nested out-of-sample discrimination is below 0.55 for every threshold, the
  feature is killed and the report records that entry-time features carry no information about
  outcome beyond what the primary model already used.
- **SC-010**: If the veto's gain disappears when ZEC is excluded, the result is recorded as
  **pair selection, not meta-labelling**, and the feature stops.
- **SC-011**: The leakage audit's planted-leak control is detected; if it is not, no downstream
  number is admissible.
- **SC-012**: Every reported return is accompanied by retained-trade count, trades per year and
  mean holding time, so a veto cannot rank well by discarding almost everything.
- **SC-013**: Every configuration evaluated appears in the trial log, each written before its
  result.
- **SC-014**: Every swept parameter is proven live before its sweep runs.
- **SC-015**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after completion.
- **SC-016**: The strategy subclass exists only if a configuration was accepted, and its trades
  match the offline replay or the divergence is reported.

## Assumptions

- The pooled walk-forward archive (`regime/wf/pooled_nnnc.json`, 378 trades / 15 folds) is the
  training set, and it is out-of-sample for the primary model by construction.
- Folds 0–4 (91 trades) are warm-up; the evaluation span is folds 5–14 (287 trades, 2.45 yr).
- The base win rate on the evaluation span is 73.2%, so discrimination — not accuracy — is the
  screening metric.
- The secondary model is a small gradient-boosting model rather than a neural network, because
  the evaluation set is ~287 rows.
- The veto action is tested before the sizing action, as the brief directs, because it has
  fewer degrees of freedom.
- The floor for the evaluation span is **not** 3.35 pp/yr — that figure was measured on the
  full 378-trade, 3.68-year archive. It must be recomputed for 287 trades over 2.45 years and
  is expected to be materially larger (roughly 5–6 pp/yr); the measured value governs.
- The primary model, its guards, its exits and its stop are untouched by this feature.

## Non-Goals

The approaches recorded in `briefs/09-excluded-and-why.md` are explicitly out of scope, in
particular: pseudo-labelling and label propagation; more hand-crafted labels; FreqAI
`continual_learning` and the FreqAI RL module; offline RL; LLM trading agents; time-series
foundation-model fine-tuning; learned market simulators; and full decision-focused
optimisation layers.

Also out of scope: changing the primary model, its guards, its exit logic or its stop; the
sizing action beyond a single screen after the veto is resolved; and any per-pair override,
which this repo's governance prohibits.
