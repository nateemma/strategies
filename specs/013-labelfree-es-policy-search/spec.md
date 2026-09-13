# Feature Specification: labelfree-es-policy-search

**Feature Branch**: `013-labelfree-es-policy-search`

**Created**: 2026-09-13

**Status**: Draft — clarified

**Input**: User description: "Feature: labelfree-es-policy-search. This is research/label-free-study brief 05 … implement research/labelfree/es/ with CMA-ES over the exit/guard parameter space, a random-search control with the identical budget, and CSCV/PBO over the full evaluation history. Every evaluation must be appended to research/labelfree/trials.jsonl. Evaluate the test fold exactly once per fold. Write briefs/results/05-<date>.md per protocol §6 with PBO as the headline number."

## Context

The hypothesis is specific: the levers gradient descent **cannot reach** — ATR stop multiplier,
`stoploss_grace_*`, guard thresholds, minimum holding time — *jointly* matter, but have only
ever been swept **one at a time**, on windows that cannot resolve them. Searching them jointly
against an honest objective may find an operating point that one-at-a-time sweeps cannot.

The brief calls its own design "defensive" because the prior evidence is against it: Allen &
Karjalainen 1999 (GA rules, no excess return after costs); NEAT 2025 (below buy-and-hold); *Red
Queen's Trap* 2025 (15-minute crypto neuroevolution, +300% validation → −70% live); MadEvolve
2026 (the gap widens with search-space size). The consistent failure mode is **multiple
testing**, which is why trial accounting is the brief rather than a footnote.

### The measurement that governs this feature

`AGENT_GUIDE.md` records both numbers, and together they are the reason for the brief's own
kill criterion:

```
noise floor (W1, 99 trades, bootstrapped)  : 7.633 pp
span of an entire rvol parameter sweep     : 4.93 pp
```

**A whole parameter sweep fits inside the noise floor.** The search space's total range of
effect is smaller than the uncertainty on measuring it. Any optimum found within 4.93 pp of the
incumbent is indistinguishable from the sweep's own noise, which the brief makes a kill.

### What the study has already established

| brief | verdict |
|---|---|
| 01 direct policy optimisation | NOT RESOLVED — no configuration reached a pooled run |
| 02 label warm-start | NOT RESOLVED — 92% of the advantage was turnover, not signal |
| 03 meta-labelling | KILL — nested AUC 0.4978; the confidence→outcome map is non-stationary |
| 04 RL in a pooled env | KILL — pooled reward IQM −0.1301, 5/15 folds positive |

Briefs 01, 02 and 04 failed the **same objective** through three different searches. This is a
fourth search over that objective — but over a **different space**: the discrete exit and guard
levers none of the three could touch. That distinction is the feature's justification and is
recorded so the gates are set with it in view.

## Clarifications

### Session 2026-09-13

- Q: Brief 1's simulator cannot represent this search space — `sim.replay()` takes the trade
  series as *input*, `net_returns` has no stop/grace/min-hold logic, and a backtest per
  evaluation would cost ~1,167 hours. How is a candidate evaluated? → A: **Build a fast exit
  surrogate** inside the simulator (stop, grace, minimum hold and guards applied to the exposure
  path) and **calibrate it against real freqtrade backtests**, exactly as feature 009 calibrated
  its simulator to the arbiter. The calibration is a **blocking gate**: an uncalibrated surrogate
  means the search optimises a fiction, and the search's whole budget would be spent doing so.
- Q: The trial log holds 1,158 lines for the 505 trials of four features and is committed to
  git; this brief implies up to 48,000 evaluations. Where does that history live? → A: **A
  per-feature sidecar log.** The shared `trials.jsonl` receives **one summary row per search
  run**; the full evaluation history goes to the sidecar, which the overfitting statistic reads.
  The trial-log audit must be taught about both files, or it would certify a record it never saw.
- Q: The brief says "Expect PBO to be high — that is the measurement", while protocol §5 makes
  PBO > 0.2 a failure. Kill or deliverable? → A: **Both, and they do not conflict.** PBO > 0.2
  **kills the candidate** — no operating point is adopted — while the **measured PBO is the
  reported finding**. The feature is therefore expected to deliver a *quantified negative*,
  which is how briefs 01–04 delivered their value. §5 is preserved, not relaxed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - An evaluation that means something (Priority: P1) 🎯 MVP

A researcher needs a way to score a candidate parameter set that is both **fast enough to run
thousands of times** and **faithful to what the strategy would actually do**.

**Why this priority**: every later story calls it thousands of times. If the evaluation does not
reflect real exit behaviour, the search optimises a fiction and no amount of trial accounting
redeems it. This is also the feature's dominant risk — see the outstanding clarification.

**Independent Test**: score a set of candidate parameter sets and confirm the ranking matches
what a full, honest backtest of the same candidates produces.

**Acceptance Scenarios**:

1. **Given** a candidate parameter set, **When** it is evaluated, **Then** the score reflects the
   exit and guard behaviour those parameters actually produce, not a fixed trade series that
   ignores them.
2. **Given** a sample of candidates, **When** their fast scores are compared with full-backtest
   scores of the same candidates, **Then** the two rank them consistently, and the agreement is
   reported as a number rather than asserted.
3. **Given** the evaluation, **When** it is run on a fold, **Then** it uses that fold's
   **training** range only.

---

### User Story 2 - Search, and a control that spends the same budget (Priority: P2)

A researcher runs an evolutionary search over the parameter space, and — with an **identical
evaluation budget** — a uniform random search, so the optimiser's contribution is separable from
the budget's.

**Why this priority**: without the control, a good result cannot be distinguished from "we
looked 1,600 times". The brief makes the control a kill gate, not a nicety.

**Independent Test**: run both at the same budget and compare their out-of-sample results.

**Acceptance Scenarios**:

1. **Given** a budget, **When** both methods run, **Then** they consume the **same number of
   evaluations**, and that count is recorded for each.
2. **Given** both results, **When** they are compared out-of-sample, **Then** a search that does
   not beat random selection is reported as the optimiser fitting noise.
3. **Given** any candidate evaluated by either method, **When** it is scored, **Then** it is
   recorded before its score is known.

---

### User Story 3 - The test fold is touched exactly once (Priority: P3)

A researcher evaluates the chosen candidate on a fold's test range **once**, at the end, and can
prove from the record that it was not touched during the search.

**Why this priority**: with thousands of evaluations per fold, a single leak of the test range
into the search invalidates everything. This is the property the whole design exists to protect.

**Independent Test**: audit the evaluation record and confirm exactly one test-range evaluation
per fold, occurring after the search concluded.

**Acceptance Scenarios**:

1. **Given** a completed fold, **When** the record is audited, **Then** exactly one test-range
   evaluation exists for it.
2. **Given** the search history, **When** it is audited, **Then** no evaluation during the search
   used the test range.
3. **Given** a re-run, **When** it repeats, **Then** the single test evaluation is reproducible
   from the recorded candidate.

---

### User Story 4 - Overfitting is the headline, not a footnote (Priority: P4)

A researcher reports the probability of backtest overfitting over the **entire** evaluation
history — every candidate, every generation, both methods — as the primary number.

**Why this priority**: the brief's own framing is that a search of this shape is expected to
overfit, and that measuring how much **is the deliverable**. A point estimate without it is
worthless here.

**Independent Test**: the reported overfitting probability is computed over the full history and
the evaluation count is stated beside it.

**Acceptance Scenarios**:

1. **Given** the complete history, **When** overfitting is computed, **Then** every evaluation
   contributes — including abandoned and discarded candidates.
2. **Given** the figure, **When** it is reported, **Then** the number of evaluations entering it
   appears beside it, since the statistic's reliability depends on that count.
3. **Given** a deflated performance figure, **When** it is reported, **Then** the deflation uses
   the **total evaluation count**, not the count of candidates that survived.

---

### User Story 5 - Did it actually move? (Priority: P5)

A researcher determines whether the found optimum differs from the incumbent operating point by
more than the sweep's own noise.

**Why this priority**: this is the brief's second kill gate, and the measured numbers say it is
the likely outcome — a whole sweep spans 4.93 pp inside a 7.633 pp noise floor.

**Independent Test**: compare the found optimum with the incumbent and report the distance
against the 4.93 pp band.

**Acceptance Scenarios**:

1. **Given** the found optimum, **When** it is compared with the incumbent, **Then** a difference
   within the sweep-noise band is recorded as **not resolved**, never as an improvement.
2. **Given** the search, **When** it is regularised, **Then** moving away from the incumbent
   carries a cost, so the search must earn the move.

### Edge Cases

- **A candidate produces no trades.** It must be inadmissible rather than scored as zero — this
  study has repeatedly had abstention compared as though it were performance.
- **A candidate loses money but beats the incumbent.** A delta against a losing comparator is
  not evidence of value; the arm must beat doing nothing.
- **The search converges to the incumbent.** That is a legitimate and informative outcome, and
  must be reported as "no better operating point found", not as a failure to run.
- **Two candidates are numerically identical.** They are one evaluation for scoring but two for
  the trial count, because the deflation counts attempts.
- **A parameter hits its bound.** The optimum sitting on a boundary means the space was drawn
  too narrowly; this must be reported rather than silently accepted.
- **The fast evaluation and a full backtest disagree on a candidate's rank.** The disagreement
  must be measurable and reported; if it is large the search is optimising the wrong thing.

## Requirements *(mandatory)*

### Functional Requirements

**Evaluation**

- **FR-001**: A candidate's score MUST reflect the exit and guard behaviour its parameters
  produce, not a trade series fixed independently of them.
- **FR-002**: Evaluation during search MUST use the fold's training range only.
- **FR-003**: The fast evaluation's agreement with a full, honest backtest MUST be **measured on
  a sample of candidates and reported as a number**, not assumed.
- **FR-004**: A candidate producing no trades MUST be inadmissible, with its score withheld.
- **FR-005**: A candidate that does not beat doing nothing MUST NOT be reported as an
  improvement over any comparator.

**Search and control**

- **FR-006**: The search MUST cover the exit and guard parameters jointly, not one at a time.
- **FR-007**: A uniform random control MUST run at an **identical evaluation budget**, and both
  counts MUST be recorded.
- **FR-008**: If the search does not beat the random control out-of-sample, the result MUST be
  recorded as the optimiser fitting noise.
- **FR-009**: The search MUST penalise distance from the incumbent operating point and turnover,
  so a move away from the incumbent must be earned.

**Leakage control**

- **FR-010**: Each fold's test range MUST be evaluated **exactly once**, after its search ends.
- **FR-011**: The record MUST allow an auditor to verify FR-010 rather than take it on trust.
- **FR-012**: No selection decision MAY consume a quantity computed on the fold's test range.

**Trial accounting**

- **FR-013**: Every evaluation MUST be recorded before its score is known.
- **FR-014**: The overfitting probability MUST be computed over the **entire** evaluation
  history, including discarded and abandoned candidates.
- **FR-015**: The deflated performance figure MUST use the **total evaluation count**.
- **FR-016**: The evaluation count MUST be reported beside any overfitting figure.

**Reporting**

- **FR-017**: Every reported return MUST carry trade count, trades per year and mean holding
  time.
- **FR-018**: A difference from the incumbent within the sweep-noise band MUST be reported as
  "not resolved", never as an improvement.
- **FR-019**: A parameter optimum sitting on a bound MUST be reported as a space drawn too
  narrowly.
- **FR-020**: The result report MUST be written even on a kill.

**Process**

- **FR-021**: All runs MUST execute sequentially.
- **FR-022**: Only files this feature creates or modifies MAY be staged.
- **FR-023**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.

### Key Entities

- **Candidate**: one point in the exit/guard parameter space.
- **Evaluation**: a candidate scored on a named range; the unit of trial accounting.
- **Generation**: a population of candidates proposed together.
- **Search history**: every evaluation from both methods; the input to the overfitting statistic.
- **Incumbent**: the production operating point the search must beat by more than sweep noise.
- **Fidelity sample**: candidates scored both fast and honestly, to quantify their agreement.

## Success Criteria *(mandatory)*

### Protocol §5 acceptance gates

- **SC-001**: Pooled walk-forward annualised return is at least the control plus 3.35 pp/yr, with
  a bootstrap interval excluding zero.
- **SC-002**: Positive in at least 10 of 15 folds and in each of W1/W2/W3 single-window.
- **SC-003**: Beats all four mandatory baselines net of full costs. *(Measured 2026-09-13 over
  folds f05–f14: the production control is the only profitable baseline at **+7.86 pp/yr**, so
  this gate means clearing +7.86, not clearing zero.)*
- **SC-004**: **Overfitting probability at most 0.2** and the deflated risk-adjusted return above
  zero, over the complete evaluation history. **This is the primary gate for this feature.**
- **SC-005**: The return delta remains positive at 1.5× fees.
- **SC-006**: A separate process reproduces the accepted result from its saved configuration.

### Protocol §5 kill gates

- **SC-007**: If no configuration clears SC-001 within the budgeted work, the verdict is "not
  resolved", never "no effect".
- **SC-008**: If any configuration's in-sample advantage exceeds its held-out advantage by more
  than the floor, it is recorded as overfitting and stopped.

### Brief-specific gates

- **SC-009**: If the random-search control matches the evolutionary search on pooled
  out-of-sample results, the feature is **killed immediately** and the result recorded as the
  optimiser fitting noise.
- **SC-010**: If the found optimum sits within **4.93 pp** — one sweep-noise band — of the
  incumbent, the feature is **killed** and recorded as not resolved.
- **SC-011**: Each fold's test range is evaluated exactly once, verifiable from the record.
- **SC-012**: Every evaluation appears in the trial record, each written before its score.
- **SC-013**: The fast evaluation's rank agreement with full backtests is measured on a sample
  and reported; a disagreement large enough to reorder candidates invalidates the search.
- **SC-014**: Every reported return carries trade count, trades per year and mean holding time.
- **SC-015**: No candidate below the participation floor, and no candidate losing to doing
  nothing, is reported as an improvement.
- **SC-016**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-017**: The fast exit surrogate reproduces full-backtest behaviour closely enough to rank
  candidates identically. **This is a blocking gate**: until the surrogate's agreement with real
  backtests is measured and reported, no search result is admissible. An uncalibrated surrogate
  means the entire search budget optimises a fiction.
- **SC-017a**: The surrogate's rank agreement is measured on a sample of candidates scored **both
  ways**, and any candidate pair the two orderings disagree on is reported. Disagreement large
  enough to change which candidate wins invalidates the search.
- **SC-018**: The complete evaluation history is recorded in a per-feature sidecar log, and the
  shared trial log carries one summary row per search run. The overfitting statistic reads the
  sidecar; the audit verifies **both**, since an audit that only sees the summary would certify
  a record it never examined.
- **SC-019**: An overfitting probability above 0.2 **kills the candidate** — no operating point
  is adopted — and the **measured value is reported as the finding**, with the evaluation count
  beside it. A quantified negative is a successful outcome of this feature; §5 is preserved
  rather than relaxed.

## Assumptions

- Brief 1's vectorised simulator is the intended evaluator, and its calibration to the arbiter
  (0.00000000 bps across 99 W1 trades) is the standard the exit surrogate must meet.
- The surrogate adds stop, grace, minimum-hold and guard logic to the exposure path — behaviour
  the existing simulator does not model at all. It is new machinery, and its calibration is the
  feature's first blocking gate, not a later validation step.
- The expected outcome is a quantified negative: the literature the brief cites is uniformly
  against gradient-free search on this problem, and the measured sweep span (4.93 pp) sits
  inside the measured noise floor (7.633 pp).
- The incumbent operating point is `NNNC/NNNC_MLX.json`, which is **tracked**, so the
  regularisation anchor is reproducible.
- The sweep-noise band of 4.93 pp and the noise floor of 7.633 pp are measured values from
  `AGENT_GUIDE.md`, not chosen thresholds.
- The evolutionary method is CMA-ES for the parameter space; the `cma` package is **not currently
  installed** and is a new dependency.
- The random control draws uniformly from the same bounded space as the search.
- Search space (b) — a small policy head on the frozen encoder — runs **only** if space (a)
  clears the floor, per the brief.
- The shared admissibility gate applies: a candidate that abstains or loses to cash cannot be
  reported as an improvement.

## Non-Goals

The approaches in `briefs/09-excluded-and-why.md` are out of scope, in particular: NEAT, genetic
programming of rules, and full-network neuroevolution — this brief keeps only the narrow,
trial-accounted version. Also excluded: Decision-Transformer and diffusion planners; policies
trained inside a learned market simulator; LLM trading agents; offline RL; and full
decision-focused optimisation layers.

Also out of scope: evolving a full network; changing the primary model's weights; and any
per-pair parameter.
