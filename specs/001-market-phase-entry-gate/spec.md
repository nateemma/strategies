# Feature Specification: Market-Phase Framework and Phase-Adaptive Entry Gate

**Feature Branch**: `001-market-phase-entry-gate` (spec directory; no git branch created — this project commits directly to master)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "Market-phase identification framework and phase-adaptive entry gate for the NN reversion strategy family (NNNC/NNMT). Single leg only — no second momentum leg. Phases A (build classifier), B (characterize + actionability kill criterion), C (trackability go/no-go measurement), D (four gate mechanisms M3/M1/M2/M4). M5 and M6 explicitly out of scope."

## Context

The entry guard chain rejects roughly 93% of entry candidates, and the model contributes only ~7% of entries and ~8% of exits. The exit side has been measured as calibrated, with the trailing-stop and static-stoploss parameters silently inert under the active custom-stoploss path; the sole live exit lever is already at its optimum. Entry architecture is therefore the remaining lever.

Prior work built a BTC-only, five-state market taxonomy (trend-strength × trend-direction) and an opt-in entry gate based on it. That gate was rejected: it over-filtered the buy-low models (one strategy fell from 284 trades to 10) and acted as proportional de-risking rather than selective trade removal on momentum hosts. The diagnosis was that the taxonomy was scored on *passive* long exposure while the strategies are *active buy-low* models whose entries cluster in market weakness — the two interact oppositely.

Two facts from that work motivate this feature:

1. The only regime intervention that has ever improved this family is **per-asset**, not market-wide (a simple own-price trend condition improved risk-adjusted return substantially while cutting drawdown by more than three quarters).
2. In the prior taxonomy, **return discrimination was thin** (win rate spanning only 50.5–53.1% across five states) while **risk discrimination was roughly 4×** (worst-state drawdown 118.3% vs best-state 28.1%). That risk figure is not normalized for time-in-state and may be a sample-length artifact. If it survives normalization, it says the usable application of a phase signal is *exposure modulation*, not binary entry selection — an axis the rejected gate never tested.

This feature extends the taxonomy into a multi-level, lookahead-validated phase framework, tests whether phase is *predictive* rather than merely descriptive, and — only if it is — applies it as a soft, phase-adaptive modulation of the existing reversion leg.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A reusable, actionable phase framework (Priority: P1)

As the researcher, I need a market-phase classifier that spans the asset, market-breadth, and BTC levels, that is provably free of lookahead, and that reports which of its instruments produce states persistent enough to act on. Instruments whose states flip faster than a trade lives are eliminated before any strategy work begins.

**Why this priority**: This is the reusable asset the feature leaves behind regardless of every later outcome, and it is a prerequisite for every subsequent story. It also settles, on its own, the open question of whether the prior taxonomy's apparent risk discrimination is real or an artifact of unequal time-in-state. Delivered alone it is a complete, valuable research deliverable.

**Independent Test**: Run the classifier over the full history and inspect the per-instrument report of state distribution, median dwell time, and transition matrix across three temporally distant eras. Value is delivered when the report identifies which instruments survive the actionability criterion and which are eliminated — no strategy backtest required.

**Acceptance Scenarios**:

1. **Given** the full 15-minute history for the production universe, **When** the classifier is run, **Then** it emits a phase label at every bar for each of the three levels, computed only from information available at or before that bar.
2. **Given** a computed phase series, **When** it is subjected to an end-to-end lookahead test that compares labels produced from truncated history against labels produced from full history, **Then** the labels for the overlapping region are identical.
3. **Given** a phase instrument, **When** its median dwell time is shorter than a typical trade duration, **Then** the instrument is reported as ELIMINATED with the measured dwell time as the reason.
4. **Given** a phase instrument, **When** its state distribution or transition matrix differs materially between the earliest and latest era, **Then** the instrument is reported as ELIMINATED for instability.
5. **Given** the prior taxonomy's per-state drawdown figures, **When** drawdown is recomputed normalized for time spent in each state, **Then** the report states whether the risk discrimination is confirmed or refuted, with the normalized figures.

---

### User Story 2 - The trackability verdict (Priority: P2)

As the researcher, I need a single, unambiguous answer to whether a phase label observed at a point in time predicts which of two entry configurations of the reversion leg — an aggressive one and a conservative one — will win the coming period, better than always choosing one fixed configuration. The answer must be judged on the strategy's own realized trades, across temporally distant windows. This verdict decides whether any gate gets built.

**Why this priority**: This is the project's go/no-go. It is a measurement, not a build, and it costs roughly a day. A negative verdict is a successful outcome: it stops the far more expensive mechanism work. The two-configuration framing keeps the test structurally identical to the switch the mechanisms actually implement, rather than testing a conditional description of past returns — which is the mistake that invalidated the earlier market-regime gate. Direct precedent for the method exists: an earlier adaptive-parameter idea was killed by exactly this test when trailing-best lost to a constant at every horizon.

**Independent Test**: Produce the per-window comparison table — phase-chosen configuration versus each fixed configuration — for every surviving instrument across at least two temporally distant windows, and read the PASS/FAIL verdict directly from it. No gate implementation is required to obtain the verdict.

**Acceptance Scenarios**:

1. **Given** the surviving phase instruments, **When** forward performance is measured, **Then** it is computed from the strategy's own realized trades and never from passive market or basket returns.
2. **Given** a candidate phase instrument, **When** it is compared against a fixed-configuration assignment, **Then** the comparison reports the phase-chosen forecast error and the constant's forecast error side by side over the same windows.
3. **Given** the two entry configurations, **When** a window's winner is determined, **Then** the winner is decided on risk-adjusted return, so that proportional exposure changes do not register as wins.
4. **Given** the comparison results, **When** no instrument beats the constant in at least two temporally distant windows, **Then** the verdict is FAIL and User Stories 3 and 4 do not proceed.
5. **Given** a PASS verdict, **When** it is recorded, **Then** it names the specific instrument, phase level, and forward horizon that passed, so the gate work has an unambiguous input.

---

### User Story 3 - Phase-adaptive gate, no-retrain tier (Priority: P3)

As the researcher, I want to modulate the existing reversion leg's entry aggressiveness by phase — pressing harder in favorable phases and throttling in unfavorable ones — using only levers that require no model retraining, and to know which of them actually moves outcomes.

**Why this priority**: This is the payoff, but it is conditional on Story 2 passing. Within it, the mechanisms are ordered by leverage: the model-confidence bar is the dominant filter and is tested first; the per-asset trend condition second, as the cheapest lever with the strongest prior; the dip guards third as a deliberate control arm, since they are known to filter only about 12% of bars and are expected to be near-inert.

**Independent Test**: Each mechanism is an independent A/B arm against the unmodulated baseline over the same pinned windows. Any one of them can be run, measured, and accepted or rejected on its own.

**Acceptance Scenarios**:

1. **Given** a phase-adaptive mechanism, **When** it is enabled, **Then** the measured entry-candidate pass rate differs from the baseline pass rate; if it does not, the arm is reported as INERT rather than as a null result.
2. **Given** a mechanism arm, **When** it is evaluated, **Then** it is judged on realized profit and risk-adjusted return, never on any model-fit or classification metric.
3. **Given** a mechanism arm, **When** its measured difference from baseline is smaller than one percentage point, **Then** it is not accepted until it has been confirmed under paired repeated runs that isolate run-to-run variance.
4. **Given** a mechanism arm, **When** it is evaluated, **Then** the evaluation runs with realistic execution constraints active, so that fills exceeding available liquidity cannot inflate the result.
5. **Given** a mechanism arm, **When** it improves the recent window but reverses sign in the temporally distant window, **Then** it is rejected as era-specific rather than accepted.
6. **Given** any mechanism, **When** it is specified, **Then** it modifies entry behavior only; exit behavior is unchanged.

---

### User Story 4 - Phase as a learned input (Priority: P4)

As the researcher, I want the option to supply phase state to the model as an input feature, so the model can learn when phase matters instead of having a hand-set rule imposed on it — but only if the cheaper tier has already shown that phase carries signal.

**Why this priority**: It is the only path in scope that requires retraining and a feature-scaler regeneration, making it the most expensive. Prior measurement of market-state features as model inputs showed only a modest predictive lift, so it is not worth its cost unless Story 3 has already demonstrated signal.

**Independent Test**: Train with phase features included and compare realized trading outcomes against the identical configuration without them, over the same pinned windows.

**Acceptance Scenarios**:

1. **Given** phase features are added to the model's input set, **When** training is run, **Then** the feature scalers are regenerated first, so that the comparison is not made against stale normalization.
2. **Given** a trained phase-feature model, **When** it is compared to the baseline, **Then** the comparison is decided on realized trading outcomes and not on any improvement in model-fit metrics.

---

### Edge Cases

- **A phase level is unavailable for some bars.** Market-breadth measures need a minimum number of pairs with history at that time; early history has far fewer listed pairs than late history. The framework must define a minimum-coverage rule and mark bars below it as UNKNOWN rather than emitting a misleading state.
- **A phase state is rare.** If a state occupies a very small share of bars in a window, per-state statistics are unreliable. Any state below a stated minimum share must be reported with its sample count and excluded from verdict-driving comparisons.
- **BTC data is missing or stale.** The framework must degrade to a defined neutral state rather than silently blocking or permitting all entries.
- **Phase flips mid-trade.** Because exits are out of scope, a phase change while a position is open must not alter that position's handling; phase is consulted at entry evaluation only.
- **A mechanism produces zero trades in a window.** A zero-trade arm is not a result and must be reported as such, distinct from a zero-return arm.
- **A mechanism silently does nothing.** A phase-conditional threshold that never changes an entry-candidate pass rate must be surfaced as an inert-parameter defect, not recorded as evidence that phase does not help.
- **The trackability verdict is marginal.** If one instrument passes in one window and fails in another, the verdict is FAIL, not "mixed" — two windows that share an era cannot establish persistence.

## Requirements *(mandatory)*

### Functional Requirements

**Phase framework (Story 1)**

- **FR-001**: The framework MUST emit a market-phase label for every bar at three levels: the traded asset, the market as a whole, and BTC.
- **FR-002**: The framework MUST compute all phase labels causally, using only information available at or before the labeled bar.
- **FR-003**: The framework MUST include direct trend-versus-reversion measures (variance ratio at multiple aggregation lengths, a Hurst-style long-memory estimate, and short-lag return autocorrelation) in addition to the existing trend-strength, volatility, band-width, wavelet-trend, and channel-position indicators already available to the strategies.
- **FR-004**: The framework MUST reproduce the prior BTC taxonomy unchanged as a named baseline instrument, so that every new instrument is measured against a known reference rather than in isolation.
- **FR-005**: The framework MUST include market-breadth measures spanning cross-sectional return dispersion, the share of the universe in its own uptrend, and median pairwise correlation.
- **FR-006**: The framework MUST be re-runnable from a clean checkout without manual path editing, and MUST persist its outputs as durable artifacts rather than console-only output.
- **FR-007**: Every phase instrument MUST pass an end-to-end lookahead validation that compares labels computed from truncated history against labels computed from full history; instrument-by-instrument reasoning about causality is not sufficient evidence.

**Actionability characterization (Story 1)**

- **FR-008**: The framework MUST report, per instrument, the state distribution, the median dwell time per state, and the state transition matrix.
- **FR-009**: The framework MUST report each of those three characteristics separately for at least three temporally distant eras.
- **FR-010**: The framework MUST eliminate any instrument whose median dwell time is shorter than a typical trade duration, and MUST record the measured dwell time as the stated reason.
- **FR-011**: The framework MUST eliminate any instrument whose state distribution or transition matrix is materially unstable across eras, and MUST record the observed instability as the stated reason.
- **FR-012**: The framework MUST recompute the prior taxonomy's per-state drawdown normalized for time spent in each state, and MUST state explicitly whether the previously observed risk discrimination is confirmed or refuted.

**Trackability verdict (Story 2)**

- **FR-013**: The verdict MUST be derived from the reversion leg's own realized trades, and MUST NOT be derived from passive market, basket, or buy-and-hold returns.
- **FR-014**: The verdict MUST compare each surviving instrument's forward-edge forecast against a constant-assignment baseline over the same windows, reporting both forecast errors.
- **FR-015**: The verdict MUST be evaluated across at least two temporally distant windows, and MUST NOT treat two adjacent windows as independent evidence of persistence.
- **FR-016**: The verdict MUST be recorded as a single PASS or FAIL. A PASS MUST name the instrument, the phase level, and the forward horizon that passed.
- **FR-017**: A FAIL verdict MUST halt the feature; no gate mechanism may be built on an instrument that did not pass.
- **FR-018**: The quantity forecast MUST be the forward *relative* performance of two configurations of the same reversion leg — one more aggressive at entry, one more conservative — and the forecast MUST be scored against always choosing a single fixed configuration. Which configuration "wins" a window MUST be decided on risk-adjusted return, so that a configuration which raises return and drawdown in equal proportion does not count as a winner.
- **FR-033**: The trackability measurement and every mechanism arm MUST share a single host strategy, so that results are comparable across phases and no finding is confounded by a change of host mid-feature.

**Gate mechanisms (Stories 3 and 4)**

- **FR-019**: Mechanism M3 MUST make the model's entry-confidence threshold a function of phase, so that the threshold relaxes in favorable phases and tightens in unfavorable ones, without retraining.
- **FR-020**: Mechanism M1 MUST make the per-asset trend entry condition a function of phase — whether it is required at all, and over what trend length — rather than a fixed on/off setting, without retraining.
- **FR-021**: Mechanism M2 MUST make the dip-forcing entry guard thresholds a function of phase, without retraining, and MUST be reported as a control arm whose expected effect is small.
- **FR-022**: Mechanism M4 MUST supply phase state to the model as input features, and MUST regenerate feature normalization before any comparison is run.
- **FR-023**: Mechanisms MUST be independently enableable, so each can be measured as its own arm against a common baseline.
- **FR-024**: No mechanism may alter exit behavior.
- **FR-025**: No mechanism may introduce per-pair special-casing; all mechanisms MUST be structural and apply uniformly across the universe.

**Common evaluation protocol (Stories 2, 3, 4)**

- **FR-026**: Every comparison MUST run over a pinned, explicitly stated time range, so that arms measured on different days remain comparable.
- **FR-027**: Every comparison MUST span at least three windows, at least one of which is temporally distant from the others.
- **FR-027a**: The headline verdict for every comparison MUST be the result in the regime the strategy is deployed into. Results aggregated across the full history MAY be reported as context but MUST NOT be quoted as the headline outcome.
- **FR-028**: Every comparison MUST run with realistic execution constraints active, so that trades whose size exceeds available liquidity cannot inflate results.
- **FR-029**: Every comparison MUST be decided on realized profit and risk-adjusted return. Model-fit and classification metrics MUST NOT be used to accept or reject an arm.
- **FR-030**: Any accepted difference smaller than one percentage point MUST first be confirmed under paired repeated runs that isolate run-to-run variance.
- **FR-031**: Every mechanism arm MUST report its entry-candidate pass rate alongside its trading results, and an arm whose pass rate is unchanged from baseline MUST be reported as INERT rather than as a null result.
- **FR-032**: Every arm MUST report its trade count, and a zero-trade arm MUST be reported as inconclusive rather than as a zero-return result.

### Key Entities

- **Phase instrument**: A named, causal recipe that maps price and volume history to a discrete state at each bar. Carries a level (asset, market, or BTC), a state vocabulary, and a surviving/eliminated status with a recorded reason.
- **Phase label series**: The per-bar state assignment produced by one instrument for one asset, plus an UNKNOWN state for bars with insufficient coverage.
- **Actionability report**: Per instrument and per era — state distribution, median dwell time, transition matrix, and the elimination decision.
- **Trackability verdict**: A single PASS/FAIL for the feature, with the winning instrument, level, and horizon on a PASS, and the per-window forecast-error comparison against the constant baseline as evidence.
- **Mechanism arm**: One phase-adaptive entry configuration, evaluated against a common baseline over the pinned windows. Carries trade count, entry-candidate pass rate, realized profit, risk-adjusted return, and per-window results.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every phase instrument in the framework passes end-to-end lookahead validation with zero label differences in the overlapping region.
- **SC-002**: The framework reports state distribution, median dwell time, and transition matrix for every instrument across at least three temporally distant eras, with no instrument left uncharacterized.
- **SC-003**: Every instrument carries an explicit SURVIVING or ELIMINATED status with a measured, quoted reason; no instrument is carried forward without one.
- **SC-004**: The question of whether the prior taxonomy's roughly 4× risk discrimination is real is answered definitively, with time-normalized figures given.
- **SC-005**: The trackability question is answered with a single PASS or FAIL supported by a per-window comparison against a constant baseline over at least two temporally distant windows.
- **SC-006**: On a FAIL verdict, the feature stops with the negative result documented, and no mechanism is built — this counts as a successful outcome of the feature.
- **SC-007**: On a PASS verdict, at least three no-retrain mechanism arms are measured against a common baseline over the pinned windows.
- **SC-008**: Every mechanism arm reports an entry-candidate pass rate, and no arm is recorded as a null result while its pass rate is identical to baseline.
- **SC-009**: An accepted mechanism improves risk-adjusted return against baseline without reversing sign in the temporally distant window, and any sub-one-percentage-point improvement is confirmed under paired repeated runs.
- **SC-010**: Baseline exit behavior is unchanged across every arm, verifiable by identical exit configuration between each arm and its baseline.
- **SC-011**: The complete framework, characterization, and verdict are reproducible from a clean checkout by re-running the study, with no manual path or parameter edits.

## Assumptions

- **Evaluation windows**: Three windows are used, drawn from the available 15-minute history (BTC spans 2021-01 to 2026-08; 44 pairs reach back to 2023 and 24 to 2021). At least one window is separated from the others by a full market cycle, since adjacent windows share a regime and cannot establish persistence.
- **Typical trade duration**: The dwell-time kill criterion is calibrated against observed live trade durations of roughly 1.4 to 31 hours. An instrument whose median dwell is materially shorter than the low end of that range cannot be acted on.
- **Acceptance bar for a mechanism**: Improvement in risk-adjusted return with a consistent sign across all three windows, surviving paired repeated runs when the margin is under one percentage point. Absolute profit alone is not sufficient, since a drawdown-reducing change that cuts return proportionally is exposure reduction rather than selection.
- **Host strategy**: The non-GAN classifier base hosts the trackability measurement and every mechanism arm. It is the project's canonical base for comparative runs, it avoids entangling phase effects with data-augmentation effects, and it is not a weak host — the non-GAN configuration has measured higher than the augmented one. Porting an accepted mechanism to the production configuration is a follow-on step outside this feature.
- **Aggressive and conservative configurations**: The two configurations compared in the trackability test differ only in entry aggressiveness, on the lever that dominates entry filtering. All other settings — exits included — are identical between them, so the measured difference is attributable to entry aggressiveness alone.
- **Instrument reuse**: The prior BTC taxonomy and its existing gate scaffolding are reused as the baseline rather than rebuilt, so that new instruments are measured against a known reference.
- **Universe and timeframe**: The production 15-minute universe is used throughout. The prior study's 1-hour, twelve-asset dataset is treated as historical context, not as the evaluation set.
- **Data sources**: Only existing price and volume history is used. No new external data source is introduced by this feature.
- **Momentum leg**: No momentum leg is built. The prior breakout-label path was run to exhaustion and found untradeable at every timeframe and exit tested, and cross-family allocation is out of scope. This feature modulates the aggressiveness of the existing reversion leg only.
- **Exit side**: Exit behavior is fixed. The trailing-stop and static-stoploss parameters have been measured as inert under the active custom-stoploss path, and the one live exit lever is already at its optimum.
- **Governance**: The project constitution is currently an unfilled template, so no ratified project principles constrain this specification. If principles are later ratified, this spec should be re-checked against them.

## Out of Scope

- A second, momentum-oriented entry leg, in any form.
- Phase-conditional label generation or a dual-model switch (previously enumerated as M5).
- Cross-family allocation between the reversion leg and the momentum basket (previously enumerated as M6).
- Any change to exit rules, stop handling, or position sizing.
- Any new non-price data source.
- Per-pair overrides or pair-specific tuning.

## Resolved Questions

- **Question 1 — what the phase label must predict**: RESOLVED 2026-09-02. The forward relative performance of two configurations of the same reversion leg (aggressive vs conservative at entry), scored against always choosing one fixed configuration, with the winner decided on risk-adjusted return. Chosen over "forward edge" and "forward risk" because it keeps the trackability test structurally identical to the switch the mechanisms implement, and because scoring on risk-adjusted return subsumes the forward-risk formulation. See FR-018.
- **Question 2 — host strategy**: RESOLVED 2026-09-02. The non-GAN classifier base hosts every measurement. See the Host strategy assumption and FR-033.
