# Feature Specification: Held-Pair Position Add (DCA)

**Feature Branch**: `031-held-pair-dca` (spec directory; work is committed to master)

**Created**: 2026-10-13

**Status**: Draft

**Input**: User description: "Held-pair position add (DCA) for the shared entry framework (utils/EntryFramework, used by NNNC and SimpleTrigger). Today freqtrade allows one trade per pair, so a fresh entry signal on a pair that already holds a trade is ignored. [...] Goal: implement DCA as a default-off framework switch, then evaluate it under a pre-registration. [...] Out of scope: changing the model, labels, thresholds or the grace value; promoting the switch to a default (the user's decision); stacking separate positions; partial exits." (full text in the conversation of 2026-10-13)

## Background *(PROVISIONAL: screen evidence only)*

**The blocking cost.** Freqtrade holds at most one trade per pair. A fresh entry signal on a pair
that already holds a trade is ignored. The 24 h grace studies split each arm's gain against its
control (research/st_grace/report.md):

| book | entries the 24 h arm forgoes | of which the pair was held | pp forgone |
|---|---|---|---|
| SimpleTrigger reversion (`st_prom_c2`) | 957 | 944 | +37.2 |
| SimpleTrigger momentum (`st_bgo_mx`) | 1,231 | 1,203 | (mixed sign) |
| NNNC arm D (round 1) | 263 | 261 | +13.6 |
| NNNC GBR (round 1) | 432 | 427 | +7.0 |

On NNNC, about 40% of the 24 h grace gross gain went to blocked re-entries. The 10 open slots
were never all full, so the binding limit is one trade per pair.

**The screen.** `research/held_pair/screen.py` (commit 07ff0dc) rebuilt each NNNC book from its
signal tables (fidelity 99.7–100%; its no-change policy equals the study simulator exactly) and
priced four policies for a fresh signal on a held pair. Results are at 24 h grace, minimum held age
1 h, early half f00–f07 only. **The late half f08–f14 has not been looked at.**

| policy | ARO | GBR | GFM | MLX |
|---|---|---|---|---|
| ROT: close the held trade, re-enter if the signal persists | −11.5 | −23.6 | −3.6 | −5.5 |
| ANCHOR: restart the clocks, no add | −2.7 | −11.7 | +1.3 | −2.0 |
| STACK: a second position (ceiling; not possible live) | +21.5 | +22.2 | +6.7 | +7.9 |
| **DCA: add one stake to the held trade** | **+26.2** | **+42.6** | **+7.0** | **+9.6** |

Against pure extra capital at the book's base return per stake, DCA gains more on ARO
(26.2 vs 14.4), GBR (42.6 vs 13.3) and MLX (9.6 vs 5.6). On GFM it gains about the same as the
extra capital alone (7.0 vs 7.6).

**What the screen modelled.** DCA here means:
- the add fills at the next bar's open after the signal, the same timing as a fresh entry;
- at most one add per trade;
- the add is sized like a fresh entry and bounded by the wallet, with no extra slot;
- the entry price becomes the amount-weighted average;
- the trade's clocks and its take-profit and stop reference restart at the add, from the averaged
  price.

Freqtrade supports this natively as position adjustment. The framework does not use it yet.

## Clarifications

### Session 2026-10-13

- Q: Which SimpleTrigger basis do the 2022 arms use? → A: The 24 h `_G24` classes
  (`SimpleTrigger_BreakoutGbb_MomExit_G24`, `SimpleTrigger_RevOnlyVolreg2_P2_G24`), against the
  existing 2022 controls `st_e22_g24_mom` / `st_e22_g24_rev`. This matches the NNNC setting.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A default-off held-pair add switch in the shared framework (Priority: P1)

A researcher declares one switch on any strategy built on the shared entry framework (NNNC or
SimpleTrigger). With it on, a fresh entry signal on a pair holding an open long trade adds one
stake to that trade instead of being ignored. With it off, nothing changes.

**Why this priority**: Nothing can be tested in a backtest until the behaviour exists. It must be
off by default, so that every recorded result stays reproducible.

**Independent Test**: With the switch off, the reference arms reproduce their recorded trades
15/15 and their fingerprints are unchanged. With it on, a small scripted case shows exactly one add
at the expected candle, price and size, and the clocks restarting from the add.

**Acceptance Scenarios**:

1. **Given** the switch off, **When** arm D (`wf_aro`), the reference NNNC arms and the SimpleTrigger
   reference arms run, **Then** every per-fold trade set is byte-identical to its record (15/15) and
   every strategy fingerprint and model key is unchanged.
2. **Given** the switch on and a held trade on pair X, **When** the strategy's entry signal is set
   for X on candle k, **Then** one add fills at candle k+1's open, sized by the fresh-entry sizing
   rule, and the trade's average entry price is the amount-weighted average of its fills.
3. **Given** a held trade that has already taken the maximum number of adds, **When** another
   signal fires, **Then** no add happens.
4. **Given** a held trade whose last fill is younger than the minimum age, **When** a signal fires,
   **Then** no add happens.
5. **Given** the "only when under water" condition is on and the held trade is in profit at the
   add candle's open, **When** a signal fires, **Then** no add happens.
6. **Given** an add would exceed the available wallet, or fall below the minimum stake after the
   liquidity cap, **When** a signal fires, **Then** the add is shrunk to what is available, or
   skipped if below the minimum.
7. **Given** an add took place, **When** the stop and the custom exit are evaluated later, **Then**
   every age-based rule (stop grace window, unclog, maximum hold, the minimum hold before a
   discretionary exit) measures age from the last fill. The stop and take-profit are measured from
   the averaged entry price. Any signal-bar quantity (the ATR take-profit) is taken from the add's
   signal bar.
8. **Given** any switch combination, **When** the stop is computed, **Then** the existing stop
   invariants hold, and a violation stops the run.

---

### User Story 2 - Prove the backtest does what the screen priced (Priority: P2)

Before any result is judged, the researcher checks that the backtested switch reproduces the
screen's DCA simulation on the same signals. The researcher also checks that the evaluation harness
handles a trade with two entry fills correctly.

**Why this priority**: Constitution IV (verify parameters are live) and the audit-the-instrument
rule. The screen and the backtest can differ in timing, sizing or clocks. A mismatch must be found
before it is read as a result.

**Independent Test**: A fidelity report comparing the backtested DCA trades against the screen's
DCA simulation, fold by fold, on the early half. A harness check on a trade with an add.

**Acceptance Scenarios**:

1. **Given** the switch on with the screen's selected setting, **When** one NNNC model's early-half
   folds are backtested, **Then** at least 95% of adds match the screen's adds (pair and candle).
   The per-fold profit difference is reported. A larger gap is investigated and explained before
   any judging run.
2. **Given** a trade with an add, **When** the harness records and pools it, **Then** the stake, the
   absolute profit, the per-fold return and the trade identity used for pairing are correct. The
   identity is the original open date, so a DCA trade pairs with its control trade.
3. **Given** the arm records, **When** liveness is reported, **Then** each fold shows the number of
   adds and the trades changed vs the control. An arm with zero adds is reported as inert, not as
   a result.

---

### User Story 3 - Pre-registered judgement on NNNC and SimpleTrigger (Priority: P3)

The researcher commits a pre-registration, then runs the DCA arms against their controls. The
existing rule turns each into ADOPT, NOT ADOPTED or INCONCLUSIVE.

The pre-registration fixes:
- the setting selected on the early half only: add on any fresh signal, minimum held age 1 h, one
  add, 24 h grace on NNNC;
- the arms and controls;
- the decision rule;
- the reporting.

**Why this priority**: The screen saw only the early half, so the late half f08–f14 is still a
clean test for NNNC. 2022 has never been used for this question, so it is a distant era for
SimpleTrigger (constitution II).

**Independent Test**: A committed pre-registration older than every arm record, and a verdicts
file with one verdict per book.

**Acceptance Scenarios**:

1. **Given** the pre-registration, **When** any DCA arm record is written, **Then** the record is
   newer than the pre-registration's commit, and the judge refuses to run otherwise.
2. **Given** an NNNC model (ARO, GBR, GFM, MLX), **When** it is judged, **Then** its verdict is
   ADOPT only if its early and late gains over its 24 h control are both positive and the paired
   and risk verdicts map to pass. This is the existing `judge_one` rule over 15 folds. The late
   half is labelled confirmatory and the early half exploratory.
3. **Given** a SimpleTrigger book (reversion and momentum), **When** it is judged on 2022 (e00–e02),
   **Then** its verdict is ADOPT only if its 2022 gain is positive and the paired and risk verdicts
   map to pass, as in the 24 h grace study's 2022 rule. The arms are built on the 24 h `_G24`
   classes and judged against their 24 h 2022 controls. Its 2023–26 result, against the 24 h
   `st_g24_mom` / `st_g24_rev` arms, is reported alongside as exploratory.
4. **Given** any arm, **When** it is reported, **Then** the report shows:
   - the maximum drawdown and the exposure (average and peak stake committed), arm vs control;
   - the number of adds;
   - a leverage control: the gain an equal extra stake would have earned at the control's base
     return per stake.

---

### User Story 4 - Report and hand-over (Priority: P4)

One report gives the switch, the fidelity check, each book's verdict with its evidence level, the
risk and leverage comparison, and a recommendation on whether the switch should be turned on for
NNNC. Promotion stays the user's decision.

**Why this priority**: The result decides the held-pair policy for the NNNC model work.

**Independent Test**: The report exists, with every book's verdict and the leverage comparison.

**Acceptance Scenarios**:

1. **Given** the feature ran, **When** the report is written, **Then** each book shows the screen
   prediction, the arm result and the leverage control side by side, labelled with evidence level.
2. **Given** a book whose gain does not beat its leverage control, **When** it is reported,
   **Then** the report says the gain is explained by extra capital, even if the verdict is ADOPT.

### Edge Cases

- **Signal on the candle a trade exits.** Freqtrade checks entries before exits on each candle.
  An add whose candle also triggers the trade's exit is reported in the fidelity check. The screen
  assumes the exit wins.
- **Wallet nearly full.** When open stake approaches the wallet, an add competes with fresh entries
  for the remaining balance. Adds and fresh entries are counted separately in liveness.
- **Fold end.** A trade with an add that is open at the fold's end is force-closed, as every trade
  is. Its full stake counts.
- **Stop already tightened.** If the stop has tightened past the grace window before the add, the
  add restarts the grace window. The stop may then widen again for the new window. This is the
  screen's behaviour. It must not break the stop invariants, and it is noted in the report.
- **Unfilled add order.** In a backtest the add fills at the candle's open. A live unfilled add
  order is out of scope (backtest evaluation only).
- **Short trades.** NNNC and SimpleTrigger are long-only. The switch applies only to long trades.
- **The late half is touched before pre-registration.** If any late-half DCA result exists before
  the pre-registration commit, it is listed, and the late half can no longer be called
  confirmatory.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The shared entry framework MUST provide a held-pair add switch, off by default,
  declarable per strategy. Its parameters MUST be: on/off, maximum adds per trade (1 in this
  feature), minimum age of the last fill in hours, and an "only when under water" condition.
  Parameter dicts MUST merge from the parent.
- **FR-002**: With the switch on, an add MUST happen only when the strategy's own entry signal is
  set for that pair on that candle, after every entry guard. An add MUST NOT happen on a pair with
  no open trade (that is a fresh entry).
- **FR-003**: An add MUST be sized by the same rule as a fresh entry (stake amount, liquidity cap,
  minimum stake), bounded by the available wallet, and MUST NOT use an extra open-trade slot.
- **FR-004**: After an add, every age-based rule in the custom stoploss and custom exit MUST measure
  age from the trade's last entry fill. With no add, the last fill is the open, so behaviour is
  unchanged. The rules are the stop grace window, unclog, maximum hold and the minimum hold before
  a discretionary exit.
- **FR-005**: After an add, the stop and take-profit MUST be measured from the averaged entry price.
  Signal-bar quantities (the 028 ATR take-profit) MUST be taken from the signal bar of the last fill.
- **FR-006**: The switch MUST be strategy-agnostic. No strategy- or pair-specific code may be added
  under Framework/. The exit-parameter guard test MUST keep passing.
- **FR-007**: With the switch off, the reference arms (arm D `wf_aro`, the four NNNC 24 h controls,
  `st_bgo_mx`, `st_prom_c2`) MUST reproduce their recorded per-fold trades 15/15. Every strategy
  fingerprint and every model key MUST be unchanged, and no model MUST be trained.
- **FR-008**: The evaluation harness MUST record, pool, pair and risk-score trades with more than
  one entry fill correctly. It MUST record per trade the number of adds and the total stake. A
  trade is identified by its original open date.
- **FR-009**: A fidelity check MUST compare the backtested DCA arm with the screen's DCA simulation
  on the same signals (one NNNC model, early half) before any judging run. At least 95% of adds
  MUST match on pair and candle. Any profit gap MUST be explained in the report.
- **FR-010**: A pre-registration MUST be committed before any DCA arm record of a judging window.
  It MUST fix the setting, arms, controls, folds, decision rule and verdict mappings, and the
  reporting. The setting is the early-half screen selection: add on any fresh signal, minimum age
  1 h, one add. NNNC arms MUST run at its 24 h default. Any later change MUST be recorded as a
  deviation.
- **FR-011**: NNNC verdicts MUST use the existing `judge_one` rule over f00–f14, per model (ARO,
  GBR, GFM, MLX), against that model's 24 h control.
- **FR-012**: SimpleTrigger verdicts MUST use 2022 (e00–e02) as the judging window with the 24 h
  grace study's 2022 rule, for reversion and momentum. The arms MUST be built on the 24 h `_G24`
  classes and judged against `st_e22_g24_rev` / `st_e22_g24_mom` (clarified 2026-10-13, Q1: A).
  The 2023–26 result, against `st_g24_rev` / `st_g24_mom`, MUST be reported as exploratory.
- **FR-013**: Every arm MUST report liveness (adds per fold, trades changed vs control), maximum
  drawdown, exposure (average and peak committed stake), and a leverage control. A zero-add arm is
  inert and inconclusive.
- **FR-014**: The report MUST label every figure with its evidence level. The switch MUST stay off
  by default whatever the verdicts. Turning it on is the user's decision.

### Key Entities

- **Held-pair add switch**: the declared on/off flag, maximum adds, minimum last-fill age, and
  under-water condition on a strategy.
- **Last fill**: the most recent entry fill of a trade. It anchors the trade's clocks and its
  signal-bar quantities.
- **DCA trade record**: a trade with its original open date, number of adds, total stake and
  averaged entry price.
- **Pre-registration**: the committed, dated record of setting, arms, controls, folds, rule and
  reporting.
- **Book verdict**: per NNNC model or SimpleTrigger family, ADOPT / NOT ADOPTED / INCONCLUSIVE with
  the gains, paired and risk verdicts, liveness, exposure and leverage control.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Switched off, 100% of reference folds (15/15 per reference arm) reproduce their
  recorded trades. All fingerprints and model keys are unchanged, and nothing is retrained.
- **SC-002**: Switched on, at least 95% of the backtest's adds match the screen's adds on the
  fidelity model's early half.
- **SC-003**: The pre-registration's commit is older than 100% of the judging-window arm records.
- **SC-004**: Each of the four NNNC models and each SimpleTrigger family in scope gets exactly one
  recorded verdict that cites the pre-declared rule.
- **SC-005**: Every arm's report shows its gain beside its leverage control, maximum drawdown and
  exposure.
- **SC-006**: No stop-invariant violation and no exit-parameter guard-test failure occurs.

## Assumptions

- Harness, fold length, costs, liquidity cap, stake (900), wallet (10,000 × 0.99) and 10 open
  slots are as in 023–030. Adds share the wallet with fresh entries.
- `minimal_roi` is `{"0": x}` for these books, so ROI depends only on profit from the averaged
  entry price, not on age.
- The NNNC grace stays at its 24 h default. The add restarts the grace window, as the screen did.
- Freqtrade's native position adjustment provides the add, the averaged entry price and the
  per-trade fill history. No change to the outer freqtrade repository is needed.
- NNNC has no model covering 2022 and no retraining is allowed. The distant-era evidence for the
  mechanism therefore comes from SimpleTrigger 2022.
- SimpleTrigger has no signal tables, so it has no screen of its own. Its arms use the NNNC-selected
  setting unchanged.
- Out of scope: model, label, threshold or grace changes; separate stacked positions; partial
  exits; live order handling for adds; making the switch a default.
