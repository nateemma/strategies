# Feature Specification: Realistic P&L for NNNC

**Feature Branch**: `036-realistic-nnnc-pnl` (spec directory; work is committed to master)

**Created**: 2026-10-16

**Status**: Draft

**Input**: User description: "Realistic P&L improvement for NNNC (feature 036)." Full text in
`research/live_pnl/spec_036_draft.md` (commit affb6069). In summary:
- **Goal:** measure and raise NNNC's realistic dollar P&L on Binance.US.
- **Stories:**
  - (P1) a live dry-run on a refreshed model, recording orders and order-book depth;
  - (P2) an execution model from that depth;
  - (P3) a size-aware stake cap;
  - (P4, user decision) wallet sharing with the momentum basket.

## Background *(evidence)*

Sources: `research/control_sizer/report.md`, `research/model_calibration/findings.md`, memory
`project_liquidity_and_capacity`.

**NNNC leaves almost all of its capital idle.** Over the 15 walk-forward folds (2023-01 →
2026-09) the NNNC control (`wf_aro_g24`):
- averages 0.37 open trades;
- has, on average, about 1.7% of the 10,000 wallet in open trades;
- has a median stake of about 300 USDT, not the configured 900.

The stake is reduced because every order is capped at 1/10 of its candle's quote volume, and
Binance.US books for these pairs are thin. The control's summed wallet return over the 15 folds is
about 50%.

**The per-trade edge is not the bottleneck any more:**
- training-set, label and input changes (032–035) found nothing that ranks trades better than the
  current model;
- the one positive model result, the capital-neutral sizer `cs_aro` (+8.5 pp over the width book
  on identical trades, late half not blind), was INCONSISTENT on a blind 35-day forward window
  (2026-09-07 → 2026-10-12): −0.41 pp against the width book, with a 90% interval spanning zero.

That forward window also showed **operating-point drift**. Weights and a threshold frozen in June
put only 20% of trades above the threshold in September–October, against about 50% in their own
validation.

**Backtest returns are not realistic P&L at larger sizes:**
- the backtest fills at the best displayed price, with no model of how price moves against a
  larger order;
- the 1/10 cap was chosen, never measured;
- loosening it in a backtest raises reported return while profit factor stays flat. That is a
  sizing artefact, not edge.

The recorded rule (memory `project_liquidity_and_capacity`): any larger size must be justified
from fill and depth evidence, not from backtests.

**The two binding limits are therefore:**
- how much can be traded per signal at an acceptable price;
- how current the model and its threshold are.

Neither can be measured from history alone.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run NNNC live, on a current model, and record what execution would cost (Priority: P1)

The user wants NNNC running against live Binance.US data, as a dry-run with no real money, beside
the existing momentum dry-run. It should run on a model trained on data up to the present, with
the threshold chosen by the same rule the walk-forward folds use.

For every order it would place, the bot records:
- the price and time of the signal;
- the stake it wanted, and the stake after the liquidity cap;
- whether and when the order filled, and at what price;
- a snapshot of the order book on both sides at that moment.

Each day, its trades are compared against a backtest of the same days on the same model. A
mismatch is then a defect to fix before any P&L read.

**Why this priority**: every later story needs the depth record this one produces. It is also the
only realistic forward evidence for NNNC. The model is retrained on a declared schedule, which
addresses the drift seen on the forward window.

**Independent Test**: start the bot and run it for at least one week. Check that:
- the order and depth records are complete for every order;
- the daily parity report runs, and either matches or names the cause of each difference;
- the momentum dry-run's process, database and configuration are unchanged.

**Acceptance Scenarios**:

1. **Given** data up to the latest download, **When** the live model is prepared, **Then**:
   - it is trained only on data before a declared cut-off;
   - its threshold is picked on that training span's validation tail by the standard rule;
   - its identity (model key, training range, threshold, width cut if any) is recorded before the
     bot starts.
2. **Given** the bot is running, **When** a signal produces an order, **Then** the order record and
   a depth snapshot of at least the top 20 levels per side are written within the same candle. A
   failed snapshot is recorded as missing, never silently skipped.
3. **Given** a day of live trading, **When** the parity report runs, **Then** it lists:
   - the live trades and the backtest trades for that day on the same model and settings;
   - matched, live-only and backtest-only entries and exits;
   - a cause for each difference.

   An unexplained difference is flagged.
4. **Given** the live model's weights are missing or do not match the recorded identity, **When**
   the bot starts, **Then** it refuses to run rather than training a model live.
5. **Given** the declared retrain date arrives, **When** the model is refreshed, **Then** the old
   and new identities are both recorded. Trades are attributed to the model that made them.
6. **Given** the momentum dry-run is running, **When** the NNNC bot starts, stops or fails,
   **Then** the momentum bot's process, database, port and configuration are untouched.

---

### User Story 2 - Price execution from recorded depth (Priority: P2)

From the depth snapshots, the user wants an estimate of what a given stake would really cost to
enter and exit on each pair: walking the recorded book, plus fees. That estimate is then applied
to backtests as a size-dependent cost.

With it, the 15-fold NNNC control can be re-stated at realistic execution. The current 1/10 cap
can then be judged by what it costs, not by what it reports.

**Why this priority**: it turns the depth record into a usable cost. Without it, any change to
stake size is unmeasurable (the constitution's Principle V). It also re-bases every existing NNNC
number on realistic execution.

**Independent Test**: build the execution model from at least the minimum recording period. Then
check:
- it reproduces the recorded snapshots' impact for held-out days within a declared tolerance;
- the re-stated 15-fold control reports, per fold, the gross return, the estimated cost and the
  net return at the current cap.

**Acceptance Scenarios**:

1. **Given** depth snapshots for a pair, **When** the model estimates the cost of a stake, **Then**
   the estimate equals walking the recorded book for that stake, plus fees. Where depth was
   missing, the estimate is marked as an extrapolation.
2. **Given** the execution model, **When** a backtest is re-stated, **Then** every trade carries
   its estimated entry and exit cost. Trade identity is unchanged; only the costs are added.
3. **Given** a held-out set of recorded days, **When** predicted and recorded impact are compared,
   **Then** the error is reported per pair. A pair outside tolerance is flagged.

---

### User Story 3 - Set the stake cap from execution cost (Priority: P3)

The user wants the fixed 1/10-of-candle-volume cap replaced, as an option that is off by default,
by a cap set from estimated cost: the largest stake whose estimated cost stays within a declared
budget per trade.

A capacity curve shows realistic net P&L across a range of budgets and the current cap. Each
point is judged with the existing harness: paired comparison, risk verdict, trade count.

**Why this priority**: this is the lever expected to raise dollar P&L, since capital is about 98%
idle. It is only measurable once US2's costs exist.

**Independent Test**: run the 15-fold control at the current cap and at each declared budget,
with net P&L under the execution model. Check that:
- off reproduces the current results exactly;
- every on-point reports gross, cost, net, risk and the realised stake against the requested
  stake.

**Acceptance Scenarios**:

1. **Given** the option is off, **When** any existing arm runs, **Then** trades, stakes and
   results are byte-identical to before this feature.
2. **Given** a budget, **When** a trade would exceed it at the configured stake, **Then** the stake
   is reduced to the largest stake within budget, and both the requested and granted stakes are
   recorded.
3. **Given** the capacity curve, **When** it is judged, **Then** each point is compared with the
   current cap on net P&L by the paired test and the risk verdict, and early and late halves are
   reported separately. Adoption is the user's decision.

---

### User Story 4 - Measure sharing one wallet with the momentum basket (Priority: P4)

[NEEDS CLARIFICATION: is this story in scope? With about 98% of NNNC's wallet idle, running NNNC
and the momentum basket in one wallet would add their dollar P&L with little contention. This is
deployment, not framework optimisation (memory `feedback_framework_not_portfolio`).]

**Why this priority**: largest dollar effect per unit of work, but it is a deployment choice for
the user, not an improvement to NNNC itself.

**Independent Test**: replay both strategies' existing fold trades into one shared wallet, and
report blocked or shrunk entries per strategy and the combined P&L against the sum of the two.

**Acceptance Scenarios**:

1. **Given** both strategies' trades on the same folds, **When** they share one wallet, **Then**
   the report lists each entry that was blocked or shrunk by the other strategy, and the combined
   P&L against the sum of the separate runs.

---

### Edge Cases

- **Snapshots time out or arrive late** (exchange rate limits, network): the order is recorded
  with depth marked missing. Coverage is reported per pair and per day, and a pair below the
  coverage floor is excluded from the execution model, not filled in.
- **A pair is delisted, or stops trading on the venue** during the dry-run: the bot keeps running
  on the remaining pairs, and the record says when and why.
- **Gap-filled bars** (Binance.US has many zero-volume 15-minute bars, about 54% for ZEC): the cap
  already blocks entries on them. The parity report must treat a live bar that a later data
  download fills differently as a data difference, not a strategy defect.
- **The live process restarts mid-trade:** open trades and records survive the restart, and no
  order is recorded twice.
- **The retrain falls on a day with open trades:** they finish under the model that opened them.
  Only new entries use the new model.
- **Dry-run fills are simulated** (filled when the price crosses), so they are not real fills.
  Depth snapshots are the evidence for cost. Dry-run fill prices are reported but never used as
  measured slippage.
- **Too few orders:** NNNC makes about 1.5 trades a day, so a short recording window may have too
  few orders per pair. The execution model must report its sample per pair, and a thin pair is
  pooled by a declared structural rule (for example by depth tier), never given its own tuned
  value (Principle X).
- **Depth changes by time of day or regime:** the snapshot record covers whole days, and the
  execution model reports whether cost differs by hour or by market-wide volatility before
  pooling across them.

## Requirements *(mandatory)*

### Functional Requirements

**Live dry-run (US1)**
- **FR-001**: The live NNNC bot MUST run as a dry-run only (no real orders), isolated from the
  momentum dry-run:
  - its own process, database, log, listening port and configuration file;
  - the momentum bot (`config_mom_dryrun_15m.json`, running process) is never stopped, restarted
    or reconfigured by this feature.
- **FR-002**: The configuration run live MUST be the current NNNC control: ARO
  (`NNNC_A_Raw_Open`) with the current defaults (24 h stop-loss grace, in-profit held-pair add),
  on the walk-forward pair list and stake settings.
- **FR-003**: The live model MUST be:
  - trained only on data before a recorded cut-off;
  - given a threshold picked on that span's validation tail by the standard walk-forward rule;
  - identified by its model key and recorded before start.

  The bot MUST refuse to start, and never train, if those weights are missing or differ from the
  record.
- **FR-004**: The model MUST be refreshed every 90 days, matching the walk-forward fold length the
  backtests were validated at. Each refresh records old and new identities, and trades are
  attributed to the model that opened them.
- **FR-005**: For every order the bot would place, it MUST record:
  - pair and signal candle;
  - signal price;
  - requested stake, and stake after the cap;
  - order time and type;
  - fill time and price, or timeout and cancel;
  - exit reason and price at close.

  Records MUST survive restarts without duplication.
- **FR-006**: At every order, and additionally on a fixed schedule for every pair in the list, the
  bot MUST record an order-book snapshot with at least the top 20 levels per side. A failed
  snapshot is recorded as missing.
- **FR-007**: A daily parity report MUST compare live trades against a backtest of the same days,
  on the same model, threshold and settings:
  - matched entries and exits;
  - live-only and backtest-only trades;
  - a cause for each difference, from a declared list (data revision, gap-fill, timing, defect).

  An unexplained difference MUST be flagged before any P&L read.
- **FR-008**: Recorded market data MUST stay in its own directory, never be committed, and be
  parsed as data, never executed.

**Execution model (US2)**
- **FR-009**: The estimated cost of entering or exiting a stake on a pair MUST be computed by
  walking recorded depth for that stake, plus fees. Estimates beyond recorded depth are marked as
  extrapolated.
- **FR-010**: The execution model MUST be:
  - validated on held-out recorded days, with prediction error reported per pair;
  - built from at least the minimum recording period (FR-019).

  Thin pairs MUST be pooled by a declared structural rule, not tuned per pair.
- **FR-011**: A re-statement tool MUST attach estimated entry and exit costs to every trade of an
  existing backtest, without changing trade identity. It reports gross, cost and net per fold and
  for the early and late halves.
- **FR-012**: The 15-fold NNNC control and the width book MUST be re-stated at the current cap
  under the execution model, as the new realistic baselines.

**Size-aware cap (US3)**
- **FR-013**: A cost-budget stake cap MUST be available as an option, off by default. When on, a
  trade's stake is the largest stake whose estimated cost is within the budget, never above the
  configured stake. Both the requested and granted stakes are recorded.
- **FR-014**: With the option off, every existing arm MUST be byte-identical to before this
  feature: trades hash, model keys and fingerprints.
- **FR-015**: A capacity curve MUST be produced on the 15 folds:
  - the current cap, and the cost-budget cap at a declared set of budgets (default 5, 10, 20 and
    40 bps per side);
  - net P&L under the execution model, with early and late halves reported separately;
  - each point judged against the current cap with the existing harness (paired test, risk
    verdict, trade count).

  The budget set MUST be committed before the curve is run.
- **FR-016**: The live bot's realised record MUST be compared with the execution model's
  prediction for the same orders after every 4 weeks of running. A sustained bias is reported
  before any capacity result is offered for adoption.

**Shared wallet (US4)**
- **FR-017**: [NEEDS CLARIFICATION: scope — see User Story 4] If in scope: replay both strategies'
  existing fold trades into one shared wallet, and report blocked or shrunk entries per strategy
  and the combined P&L against the sum of the separate runs.

**General**
- **FR-018**: The changes MUST NOT:
  - change model weights, labels, inputs or architecture;
  - add pair-specific code;
  - put strategy-specific code under Framework/.

  Parameter dictionaries merge from the parent. Promotion of any result to a default is the
  user's decision.
- **FR-019**: US2 and US3 MUST NOT be judged before the dry-run has recorded at least
  [NEEDS CLARIFICATION: minimum recording period before the execution model is built and the
  capacity curve judged — e.g. 4 weeks (about 40 NNNC orders, about 100k depth snapshots), 8
  weeks, or until a per-pair order count is reached].
- **FR-020**: Every result MUST be labelled PROVISIONAL. The 15 folds are one era (2023–2026),
  and the recorded depth comes from a single recent period (constitution, Principle II).

### Key Entities

- **Live model record**: model key, training cut-off, threshold, pick evidence, start and retire
  dates.
- **Order record**: one per intended order: signal, requested and granted stake, order and fill
  times and prices or timeout, exit, model identity.
- **Depth snapshot**: pair, time, top-N levels per side, trigger (order or schedule), or a missing
  marker.
- **Parity report**: per day: matched, live-only and backtest-only trades, with a cause for each.
- **Execution model**: cost of a stake per pair or depth tier for entry and exit, its sample size,
  and held-out error.
- **Re-stated result**: an existing arm's trades with costs attached; gross, cost and net per fold
  and half.
- **Capacity point**: a cap setting, net P&L, risk, paired verdict against the current cap, and
  realised against requested stake.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The live NNNC dry-run runs for the minimum recording period:
  - with at least 99% of orders carrying a complete order record;
  - with at least 95% of orders carrying a depth snapshot;
  - without any change to the momentum dry-run.
- **SC-002**: The daily parity report explains every live-against-backtest difference on at least
  95% of days. Any day with an unexplained difference is resolved or recorded as a defect before
  P&L is read.
- **SC-003**: The execution model predicts held-out recorded impact within a declared tolerance
  for every pair it is applied to. Pairs outside tolerance are excluded and listed.
- **SC-004**: The 15-fold NNNC control and width book have realistic baselines: gross, estimated
  cost and net, per fold and per half.
- **SC-005**: The capacity curve gives the net dollar P&L of every declared budget against the
  current cap, with the paired and risk verdicts. The user can then decide whether a larger size
  is justified, and at what budget.
- **SC-006**: With every new option off, existing arms reproduce their recorded results exactly.
- **SC-007**: The live model's operating point (share of eligible trades above threshold) is
  reported weekly. A drift outside 0.3–0.7 is flagged before the next retrain.

## Assumptions

- **Venue and data:**
  - Binance.US live market data and order books are reachable from here (the momentum dry-run
    already uses them);
  - order-book snapshots can be read without account keys;
  - rate limits allow a snapshot per order and a schedule of one per pair per 15 minutes.
- **Dry-run fills are simulated** by the trading engine, so they are not evidence of execution
  cost. Depth snapshots are. Real-money fills are out of scope.
- **The live configuration** is the current NNNC control (ARO with current defaults), not the
  `cs_aro` sizer and not the width book. The control made the most on the forward window, and it
  is the configuration all NNNC comparisons are made against.
- **Retrain cadence** of 90 days matches what the walk-forward backtests validated. The forward
  window's drift happened on a model 3–4 months old, so this is the longest acceptable interval.
- **Fees** are Binance.US's 0.10% per side, as in the backtests.
- **Scaler leakage:** the production scaler is fitted once across history (constitution,
  Principle XI).
  - The live model reuses that scaler unchanged, so its model key stays the validated one. The
    scaler was fitted on 2026-09-10 on data before the live period, so it cannot have seen live
    bars.
  - The 15-fold re-statement compares execution costs on the same trades, so any scaler leakage
    affects both sides equally and does not bias the cap comparison.
- **The pair list** is the walk-forward list (10 pairs; SUI excluded), unchanged.
- **Out of scope:**
  - real-money trading;
  - new pairs or venues;
  - label, input or architecture changes;
  - any change to the running momentum dry-run.
