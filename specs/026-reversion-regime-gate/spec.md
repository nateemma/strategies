# Feature Specification: Regime Gate on the Reversion Family

**Feature Branch**: `026-reversion-regime-gate` (spec directory; work is committed to master)

**Created**: 2026-10-11

**Status**: Draft

**Input**: User description: "Regime gate on the reversion family (follow-up to features 024/025). Feature 025 showed the combined arm's late-half loss is entirely its reversion (gbb) side: negative late in all 15 arms (-14 to -31pp) whatever the coexistence rule or exit setting, while the momentum side (breakout_gbb) earns in both halves and momentum alone is the best arm (+25.19pp, 9/15, early +22.68 / late +2.50). gbb also collapses late on its own (023: +30.33 early / -19.28 late), the signature of a regime shift. Goal: a causal regime gate that switches reversion-family entries off when the market is in a state where reversion loses, leaving the momentum family untouched, inside the 024/025 combined arm. Constraints from prior work that the design must respect: (a) the trackability bar is the better of the two constant choices actually being switched between — reversion always on (st_fam_combo, +15.39) and reversion always off (momentum alone, st_bgo_mx, +25.19) — never the status quo; (b) five earlier adaptive schemes lost to a constant (022), and a BTC 'require uptrend' gate removed ~90% of buy-low entries because buy-low entries cluster in market weakness — so candidate regime states must be judged on whether they separate reversion's losing periods, not on passive regime returns; (c) constitution VII: a cheap feasibility measurement must gate the build — first test, on existing trades and data only, whether any pre-declared causal regime state separates reversion's profitable and losing trades consistently in BOTH halves, and stop if none does; (d) the regime state and its threshold are fixed on early folds f00-f07 and judged on f08-f14; a small pre-declared set of instruments, two states each; (e) market-wide state only, no pair-specific code; (f) per-bucket trade splits only choose what to test — walk-forward arms decide. Success = either the feasibility gate stops the feature with a recorded negative, or a gated combined arm is compared against momentum alone and the always-on combined arm by half, with paired verdicts; all PROVISIONAL under gate 2. Reuse 023-025 machinery."

## Background *(PROVISIONAL under quality gate 2)*

| arm | per-fold sum | early (f00–f07) | late (f08–f14) |
|---|---|---|---|
| reversion always ON: combined `st_fam_combo` (024) | +15.39 | +35.76 | −20.37 |
|   … its reversion side | −1.25 | +23.45 | **−24.70** |
|   … its momentum side | +16.64 | +12.31 | +4.33 |
| reversion always OFF: momentum alone `st_bgo_mx` | **+25.19** | +22.68 | +2.50 |
| reversion alone `st_gbb` (023) | +11.04 | +30.33 | −19.28 |

- Across 025's 15 arms, the reversion side was negative in the late half in every one, from −14 to
  −31pp, whatever the coexistence rule or exit setting.
- The early half is where reversion earns, which is why always-off loses early (+22.68 vs +35.76).
- A gate is worth building only if it keeps the early reversion profit and avoids the late loss.

**Prior regime work this design must not repeat** (memory):
- **The BTC "require uptrend" gate removed ~90% of buy-low entries.** Buy-low entries cluster in
  weakness, so passive regime returns are the wrong yardstick.
- **Feature 001's phase gate failed.** Marginal trades were profitable in every state.
- **022:** two-state guard switching degenerated and lost to a constant, the fifth adaptive scheme
  to do so.
- **The trackability bar must be the better of the two real choices,** never the status quo.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cheap feasibility gate: does any regime state separate reversion's losses? (Priority: P1)

Before building anything, the researcher measures, on existing trades and price data only, whether
any pre-declared causal market-wide regime state separates the reversion family's profitable trades
from its losing ones. The state's threshold is fixed on the early folds. The separation must point
the same way in BOTH halves. If no instrument passes, the feature stops here and the negative is
recorded.

**Why this priority**: Constitution VII. Five adaptive schemes have already lost to a constant in
this project, and a regime gate is one more adaptive scheme. An afternoon's measurement must be
able to kill it before a build.

**Independent Test**: A feasibility report covering each pre-declared instrument. For each it gives:
the threshold fixed on f00–f07; the reversion trades and P&L per state in each half; whether the
"off" state carries reversion's loss in both halves; and the share of reversion's early profit the
gate would sacrifice. It ends with a PASS/STOP verdict per instrument and overall.

**Acceptance Scenarios**:

1. **Given** the pre-declared instruments, **When** their states are computed, **Then** each is causal:
   computed from bars at or before the entry bar, market-wide, and identical for every pair at a given time.
2. **Given** an instrument, **When** its threshold is chosen, **Then** only f00–f07 data is used.
3. **Given** an instrument, **When** it is scored, **Then** it PASSES only if its "off" state has
   negative reversion P&L in BOTH halves, its "on" state has positive reversion P&L in the early half,
   and the "on" state keeps at least a stated minimum share of early reversion trades (no gate that
   simply switches reversion off almost everywhere).
4. **Given** no instrument passes, **When** the gate finishes, **Then** the feature stops and the
   negative result is recorded with its numbers (constitution: a negative result that passes the gates
   is a successful outcome).

---

### User Story 2 - Walk-forward test of the gated combined arm (Priority: P2)

For each instrument that passed US1, the researcher runs the combined arm with reversion entries
switched off in the "off" state. The momentum family is untouched. The early-fold winner among the
passing instruments is judged on the late folds against the trackability bar: the better of reversion
always-on and reversion always-off, which is momentum alone.

**Why this priority**: Per-bucket trade splits only choose what to test. Removed entries get replaced
by other entries (023 Addendum 7), and only a walk-forward arm shows the real effect.

**Independent Test**: Gated arms with liveness evidence (entries refused by the gate, per half). An
early-chosen winner. Its late-half paired verdict against momentum alone AND against the always-on
combined arm.

**Acceptance Scenarios**:

1. **Given** a gated arm with the gate disabled, **When** it runs, **Then** it reproduces `st_fam_combo`
   exactly.
2. **Given** a gated arm, **When** it runs, **Then** it reports how many reversion entries the gate
   refused per half, and the momentum family's entries are not refused by the gate.
3. **Given** the early winner, **When** it is judged, **Then** it is adopted only if its late-half
   verdict is IMPROVED against the better of the two constants. Beating only the always-on arm is
   reported but is not adoption.

---

### User Story 3 - Report (Priority: P3)

One report gives the feasibility table, the gated arms, and the final comparison against both
constants and reversion alone, by half. Every figure is labelled PROVISIONAL.

**Why this priority**: The decision it feeds is whether the two-family arm with a regime gate replaces
momentum alone as the reference.

**Independent Test**: The report exists with all rows, halves, paired verdicts and labels.

**Acceptance Scenarios**:

1. **Given** the feature stopped at US1, **When** the report is written, **Then** it shows why each
   instrument failed and states the stop.
2. **Given** US2 ran, **When** the report is written, **Then** it states plainly whether the gated arm
   beat momentum alone in each half.

---

### Edge Cases

- **An instrument whose "off" state covers nearly all of the late half.** The gate then equals
  "reversion off late", which is a calendar switch in disguise. The minimum-share rule on early "on"
  trades guards one side. The report also states the "off" share of bars in each half, so a gate that
  flips once and stays flipped is visible.
- **Warm-up.** An instrument needing long history (a long EMA, a rolling correlation) is undefined at
  the start of the data. Those bars count as "on" (reversion allowed, the 024 behaviour) and the
  report counts them.
- **Market-wide state with pairs missing data at a time** (late listings, the XRP gap): the state is
  computed from the pairs that have data at that bar, with a minimum pair count. Below the minimum
  the state is undefined, so "on".
- **BTC as an instrument.** BTC is not one of the traded pairs. Its state is used as market context
  only, never as a per-pair rule.
- **Only one instrument passes US1.** US2 runs with it alone. The stage choice is then trivial, but
  the late judgement still applies.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST evaluate a small pre-declared set of causal, market-wide regime
  instruments, each reduced to two states ("reversion on" / "reversion off"):
  - (a) market breadth: the share of the traded pairs whose close is above their own long moving
    average;
  - (b) a cross-pair volatility regime: median pair volatility against its own trailing median;
  - (c) trend persistence: cross-pair median return autocorrelation or variance ratio;
  - (d) a BTC trend state.
  The set is fixed before any measurement.
- **FR-002**: Every instrument MUST be causal, using only bars at or before the decision bar. It MUST be
  computed identically for every pair at a given time, with no pair-specific logic (constitution X),
  and MUST be verified causal by the 023 truncation method.
- **FR-003**: Each instrument's two-state threshold MUST be fixed using f00–f07 data only, by a
  pre-declared rule (e.g. the early-period median, or the threshold maximising early reversion P&L
  separation over a small fixed grid).
- **FR-004**: The feasibility gate (US1) MUST use only existing trades and price data, with no new
  backtests. It MUST score each instrument on the reversion family's trades from the always-on combined
  arm and from reversion alone. PASS requires all of:
  - the "off" state's reversion P&L is negative in BOTH halves;
  - the "on" state's reversion P&L is positive in the early half;
  - the "on" state keeps ≥ 50% of early reversion trades.
  If no instrument passes, the feature MUST stop after the report.
- **FR-005**: The gate MUST affect only reversion-family entries. Momentum-family entries, all exits
  and all other framework settings MUST be unchanged. Entries are refused, never closed.
- **FR-006**: A gated arm with the gate disabled MUST reproduce `st_fam_combo` trade-for-trade.
- **FR-007**: Every gated arm MUST report the number of reversion entries the gate refused, per half
  (liveness, constitution IV), and the share of bars in the "off" state per half.
- **FR-008**: Among passing instruments, the gated arm MUST be chosen on f00–f07 and judged on f08–f14
  (the 025 stage runner). It is adopted only if the late verdict is IMPROVED against the better of the
  two constants (reversion always-on `st_fam_combo`; reversion always-off `st_bgo_mx`).
- **FR-009**: Every gated configuration MUST be distinguishable by its framework fingerprint, and
  recorded 023–025 arms MUST keep theirs.
- **FR-010**: The report MUST cover the feasibility table, all gated arms, and the final comparison
  against both constants and reversion alone, by half, with paired verdicts. Every figure MUST be
  labelled PROVISIONAL (gate 2).

### Key Entities

- **Regime instrument**: a named causal market-wide series, its two-state rule and its early-fixed
  threshold.
- **Feasibility record**: per instrument and per half, the reversion trades and P&L in each state,
  the share of "off" bars, the early "on"-trade share, and PASS/STOP.
- **Gated arm**: the combined arm with one instrument gating reversion entries, plus its refusal
  counts.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every instrument's threshold is documented as fixed on f00–f07, and its causality check
  passes.
- **SC-002**: The feasibility report gives, for every instrument, per-state reversion P&L in both
  halves and a PASS/STOP verdict against FR-004's three conditions.
- **SC-003**: If any instrument passes, the gated arm with the gate disabled reproduces `st_fam_combo`
  in 15/15 folds, and every gated arm shows non-zero gate refusals.
- **SC-004**: The final answer states, with paired intervals, whether the gated combined arm beats
  momentum alone in each half and overall. Otherwise it records the stop at US1 with the reason.
- **SC-005**: Arm D and all 023–025 recorded arms keep their fingerprints and stay reproducible.

## Assumptions

- Data, folds, harness, causality gate and quality gates are unchanged from 023–025. There is no
  distant era, so every result is PROVISIONAL (gate 2), with persistence tested by halves only.
- The families stay as in 024: `gbb` (reversion, ARM_D exits) and `breakout_gbb` (momentum, no
  take-profit, 24h hold). No coexistence rule and no exit changes (025 adopted none).
- BTC 15m data (`BTC_USDT-15m`) is available over the full span and is used as market context only.
- "Long moving average" and other instrument lengths are fixed in the plan from a small pre-declared
  set, not tuned per fold.
- Regime instruments using the traded pairs use the walk-forward whitelist (10 pairs).
- Per-bucket trade splits (US1) choose what to test. They are never quoted as the expected gain
  (023 Addendum 7, 025).
