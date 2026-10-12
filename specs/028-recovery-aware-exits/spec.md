# Feature Specification: Recovery-Aware Reversion Exits

**Feature Branch**: `028-recovery-aware-exits` (spec directory; work is committed to master)

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "Recovery-aware reversion exits: confirmatory follow-up to 027 (target-aware exits), which STOPPED at its feasibility gate because its pre-declared descriptive conditions targeted label-horizon 'killed winners' (only 2-3% of label-positive entries), while the measured mechanism is different: 57-63% of reversion stop exits recover to the +2% target LATER, within the 10-day max hold. 027's exploratory arms (post-hoc, NOT out-of-sample) all pointed the replay's way in both halves: C4 grace 6.5 h -> 12 h (late +10.46 vs +4.20), C2 take-profit = 2.897 x ATR (late +8.37), C1 stop never tighter than -3.71% (late +6.07); all UNCHANGED, arms realised 44-68% of the replay gain. This feature decides ADOPTION: pre-declare a mechanism-targeted feasibility condition, the candidates (C4, C2, C1, C4+C2; values frozen), and a decision rule on a FRESH judging window the 027 arms have not seen, using the paired P&L and risk verdicts; promote any adopted coupling into the ONE shared exit definition as a default-off switch so NNNC inherits it. Reuse 023-027 machinery. No pair-specific code. Framework optimisation for NNNC, not portfolio construction. Success = each candidate gets a recorded ADOPT / NOT ADOPTED verdict on the fresh window under the pre-declared rule, and any adopted coupling is a default-off switch in the shared exit with its stop invariants tested."

## Background *(PROVISIONAL under quality gate 2)*

**What 027 found.** The reversion family's exit is consistent with its 2% target over the 12-hour
label horizon. Few winners are stopped before they reach it. But over the 10-day max hold, most
stop exits recover:

| reversion (`st_book_rev2`) | early f00–f07 | late f08–f14 |
|---|---|---|
| stop exits | 699 | 466 |
| … that reach +2% later, within 12 h | 6.7% | 4.5% |
| … that reach +2% later, within the max hold | 63.4% | 57.5% |
| killed winners (share of label-positive) | 3.4% | 2.4% |

027's rules were written for killed winners, so they blocked all three couplings that gained in the
replay. Run afterwards as exploratory arms, all three gained in both halves:

| candidate (frozen value) | early / late pp | vs control +50.23 / +4.20 | late paired |
|---|---|---|---|
| C4 grace window 6.5 h → 12 h, same depth | +58.67 / +10.46 | +8.44 / +6.27 | UNCHANGED, p 0.32 |
| C2 take-profit = 2.897 × signal-bar ATR, clamped 1–5% | +54.70 / +8.37 | +4.46 / +4.17 | UNCHANGED, p 0.56 |
| C1 stop never tighter than −3.71% | +53.52 / +6.07 | +3.29 / +1.88 | UNCHANGED, p 0.50 |

**Why a new window is needed.** These candidates were picked after their late-half results were
seen. f00–f14 can rank them but can no longer judge them.

**What the data allows.** The walk-forward folds test 2023-01 to 2026-09. Every pair in the
universe has 15-minute data from 2022-01-20 on (five of ten from 2021). The reference arms carry
no model, so the folds' training windows were never used by them. 2022 has never been backtested
by any SimpleTrigger arm, and the reversion gate's threshold came from 2023+ trades. Data after
the last fold (2026-09-07 to 2026-10-10) is about one month.

## Clarifications

### Session 2026-10-12

- Q: Which fresh judging window? → A: 2022 on binance.us, from the end of the warm-up (about 2022-03) to 2022-12-31.
- Q: What counts as ADOPT? → A: Persistence. The fresh-window gain is positive, P&L and risk are not WORSE, and the sign matches both 2023–26 halves. (FR-005 formalises this as "not WORSENED", with every verdict value mapped.)
- Q: Family scope? → A: Reversion only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Pre-declare everything, then run the mechanism check on the fresh window (Priority: P1)

Before any judging-window data is touched, the researcher commits a pre-registration. It holds
the candidates and their frozen values, the judging window, the feasibility condition, and the
decision rule. Then, on the fresh window's control trades only, the replay measures:
- the share of stop exits that recover to the target within the max hold;
- the P&L those recoveries forgo;
- each candidate's replay gain.

Candidates that pass the condition go to US2. If none passes, the feature stops with the negative
recorded.

**Why this priority**: Constitution VII (validate cheaply) and II (persistence). The pre-registration
is what makes the fresh window a test, not another look.

**Independent Test**: A committed pre-registration whose commit precedes every fresh-window run
record; a feasibility report on the fresh window with the condition's values and a PASS/FAIL per
candidate.

**Acceptance Scenarios**:

1. **Given** the pre-registration, **When** the fresh window's first run record is written,
   **Then** its timestamp is later than the pre-registration's commit, and the record names it.
2. **Given** the fresh window's control trades, **When** the replay runs, **Then** its null replay
   reproduces at least 90% of recorded exits (reason AND bar) before any candidate is scored.
3. **Given** a candidate, **When** it is scored, **Then** it passes only if its replay gain is
   positive AND, for the stop-side candidates (C4, C1, C4+C2), the recovery share is at least the
   pre-declared floor. C2 alone changes the take-profit, not the stop, so the floor does not apply
   to it. The replay gain discounted by 2× is reported as the expected arm gain.
4. **Given** no candidate passes, **When** the report is written, **Then** the feature stops and
   records the negative with its numbers.

---

### User Story 2 - Judge each passing candidate on the fresh window (Priority: P2)

Each passing candidate runs as a walk-forward arm over the fresh window's folds, against the
unchanged control on the same folds. The pre-declared rule turns the paired P&L verdict and the
risk verdict into ADOPT or NOT ADOPTED.

**Why this priority**: Only an arm shows the real effect: the replay overstates the gain about 2×
because held trades change later entries.

**Independent Test**: For each candidate, the fresh-window per-fold P&L, the paired verdict, the
risk verdict, liveness counts and the resulting ADOPT / NOT ADOPTED.

**Acceptance Scenarios**:

1. **Given** a candidate arm, **When** it runs, **Then** its values are exactly those frozen in the
   pre-registration, and its run record shows it is distinguishable from the control (class,
   strategy hash or a recorded parameter, since the fingerprint cannot see method overrides).
2. **Given** a candidate arm, **When** it is judged, **Then** the verdict follows the pre-declared
   rule with no change after results are seen.
3. **Given** the C4+C2 combination, **When** it is judged, **Then** it is judged as its own arm,
   not inferred from its parts.
4. **Given** a candidate arm that changes no trades, **When** it is reported, **Then** it is
   inconclusive, not NOT ADOPTED.

---

### User Story 3 - Promote an adopted coupling into the shared exit (Priority: P3)

An adopted coupling moves out of its exploratory subclass and into the single shared exit
definition, as a switch that is off by default. Then any strategy (SimpleTrigger or NNNC) can turn
it on by declaration.

**Why this priority**: Adoption is only useful to NNNC if the coupling lives in the shared exit.

**Independent Test**: With the switch off, every recorded arm is unchanged. With it on, it
reproduces the exploratory arm's trades exactly. The stop invariants hold.

**Acceptance Scenarios**:

1. **Given** the switch off, **When** recorded arms run, **Then** their fingerprints and the
   reference arms' trades in all 15 folds are unchanged.
2. **Given** the switch on in a strategy otherwise identical to the exploratory arm, **When** it
   runs, **Then** its trades equal the exploratory arm's in every fold it ran.
3. **Given** any switch combination, **When** the stop is computed, **Then** the stop invariants
   (sign, never looser than the documented bound, grace rescale direction) hold, and a violation
   stops the run.
4. **Given** a coupling that is already an existing parameter (C4 is the grace length), **When** it
   is adopted, **Then** no new switch is added; the adoption is a strategy-level declared
   parameter (the grace stop is shared across families by definition), recorded as such.

---

### User Story 4 - Report and hand-over (Priority: P4)

One report gives the pre-registration, the fresh-window feasibility, each candidate's verdict, any
promoted switch, and the reversion exit NNNC should inherit. Every figure is labelled with its
evidence level.

**Why this priority**: This sets the default reversion exit for the model work.

**Independent Test**: The report exists with every candidate's verdict and the recommendation.

**Acceptance Scenarios**:

1. **Given** the feature ran, **When** the report is written, **Then** each candidate shows its
   2023–26 exploratory result and its fresh-window result side by side, labelled.
2. **Given** no candidate was adopted, **When** the report is written, **Then** it states that the
   current reversion exit stays, with the reasons.

### Edge Cases

- **Thin fresh window.** A fresh window shorter than the 15-fold history gives a low-powered paired
  verdict. The decision rule must be declared knowing that, and the report states the power.
- **Different regime.** If the fresh window is a different regime (2022 was a bear market), the
  reversion gate may switch most entries off. The pre-registration sets a minimum trade count.
  Below it the verdict is inconclusive, not NOT ADOPTED.
- **Indicator warm-up.** Rolling instruments (the gate uses 30-day trailing windows) need history
  before the window starts. The window must begin after every pair has that much data.
- **Pairs listed mid-window.** A pair listed after the window starts enters when it has the
  warm-up history. The universe is fixed in advance and is the same for control and candidates.
- **Trades open at a fold's end** are force-closed as in every arm; the replay already models this.
- **Candidates disagree.** If C4 is adopted and C2 is not, the combination's verdict does not
  override its parts. Only an adopted combination is promoted as a combination.
- **The fresh window already seen.** If any fresh-window backtest existed before the
  pre-registration, it is listed and the window cannot be used without saying so.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The candidates MUST be exactly C4 (grace 12 h, same depth), C2 (take-profit 2.897 ×
  signal-bar ATR, clamped [1%, 5%]), C1 (stop never tighter than −3.71%), and C4+C2, all at 027's
  early-chosen values. They MUST NOT be re-tuned.
- **FR-002**: A pre-registration MUST be committed before any judging-window backtest or replay.
  It MUST hold: candidates and values, judging window and universe, feasibility condition and its
  floor, decision rule, minimum trade count. Any later change MUST be recorded as a deviation in
  the report.
- **FR-003**: The judging window MUST be data no 027 candidate arm or reference arm has been
  backtested on, and MUST be 2022 on binance.us. It starts once every pair in the universe has the
  gate's 30-day warm-up after its listing (about 2022-03). It ends before f00's test window
  (2023-01-01) and is split into folds of the usual length. This is about 10 months of a bear
  regime that no SimpleTrigger arm has been backtested on (clarified 2026-10-12, Q1: A).
- **FR-004**: The feasibility condition MUST target the measured mechanism: the share of stop exits
  that recover to the target within the max hold (floor pre-declared, applied to the stop-side
  candidates), plus a positive replay gain for every candidate. The P&L those recoveries forgo
  MUST be reported alongside, not used as a condition. It MUST be computed on the judging window's control trades with the existing replay, after the
  replay passes its ≥ 90% fidelity check on that window.
- **FR-005**: The decision rule MUST be fixed in the pre-registration and MUST combine the
  fresh-window paired P&L verdict and the risk verdict. Its form is a persistence rule (clarified
  2026-10-12, Q2: B). A candidate is ADOPTED only if ALL of these hold:
  (a) its fresh-window gain over the control, measured as summed per-fold P&L, is positive;
  (b) its fresh-window paired P&L verdict is not WORSENED;
  (c) its fresh-window risk verdict is not WORSENED, for Sharpe or for max drawdown;
  (d) its gain has the same sign as its 2023–26 early AND late gains, also as per-fold sums
      (positive for all 027 candidates).
  Every value either verdict can return MUST be mapped to pass, fail or INCONCLUSIVE in the
  pre-registration, including INCONCLUSIVE_ARM_LOSES, which a bear-market window makes likely.
  The paired p-value and the window's power MUST be reported alongside, so the verdict is never
  mistaken for a significance test.
- **FR-005a**: For an ADOPTED gain below 1 pp, run-to-run variance MUST be shown to be nil
  (constitution quality gate 5). These arms carry no model and no seed, so this means one 2022
  fold re-run under a fresh tag reproducing an identical trade set.
- **FR-006**: Every candidate arm MUST be distinguishable from the control in its run record
  (class name and strategy hash, or a recorded parameter), because the framework fingerprint does
  not see method overrides.
- **FR-007**: Every candidate arm MUST report liveness: trades changed vs the control per fold. A
  zero-change arm is inconclusive.
- **FR-008**: The family scope MUST be reversion only (`st_book_rev2`) (clarified 2026-10-12,
  Q3: A). Momentum (`st_bgo_mx`) is out of scope: its 027 replay was consistent with its target.
- **FR-009**: An adopted coupling MUST be promoted into the single shared exit definition as a
  default-off switch, or as a strategy-level declared parameter where the coupling is already one (C4).
  No exit logic may be duplicated per strategy or family. No pair-specific code.
- **FR-010**: With every new switch off, every recorded arm MUST keep its fingerprint and both
  reference arms MUST reproduce their recorded trades in 15 of 15 folds.
- **FR-011**: With a promoted switch on, the strategy MUST reproduce the exploratory subclass's
  trades exactly on every fold both ran.
- **FR-012**: The stop invariants of 023 FR-017 MUST hold in every switch combination, with any new
  floor included in the documented bound. A violation MUST stop the run.
- **FR-013**: The report MUST label 2023–26 results EXPLORATORY and fresh-window results with their
  evidence level. Any result not meeting quality gate 2 MUST stay PROVISIONAL.

### Key Entities

- **Pre-registration**: the committed, dated record of candidates, values, window, universe,
  condition, floor, decision rule and minimum trades.
- **Judging window**: the fresh period, its folds, warm-up start and pair universe.
- **Candidate verdict**: per candidate (and family, if in scope), feasibility PASS/FAIL,
  fresh-window P&L and risk verdicts, liveness, and ADOPT / NOT ADOPTED / INCONCLUSIVE.
- **Exit switch**: a named, default-off behaviour in the shared exit, with its frozen value.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The pre-registration's commit is older than every judging-window run record (100% of
  records).
- **SC-002**: Every candidate in scope gets exactly one recorded verdict: ADOPT, NOT ADOPTED,
  INCONCLUSIVE, or stopped at feasibility. Each verdict cites the pre-declared rule.
- **SC-003**: The replay reproduces at least 90% of the judging window's control exits before any
  candidate is scored.
- **SC-004**: Every adopted coupling is in the shared exit. Switched off, 15 of 15 reference folds
  are unchanged. Switched on, its trades match the exploratory arm's in 100% of compared folds. No
  stop-invariant violation occurs.
- **SC-005**: The report names the reversion exit NNNC should inherit, with its evidence level.

## Assumptions

- Harness, fold length (about 90 days), costs, liquidity cap, stake and wallet are as in 023–027.
- The control is `st_book_rev2` (gated reversion alone) with today's exit. The candidates are its
  027 subclasses. Combining families or books is out of scope (framework, not portfolio).
- 027's replay discount (arms realise about half the replay gain) is used only for the feasibility
  gate and for ranking, never as the expected gain.
- Exit parameters that predate 023 (ATR band, grace depth, take-profit 2.2%) may have been tuned on
  periods that include 2022. The plan checks this. If they were, the window is fresh for the
  candidates, not for the whole exit, and the report says so.
- The model (NNNC) is not run. It inherits an adopted switch through the shared exit, and a later
  NNNC feature evaluates it with the model.
