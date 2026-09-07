# Quickstart — 006 Oracle-Instrumented Exit Diagnosis

Run from the repository root. **Nothing here trains a model** (research R9).

## Standing rules

W1 `20250901-20260831` is the SOLE acceptance window. W2 `20240601-20250531` and
W3 `20230101-20231231` are in-sample. **Report every window with its net/gross
ratio** — NNMT is 29.8% / 18.7% / 7.5%, and W3 cannot carry a sub-threshold
conclusion.

Arms run SEQUENTIALLY. Every perfect-entry figure is a LOOKAHEAD CEILING. Read
configuration ONE STRATEGY PER PROCESS (R10) or the values are wrong.

## Scenario 1 — validate the instrument (Phase 1, BLOCKING)

Recompute the share of perfect entries meeting their own label, against the
STRATEGY's dataframe rather than the raw feather.

**Expected**: either a named cause for the 84.1% and a corrected figure, or a
stated bound. Test order: dataframe/startup-candle mismatch, window-edge horizon
truncation, limit-fill timing, then eliminate the subset operations
(`_filter_trading_by_tasks`, `filter_peaks_by_future_performance`) which cannot
add non-compliant entries.

**Do not proceed to Scenario 2 until this closes.**

## Scenario 2a — the payoff geometry (Phase 2, first and nearly free)

    label target 2.0%   harvest 2.2%   stop -2%..-4%   =>  reward:risk 1:1 .. 1:2

Recompute the stop-breach rate per family with the CORRECT band — the earlier
6.6% used NNNC's band on NNMT data.

**Expected**: a statement of whether a perfectly-labelled trade is a
positive-expectancy bet at all. If it is not, that alone explains the paradox.

## Scenario 2b — the bound (Phase 2, HARD GATE)

    ... run_arm.py --strategy NNMT_Ora_Patient --window W1 --tag ora_patient \
        --roi 999 --set-sell cexit_take_profit=9.9 --set-sell cexit_max_days=999

with `use_atr_adaptive_stoploss=False` and HoldWindowMixin on the class.

**Expected**: the opportunity cost, against the current exit and against the real
model, with paired-by-trade resolution. **If it is not materially larger than
that resolution, STOP and re-scope Phases 3-5** — the exit is not the constraint.

## Scenario 3 — attribute the bound (Phase 3)

Re-enable one mechanism at a time against the patient arm. Also re-run feature
004's stale stop bands at the current label.

**Expected**: per-mechanism contribution and exit-reason mix; any inert mechanism
NAMED as inert; and the sum compared against the joint bound, since the entry-side
equivalent was 4.20x super-additive and addition must not be assumed.

## Scenario 4 — the target/harvest ordering (Phase 4)

Harvest point below, at, and above the label's target.

**Expected**: whether a gain comes from harvesting MORE moves or SMALLER ones.
Feature 004 found raising the take-profit monotonically worse; R3 predicts that,
because the label only guarantees +2%.

## Scenario 5 — confirm on the REAL model (Phase 5, the only path to promotion)

Every candidate re-measured on the real model, two seeds, paired by trade, W1.

**Expected**: a verdict decided by the real model alone. A patient exit is
rewarded where every entry is guaranteed to reach its target and punished where
they are not — an instrument-only win is the EXPECTED failure mode, not an
anomaly.

## Scenario 6 — hygiene

    git status --short Framework/    # must be clean

**Expected**: no shared mechanism modified; every ceiling figure labelled; no
perfect-entry configuration promoted.

## Success criteria coverage

| SC | validated by |
|---|---|
| SC-001 | Scenario 1 |
| SC-002 | Scenario 2b |
| SC-003 | Scenario 3 |
| SC-004 | Scenario 4 |
| SC-005 | Scenario 5 |
| SC-006 | every scenario (net/gross reported per window) |
| SC-007 | Scenarios 2a–4 (ceiling labelling) and 6 |
| SC-008 | Scenario 6 |
| SC-009 | Phase 6 conclusion |
