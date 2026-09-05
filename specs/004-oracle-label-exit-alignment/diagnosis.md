# T012 — Diagnosis (US1 + US4). **HARD GATE: satisfied.**

Read entirely from backtest archives already on disk (R1). **No parameter has
been varied at the time of writing** (FR-006). Every figure below is a
LOOKAHEAD CEILING measurement where an oracle arm is involved and is not an
achievable return (FR-012).

Mechanism labels use the R2 mapping: `trailing_stop_loss` is the **ATR adaptive
stop after it moved**, not a native trailing stop (`trailing_stop = False`
framework-wide). Coverage is 100% on every arm (SC-002).

## The arm under diagnosis — oracle, guards off, W1 (1760 trades)

    mechanism                             n     sum P&L    mean   mean h
    ATR stop (moved)                    709    -10774.3  -15.20    10.2
    ATR stop (initial)                    8     -1192.9 -149.11     3.0
    profit check (metric_overbought)      1         3.1    3.08    11.2
    hold window (unclog_1d)              23        14.0    0.61    27.2
    ROI table                            14       332.1   23.72     5.1
    hold window (unclog_12h)            385      1470.6    3.82    14.1
    cexit_take_profit                   620      8258.3   13.32     5.2
    TOTAL                              1760     -1889.2

## SC-001 — the dominant mechanism

**The ATR adaptive stop.** 717 trades of 1760 (**40.7%**) losing **−11,967.2**
in total. It is the only materially negative mechanism, and it destroys
**1.45× what the take-profit earns** (+8,258.3 over 620 trades).

Its share of destroyed P&L is effectively all of it: every other negative
contributor together accounts for under 4 units.

## SC-006 / FR-019 — does the loss survive costs?

**Yes, but only just — and cost is 79% of it.**

    arm             tr       net     fees     GROSS   loss survives costs
    gctl            95    1044.6    129.9    1174.5   n/a (profitable)
    oracle          89    1112.2    121.7    1233.9   n/a (profitable)
    gqoff         1970   -1446.5   1733.9     287.4   NO  — gross POSITIVE
    oracle_nog    1760   -1889.2   1484.1    -405.1   YES — gross −405.1

Two things follow, and the second is uncomfortable:

1. **Cost is the larger part of the guards-off loss but not the whole of it.**
   Fees are 1484.1 of the 1889.2 net loss. Removing them leaves −405.1: still
   a loss, so the label/exit mismatch is real and not merely a trading-cost
   artefact. US4's question is answered: the label is not exonerated.
2. **Gross of fees the REAL model is positive (+287.4) while the ORACLE is
   negative (−405.1).** Perfect foreknowledge of the label is worse than the
   real model even before costs. The oracle takes 210 fewer trades and loses
   more, so this is not a trade-count effect.

## FR-002 — what the guard chain suppresses

    arm                    tr   ATR-stop n   ATR-stop P&L   TP n   fees/|gross|
    oracle  guards ON      89           22         -651.1     52          9.9%
    oracle  guards OFF   1760          717       -11967.2    620         366%

The guards do not change the exit machinery at all — they change which bars
reach it. Guards-on, the stop fires on 24.7% of trades; guards-off, on 40.7%.

## FR-004 / FR-005 — the excursion evidence, and the actual mechanism

    arm                  favourable p50   adverse p50   reached −5% stop   exceeded +2.4% TP
    oracle  guards ON            +3.20%        −2.38%              24.7%              76.4%
    oracle  guards OFF           +2.13%        −2.15%              14.5%              44.7%
    gctl    guards ON            +3.18%        −2.23%              23.2%              75.8%
    gqoff   guards OFF           +2.13%        −2.15%              13.7%              45.3%

**This is the finding.** The exit stack caps a winner at `cexit_take_profit`
= +2.4% while letting a loser run to the ATR band at −4% to −6%. That is a
payoff ratio of roughly **0.45 : 1** — built into the EXITS, independent of any
label. Such a stack is only viable at a high win rate.

The guard chain supplies exactly that. Guards-on bars have a median favourable
excursion of +3.20% against −2.38% adverse, and **76.4%** of them run past the
take-profit — so nearly every trade reaches its cap and the 0.45 payoff ratio is
paid for by a ~75% win rate. Guards-off bars are ordinary: +2.13% against
−2.15%, and only **44.7%** ever reach the cap. At that hit rate the same
asymmetric stack loses, and it loses no matter how perfect the entry signal is.

**So the label is not what fails. The EXIT GEOMETRY fails on ordinary bars, and
the guard chain works by admitting only the extraordinary ones.** That is why a
perfect model cannot rescue it, and why six model-side changes were absorbed:
none of them touched the geometry or the selection.

## What this implicates, and what it does not

- **IMPLICATED: the ATR stop band and the take-profit cap, jointly.** They are
  one mechanism — a payoff ratio — and Phase 4 must vary them as such. T019
  (ATR band subclass) is therefore ON.
- **NOT implicated: the hold window.** `unclog_12h` contributes **+1470.6 over
  385 trades**, positive, and `unclog_1d` +14.0. The hold window is helping.
  **T020 is SKIPPED** and this is the record of why — it is not to be run
  speculatively (FR-013).
- **NOT implicated: pair mix (FR-020).** ZEC dominates both guards-on arms
  (35.8% of gctl trades, 36.0% of oracle) and remains the largest guards-off
  contributor (24.2%). The guards concentrate into ZEC but do not create it;
  guards-off admits the same instruments in different proportion, so the loss
  is a difference in BARS, not in instruments.

## Consequence for Phase 4

Variants must attack the payoff ratio, cheapest first: `cexit_take_profit` via
`--set-sell` (no new class), then the ROI table, then the ATR band via a
subclass. A variant that raises the cap without widening the stop should move
the oracle's guards-off return directly if this diagnosis is right — that is the
prediction the screening tests.
