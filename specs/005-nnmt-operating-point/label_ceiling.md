# FR-005 — the ceiling at both label settings, and the oracle gate

Measured with NO training. An oracle replaces the model's entry opinion with the
training label, so it never consults the model: comparing two labels at the
ceiling costs two backtests rather than two retrains (Constitution VII).

**Every figure here is a LOOKAHEAD CEILING. None is promotable.**

## Guards-on ceiling, at NNMT's own gate

| window | label 0.007 (pre-inheritance) | label 0.02 (inherited) | Δ |
|---|---|---|---|
| **W1** (acceptance) | 155 tr, +6.150%, PF 1.460 | 145 tr, **+8.733%**, PF 1.771 | **+2.583pp** |
| W2 (in-sample) | 312 tr, +5.400%, PF 1.226 | 292 tr, +9.053%, PF 1.435 | +3.653pp |
| W3 (in-sample, near-flat, 98–118 tr) | 118 tr, +0.148%, PF 1.014 | 98 tr, **+2.747%**, PF 1.363 | +2.599pp |

The inherited label raises the ceiling in **all three windows**, by a consistent
+2.6 to +3.7pp, and raises profit factor in all three while *reducing* trade
count. It is a better target, not merely a busier one.

## The oracle gate (`regime/ORACLE_GATE.md`) — the standing screen

PASS requires W1 **guards-off** oracle return > 0 **and** at least 200 trades.

| label | W1 guards-off return | trades | max DD | PF | verdict |
|---|---|---|---|---|---|
| 0.007 (pre-inheritance) | **−28.113%** | 2126 | 31.85% | 0.796 | **FAIL** |
| 0.02 (inherited) | **+17.791%** | 1561 | 7.63% | 1.200 | **PASS** |

**The inherited label change takes NNMT from a gate FAIL to a gate PASS.**

With perfect foreknowledge and the guards removed, the old label describes a set
of trades that *loses 28%* — the label marks bars whose forward gain need only
reach 0.7%, and the −4% to −6% ATR stop closes many of them first. The new label
describes trades that make 17.8% under the same conditions, on 1561 trades and
with drawdown cut from 31.9% to 7.6%.

This reproduces, on a different model class, exactly what was found on NNNC
(−18.89% → +26.54% at the ceiling). The label was mis-specified against the stop
band, and it was mis-specified on both families.

## What this does and does not license

**Licensed**: training. The gate's purpose is to refuse retrains on labels that
cannot pay even when predicted perfectly. Label 0.02 passes, so the seed-paired
retrains in T015 are justified spending.

**Not licensed**: any claim that NNMT will realise this. A gate PASS says a label
*could* pay, not that a model *will* learn it — on NNNC the model captured under
4% of a large ceiling gain. And note the guards-**on** ceiling at label 0.02 is
+8.733% on W1, still **below** the real model's +10.708%, so headroom at NNMT's
current gate remains negative even with the better label. The +17.791% appears
only with the guards removed, and Phase 3 measured guards-off on the real model
at label 0.007 as costing −34.15pp.

So the two facts that must be held together are:

1. At NNMT's **current** gate, the label change lifts the ceiling but leaves no
   headroom — the guards still select, and the model still cannot pay.
2. At an **open** gate, the label change is the difference between a −28% and a
   +18% ceiling. Whether a trained model realises any of that is precisely what
   T015 measures, and it is the only remaining question in this feature with a
   plausible positive answer.

## Bearing on the Phase 2 gate decision

This does not reopen Phase 5. The re-scoping decision was that sweeping a
confidence threshold is meaningless while the model contributes 0.019pp across
7 differing trades, and that remains true at the control label. If the retrained
label-0.02 arm turns out to select trades materially differently, the operating
point becomes a live question again — and that is recorded here as the specific,
falsifiable condition under which Phase 5 would be reinstated, rather than left
as a matter of judgement later.
