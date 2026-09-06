# T008 / T009 — NNMT model headroom. **THE HARD GATE.**

**VERDICT: there is no headroom. A perfect entry signal is worth NOTHING on
NNMT at this gate, and is directionally WORSE in all three windows. Further
model-side work on this family cannot pay here (FR-004).**

All figures below from arms with `returncode` 0 and distinct trade hashes.
**Every oracle figure is a LOOKAHEAD CEILING. Nothing here is promotable.**

## The three-way comparison (SC-002)

| window | arm | trades | return | max DD | profit factor |
|---|---|---|---|---|---|
| **W1** (acceptance, OOS) | no model | 179 | +10.689% | 1.967% | 1.728 |
| | real model | 176 | **+10.708%** | 1.962% | 1.734 |
| | perfect model *(ceiling)* | 155 | +6.150% | 3.491% | 1.460 |
| W2 (in-sample) | no model | 365 | +10.148% | 4.344% | 1.378 |
| | real model | 347 | +8.331% | 4.106% | 1.319 |
| | perfect model *(ceiling)* | 312 | +5.400% | 4.613% | 1.226 |
| W3 (in-sample, **near-flat, 126 trades**) | no model | 134 | +1.133% | 5.108% | 1.099 |
| | real model | 126 | +1.238% | 4.938% | 1.118 |
| | perfect model *(ceiling)* | 118 | +0.148% | 4.935% | 1.014 |

Derived:

| quantity | W1 | W2 | W3 |
|---|---|---|---|
| **headroom** (perfect − real) | **−4.558pp** | −2.931pp | −1.090pp |
| **model contribution** (real − no model) | +0.019pp | −1.817pp | +0.105pp |

## T008 — the power threshold, from the OBSERVED count (R6)

`compare.py`'s 1.0pp floor was calibrated for ~97-trade arms and is **not**
imported. These arms are paired — they share most of their trades — so the
resolution is set by the SYMMETRIC DIFFERENCE of the trade sets, not by the
whole book. Bootstrapping the net P&L of the differing trades (4000 resamples):

| comparison (W1) | differing trades | net effect | paired 2sd | resolved? |
|---|---|---|---|---|
| no model vs real | 7 | −0.025pp | **1.049pp** | no |
| perfect vs real | 41 | −4.393pp | **4.659pp** | no |

**The rule for this feature**: an arm's threshold is `2 × sd` of a bootstrap over
the trades that differ from its control — roughly **1.0pp for a small entry-side
arm (~7 differing trades) and ~4.7pp for an oracle-scale arm (~41)**. Applying
NNNC's flat 1pp floor to the oracle arm would have overstated its resolution by
about 4.7×, and would have let this feature declare a −4.4pp "result" that its
own trade count cannot support.

For reference, the unpaired bootstrap over the whole W1 book (176 trades,
per-trade sd 34.10 on a 10 000 stake) gives a 2sd band of 9.10pp. That is the
right number for an UNPAIRED comparison and the wrong one here; it is recorded
so the paired figure is not mistaken for the only one available.

## What this means, stated carefully

**Nothing here is statistically resolved, and that is itself the finding.**

1. **The model's entry opinion does essentially nothing.** On W1 the real model
   and the no-model arm differ by 7 trades and 0.025pp against a 1.049pp
   resolution. The guard chain selects the trades. This reproduces NNNC's result
   on a different model class, and more extremely — there the bypass cost
   +0.78pp, here it costs nothing measurable.
2. **A perfect entry signal does not help, and points the other way.** Headroom
   is negative in all three windows. On W1 the magnitude (−4.393pp paired) sits
   just inside its own 4.659pp resolution, so the honest statement is *not*
   "the oracle is significantly worse" — it is **"headroom is certainly not
   materially positive, and is directionally negative in 3 of 3 windows"**. For
   the gate, that is the same answer.
3. **Do not read W2's +1.505pp no-model advantage as the model being harmful.**
   It is inside its own 2.435pp band, and W2 is in-sample.
4. **W3 carries 126 trades and is near-flat** (+1.238% control, PF 1.118), as
   research R7 predicted. It corroborates direction only; it accepts nothing.

## Why headroom is NEGATIVE — the mechanism, and where it sends the feature

A perfect signal on a BAD LABEL is a bad signal. The control runs the
pre-inheritance label gain threshold of **0.007**, and NNNC established that
0.007 is mis-specified against the operative ATR stop band: at the ceiling it
returned −18.89% on W1 while 0.02 returned +26.54%. A label that marks bars
whose forward gain need only reach 0.7% will mark many bars that the −4% to −6%
stop closes first. The oracle enters exactly those bars, with perfect knowledge,
and loses on them.

So this measurement does not merely say "no model work can pay". It says **the
label is the binding constraint on this family**, and it is direct evidence that
the inherited change from 0.007 to 0.02 — the very change this feature was
convened to adjudicate — is pointing the right way.

That is a Phase 4 question, and Phase 4 still runs.

## GATE DECISION (FR-004)

Headroom is not materially larger than the study's resolution. Per FR-004 and
the identical statements in tasks.md's Phase 2 callout, T009 and the Dependencies
section:

- **Phase 3 (guard attribution) RUNS.** It explains *why* headroom is absent, and
  it is the natural follow-up to a no-model arm that matches the real model.
- **Phase 4 (the inherited changes) RUNS.** It is a defect fix that stands
  regardless of headroom, and the negative ceiling has just made its label arm
  the most interesting measurement in the feature.
- **Phase 5 (operating point and exit side) is RE-SCOPED, not executed.**
  Sweeping a confidence threshold is meaningless when the confidence signal
  contributes 0.019pp across 7 trades. Any Phase 5 work must wait until a
  configuration exists in which the model actually selects trades.
- Phases 6 and 7 run.

**The COCONUT non-goal stays closed.** It was re-openable only if NNMT's headroom
proved materially larger than NNNC's. It is smaller — negative, in fact — so the
condition is not met and must be recorded as not met.

## An operational defect found here (not fixed — shared code)

`NNMT_Oracle` on W3 first failed with
`ValueError: Length of values (16269) does not match length of index (35009)`.
`NNMTStrategy.get_trading_classes` (~1459) writes its `%train_buy` / `%train_sell`
debug columns into `self.dbg_curr_df`, but `get_training_labels` only ASSIGNS
that attribute *after* `get_trading_classes` returns (~649). At training time
this is harmless — one call, one frame. Called per pair at inference, as an
oracle must, the attribute still holds the PREVIOUS pair's dataframe, so the
write raises whenever two pairs differ in length. W1 and W2 survive only because
every pair has full history there; W3 (2023) has ragged histories.

Worked around inside `NNMT_Oracle` by pointing the attribute at the current frame
before labelling. The defect itself is in shared family code and was deliberately
NOT fixed here (FR-026). It is latent for every NNMT strategy that ever calls
`get_training_labels` at inference.

`run_arm.py` refused to record metrics for the failed run and reported the
failure instead of the previous arm's numbers — the guard added on 2026-09-05
did its job (FR-027).
