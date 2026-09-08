# The exit is not the constraint — and the instrument was broken

**Status**: settled 2026-09-07 (feature 006). Supersedes the exit-side reasoning
in `WHY_ORACLES_UNDERPERFORM.md`, which was rewritten, and corrects the headroom
justification in `NNMT_OPERATING_POINT.md`.

**One-line answer: the exit stack is not withholding return — a patient exit is
WORSE on the acceptance window. The feature's real product was discovering that
every oracle figure in this repository had been measured on a broken instrument.**

## What the feature was convened to explain, and why the question dissolved

A perfect-entry arm earned LESS than the real model (+8.733% against +10.708% on
NNMT W1). That should be impossible, and the diagnosis was expected to implicate
the exit.

It was the instrument. `BaseNNStrategy.augment_training_signals` dilates every
buy label onto the **two preceding bars** — correct for training, since it gives
the classifier more positive examples. But the oracle arms read
`get_training_labels()`, which runs AFTER augmentation, so **they entered up to
two bars early**, on bars whose forward move had not yet been established. Only
**84.1%** of entries satisfied the label they claimed to follow.

Corrected (`NNMT_Oracle` / `NNNC_Oracle` now AND the augmented class with the
unaugmented signal; `Framework/` untouched, so training keeps its augmentation):

| NNMT W1 | trades | return |
|---|---|---|
| real model | 176 | +10.708% |
| **perfect entry, corrected** | 136 | **+11.809%** |

The oracle now beats the real model, as it always should have. Compliance
84.1% → 90.4%; the 9.6% residual is a carried bound on every figure here.

## The exit measurement, once the instrument was trustworthy

Perfect entries under the current exit versus a patient exit held to the label's
own 12-hour horizon — stop off, ROI off, take-profit lifted, `unclog_12h`
deferred. **All LOOKAHEAD CEILINGS; the patient arm disables the stop and could
never run live.**

| window | current exit | patient exit | bound | paired 2sd | verdict |
|---|---|---|---|---|---|
| **W1** (acceptance) | +11.809% | +8.448% | **−3.361pp** | 8.37 | within noise |
| W2 (in-sample) | +13.458% | +33.066% | +19.608pp | 11.32 | RESOLVED |
| W3 (in-sample) | +7.517% | +10.098% | +2.581pp | 2.95 | within noise |

**A patient exit is WORSE on W1.** Waiting for the promised move means sitting
through the drawdown on the way, and the ATR stop was banking more than it cost.

W2's +19.6pp is resolved and is the largest exit-side effect measured in this
programme — but W1 and W2 disagree in SIGN, the signature of a regime-specific
effect rather than a mechanism. It is recorded as an open observation, never a
finding: nothing is accepted on an in-sample window that contradicts W1.

## The payoff geometry, which made this predictable

| family | median actual stop | harvest | reward : risk | breaches its own stop before target |
|---|---|---|---|---|
| NNMT | −2.58% | +2.2% | **0.85 : 1** | **16.3%** |
| NNNC | −4.15% | +2.4% | **0.58 : 1** | 5.8% |

Both families risk more than they harvest even on guaranteed moves. That is
survivable rather than broken — the corrected oracle wins 64.7%, so
`0.65 × 2.2% − 0.35 × 2.58% ≈ +0.53%` per trade.

**The one concrete exit observation worth keeping**: NNNC has the WORSE ratio but
breaches its stop **3× less often**, because its wider band (−4.15% against
−2.58%) gives the move room to arrive. Tightening the stop improves the ratio and
costs breaches; loosening does the reverse. **NNMT sits on the tight side of that
trade-off.** Not actionable without a real-model test, which the re-scope removed.

## What this FORECLOSES

A future proposal to change the exit on either family can be rejected without new
measurement if it relies on:

1. **"The exit closes trades before their edge materialises."** Measured. Holding
   to the label horizon is −3.361pp on W1.
2. **"Perfect entries are being wasted by the exit."** Perfect entries now BEAT
   the real model; there is no waste to recover.
3. **"The take-profit is mis-ordered against the label."** Feature 004 found
   raising it monotonically worse, and the geometry explains why: the label
   guarantees only +2.0%, so lifting the harvest above it converts certain
   winners into trades that run to a stop.

**NOT foreclosed**: the stop-band trade-off above, and W2's unexplained +19.6pp.
Both need a real-model test, which is where any reopening must start.

## The methodological result, which outlived the feature

**A result contradicting something as basic as "foreknowledge helps" should
trigger an instrument audit before it triggers a theory.** The prior explanation
— that the label is a "binary floor" while the model learns a "continuous
ranking" — was plausible, internally coherent, fitted every number, and was an
elaborate account of a bug. The measurement that would have caught it was cheap
and had never been run: *does the instrument do what it claims?*

Two further habits earned here:

- **An exit change that lengthens holds silently shrinks the entry set.** The
  patient arm took 26% fewer trades. A slot-contention hypothesis was raised and
  then REFUTED by raising the limit (counts identical, peak 10/40) — the cause is
  per-pair occupancy, and it cannot be separated from the intervention because it
  IS the intervention.
- **Report a window's net/gross ratio, not just its trade count.** See
  `W3_INSTABILITY.md`.

## Corrected ceilings, and the question this leaves open

| family | real | ceiling `gbb` | headroom | ceiling `gbb_path` | headroom |
|---|---|---|---|---|---|
| NNMT | +10.708% | +11.809% | **+1.101pp** | +14.248% | **+3.540pp** |
| NNNC | +10.446% | +13.914% | **+3.468pp** | +15.090% | **+4.644pp** |

Headroom is POSITIVE on both families — feature 005 foreclosed model work on
NNMT for having negative headroom, and that justification is now corrected.

**And no trained model has captured any of it**: `gbb_path` was retrained at two
seeds on both families and realised −0.091pp (NNMT) and +0.747pp (NNNC, noise)
against ceiling gains of +3.5 and +4.6pp.

**That is the successor question, and it is not an exit question:** why is a
3–5pp measured gap unreachable by every model-side change tried?
