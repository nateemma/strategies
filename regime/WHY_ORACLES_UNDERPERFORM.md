# Why an oracle can underperform a real model

**⚠️ SUBSTANTIALLY REWRITTEN 2026-09-07. The original version of this document
was WRONG, and it was wrong because the instrument it measured was broken.**
The retraction is at the bottom; it is kept rather than deleted because the
reasoning failure is instructive.

## The corrected picture

An oracle whose entry signal is the training label SHOULD beat a real model, and
once the instrument is fixed, it does:

| NNMT W1 | trades | return | % stopped | % take-profit | win rate | avg MFE |
|---|---|---|---|---|---|---|
| real model | 176 | +10.708% | 44.3% | 22.2% | 55.7% | 4.01% |
| **oracle @ label 0.02** | 136 | **+11.809%** | **35.3%** | 22.1% | **64.7%** | **4.57%** |
| oracle @ label 0.007 | 150 | +6.544% | — | — | — | — |

The corrected oracle stops out LESS, wins MORE, carries LARGER favourable
excursions, and returns MORE. Exactly what perfect foreknowledge should buy.

## So when DOES an oracle underperform? Two real cases remain

**1. When the label itself is bad.** At label 0.007 the oracle earns +6.544%
against the real model's +10.708% — a genuine −4.16pp deficit that survives the
instrument fix. A label promising only +0.7% against a −2%..−4% stop describes
trades that lose money even when predicted perfectly. The oracle is a faithful
upper bound on *that label*, and the bound is below what a model trained on it
actually achieves, because the model learns something better than its target.

**2. When the payoff geometry is adverse.** NNMT promises +2.0%, harvests at
2.2%, and risks 2–4%. Even a correct label can be a poor bet. This is measured
per family in feature 006 and is the reason the 0.007 label fails so badly.

**Both are statements about the LABEL, not about the exit.**

## What an oracle bounds — unchanged, and still the important caveat

An oracle bounds **what perfect LABEL-FOLLOWING is worth**, not what perfect
trading is worth. `ORACLE_GATE.md` rule 0 stands: a ceiling below the real
model's return is evidence about the LABEL. What has changed is that on NNMT at
its current label, the ceiling is now correctly ABOVE the real model, so that
caveat no longer applies there.

---

# RETRACTION — what this document used to claim, and why it was wrong

**The original claimed**: that a perfect model underperforms a real one because
the label is a "binary floor" while the model learns a "continuous ranking", and
offered three measurements as proof:

| original claim | original figure | corrected figure | verdict |
|---|---|---|---|
| the oracle stops out MORE | 48.4% vs 44.3% | **35.3% vs 44.3%** | **REFUTED** — it stops out less |
| the oracle reaches the take-profit LESS | 15.5% vs 22.2% | **22.1% vs 22.2%** | **REFUTED** — identical |
| the oracle's trades have SMALLER favourable excursions | 3.72% vs 4.01% | **4.57% vs 4.01%** | **REFUTED** — larger |

**Why it was wrong.** Every figure came from an oracle that was entering **up to
two bars early**. `BaseNNStrategy.augment_training_signals` dilates each buy
label onto the two preceding bars — correct for training, catastrophic for an
instrument that reads the labels as an entry signal. Compliance with the raw
label was only 84.1%. Entering early against a −2%..−4% stop produces precisely
the symptoms the document then explained with a theory: more stops, fewer
take-profits, smaller excursions.

**The methodological lesson, which is the reason this retraction is kept.** The
"binary floor versus continuous ranking" story was plausible, internally
coherent, and fitted every number — and it was an elaborate explanation of a bug.
The measurement that would have caught it was cheap and was not run until much
later: *does the instrument actually do what it claims?* Only 84.1% of the
oracle's entries satisfied the label the oracle was supposedly following, and
nobody had checked.

**Before theorising about a surprising measurement, verify the instrument
produced the measurement it claims to.** A result that contradicts something as
basic as "foreknowledge helps" is far more likely to be an instrument defect than
a deep truth about labels.

See `specs/006-oracle-exit-diagnosis/instrument.md` for the diagnosis, the fix,
and the full blast radius across prior features.
