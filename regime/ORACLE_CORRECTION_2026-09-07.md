# Oracle instrument defect — corrected figures for every affected result

**One defect, found 2026-09-07, invalidated every oracle number in this
repository.** This document is the canonical corrected record. Affected documents
carry a banner pointing here rather than being silently edited.

## The defect

`Framework/BaseNNStrategy.augment_training_signals` dilates every buy label onto
the **two preceding bars** ("Trick 1: artificially extend positive signals two
entries earlier"). Correct for training. But the oracle arms read
`get_training_labels()`, which runs AFTER augmentation, so **they entered up to
two bars early** — only 84.1% of entries satisfied the label they claimed to
follow. Entering early against a −2%..−4% stop is strictly worse, so **every
oracle figure was biased LOW**.

Fixed in `NNMT_Oracle` / `NNNC_Oracle` by ANDing the augmented trading class with
the unaugmented buy signal. `Framework/` untouched. Compliance 84.1% → 90.4%;
the 9.6% residual is a carried bound. Full diagnosis:
`specs/006-oracle-exit-diagnosis/instrument.md`.

## Corrected figures

### Guards-off oracle gate, W1 (PASS = >0 and ≥200 trades)

| arm | degraded | **corrected** | verdict |
|---|---|---|---|
| NNMT `gbb` 0.02 | 1561tr +17.791% | 1317tr **+62.006%** | PASS |
| NNMT `gbb_path` 0.02 | 1560tr +23.991% | 1304tr **+75.379%** | PASS |
| NNNC `gbb` 0.02 | 1333tr +26.544% | 1163tr **+71.552%** | PASS |
| NNNC `gbb_path` 0.02 | 1342tr +29.656% | 1173tr **+87.192%** | PASS |

### Guards-on ceiling, W1 — the gates the families actually run

| family | `gbb` | `gbb_path` | path advantage | real model | **headroom (gbb)** |
|---|---|---|---|---|---|
| NNMT | +11.809% | +14.248% | **+2.439pp** | +10.708% | **+1.101pp** |
| NNNC | +13.914% | +15.090% | **+1.176pp** | +10.446% | **+3.468pp** |

### Other corrected oracle baselines

| arm | degraded | corrected |
|---|---|---|
| NNMT oracle @0.007 W1 | +6.150% | +6.544% |
| NNMT oracle @0.02 W2 / W3 | +9.053% / +2.747% | +13.458% / +7.517% |

## What this CHANGES

**1. `WHY_ORACLES_UNDERPERFORM.md` was wrong and is rewritten.** All three of its
measurements reverse on the corrected instrument: the oracle stops out LESS
(35.3% vs the real model's 44.3%), reaches the take-profit the SAME (22.1% vs
22.2%), and carries LARGER excursions (4.57% vs 4.01%), with a higher win rate
(64.7% vs 55.7%). Its "binary floor versus continuous ranking" thesis was an
elaborate explanation of a bug and is retracted in place.

**2. A conclusion in `NNNC_PATH_AND_EXIT.md` is REFUTED.** It stated that
`gbb_path`'s ceiling was **0.338pp LOWER** at NNNC's production gate and used
that to reject a retrain. On the corrected instrument it is **+1.176pp HIGHER**.
The reasoning was wrong. (The retrain was run anyway on the open gate, so no
decision was lost — see point 4.)

**3. `NNMT_OPERATING_POINT.md`'s headroom claim needs qualification.** Its
−4.558pp was measured at the label-0.007 control and corrects to −4.164pp, so
that specific figure survives. But **at the label NNMT actually runs, corrected
headroom is +1.101pp — positive.** "Model work on NNMT is foreclosed" is true of
the 0.007 control and NOT of the current configuration.

**4. Every ceiling in `PATH_LABEL_GATE.md` is understated**, and the path
label's advantage more than doubles on NNMT (+6.2 → +13.4pp) and quintuples on
NNNC (+3.1 → +15.6pp).

## What this does NOT change

**Every realised-return result stands.** Real models never read the augmented
labels as an entry signal — augmentation affects what a model is TRAINED on, and
the trained models were unaffected by this bug. So:

- `gbb_path` retrained: NNMT **−0.091pp**, NNNC open gate **+0.747pp** — both
  still NULL on W1.
- The exit-signal promotion (+0.78pp, two genuine seed replicates) stands.
- The guard-chain findings, `entry_guard_threshold` inert, the 4.20×
  super-additivity, W3's net/gross instability, and the exchange survey all stand.

**And the central theme is STRENGTHENED, not weakened.** The corrected ceilings
are far larger, the corrected headroom on NNMT is positive rather than negative,
and the trained models still capture none of it:

| change | corrected ceiling movement | realised on W1 |
|---|---|---|
| `gbb` → `gbb_path`, NNMT | **+13.373pp** guards-off, +2.439pp guards-on | **−0.091pp** |
| `gbb` → `gbb_path`, NNNC | **+15.640pp** guards-off | **+0.747pp** (noise) |

The gap between what a label COULD pay and what a model DOES realise is not
smaller than previously thought. It is much larger.

## The methodological lesson

The measurement that would have caught this was cheap and was never run until
feature 006 made it a blocking task: **does the instrument do what it claims?**
Only 84.1% of the oracle's entries satisfied the label the oracle was supposedly
following, and no prior feature checked. A result contradicting something as
basic as "foreknowledge helps" should trigger an instrument audit before it
triggers a theory.
