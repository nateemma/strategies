# Label dilation width: the metrics improve, the money does not

Measured 2026-09-08 on NNNC's open gate, W1, two seeds. **The sharpest
metric-versus-P&L divergence recorded in this repository**, because the metric
that improved was the exact one the P&L decomposition had identified as binding.

## The hypothesis, and why it was well-posed

`BaseNNStrategy.augment_training_signals` "Trick 1" copies every buy label onto
the **two preceding bars** — bars that do NOT meet the label's condition. It
deliberately blurs the decision boundary to give the classifier more positives.

`RECALL_NOT_PRECISION.md` had just measured that on NNNC's open gate, buying
recall by lowering the threshold costs **−53.2pp in bad picks** against +31.7pp
recovered. **Precision is the expensive axis.** Blurring the boundary is a
reasonable trade when precision is cheap, and this says it is not.

So: narrowing the dilation should sharpen precision, and precision is what costs.

## Control first

The dilation is MASKED rather than reimplemented — the parent's method still
runs in full, so tricks 2 and 3 are untouched and only buys outside the permitted
window are cleared. At width 2 this must reproduce the shipped behaviour.

| seed | width-2 arm | shipped `NNNC_MLX_Open` | trade hash |
|---|---|---|---|
| 1 | 321tr +5.719% | 321tr +5.719% | **MATCH** |
| 7 | 413tr +2.972% | 413tr +2.972% | **MATCH** |

**Control valid.** Byte-identical, so the other arms are readable.

## The result: BOTH metrics improve, return does not

| width | trades | return | recall | precision |
|---|---|---|---|---|
| **0** (none) | 491 / 490 | +0.368 / +3.043 | **18.4% / 17.8%** | **43.6% / 42.2%** |
| 1 | 327 / 339 | +3.969 / +2.300 | 11.3% / 11.8% | 40.4% / 40.4% |
| 2 (shipped) | 321 / 413 | +5.719 / +2.972 | 11.3% / 13.3% | 40.8% / 37.5% |

Removing the dilation improves **precision by +2.8 and +4.7pp** and **recall by
+7.1 and +4.5pp** — both axes, both seeds, exactly as predicted.

**Realised return moves the opposite way and is NOT resolved:**

| width | mean return | seed spread |
|---|---|---|
| 0 | +1.706% | 2.675pp |
| 1 | +3.135% | 1.669pp |
| 2 (shipped) | **+4.346%** | 2.747pp |

−2.640pp on the mean, but per seed: **−5.351pp against a 10.57pp band** and
**+0.071pp against 14.19pp**. Within noise on both, and the seeds disagree in
sign. The apparent monotonic ordering is carried almost entirely by seed 1.

## Verdict

**No basis to change the dilation. Width 2 stands** — not because it is proven
better, but because nothing is resolved and it is the incumbent.

**The mechanism the hypothesis proposed is REAL.** The blur genuinely was costing
discrimination; removing it genuinely sharpens the boundary on both axes. It
simply does not convert into return.

## Why this instance matters more than the previous six

| intervention | metric moved | realised W1 |
|---|---|---|
| label 0.007 → 0.02 | ceiling +45.9pp | −0.157pp |
| `gbb` → `gbb_path` | ceiling +13.4pp | −0.091pp |
| nine architecture variants | val_mcc within a 0.012 band | all negative |
| P&L-weighted loss | val_mcc up 6/6 monotonic | noise, sign-flips |
| **dilation 2 → 0** | **precision +4.7pp AND recall +7.1pp** | **−2.64pp, unresolved** |

Every earlier null could be explained away as "the metric was the wrong proxy".
This one cannot. The metric was chosen BY the P&L decomposition itself — bad
picks cost 53pp, so precision was named as the constraint, and precision then
improved without the P&L following.

**The conclusion is that classification quality and realised return are decoupled
on this data even when the metric is selected by measuring the P&L.** That is a
stronger and more uncomfortable statement than "MCC is a poor proxy", and it
argues that the remaining gap is not reachable by improving what the classifier
does with the information it has.
