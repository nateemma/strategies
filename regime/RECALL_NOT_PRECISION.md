# The gap is RECALL, and it is not reachable by the operating point

Measured 2026-09-07/08 on NNNC's open gate, W1, corrected oracle instrument.
**Oracle figures are LOOKAHEAD CEILINGS and nothing here is promotable.**

## What the gap actually is

NNNC at its OPEN gate carries the largest measured headroom in this repository:

| | trades | return |
|---|---|---|
| perfect entry (ceiling) | 1163 | **+71.552%** |
| real model (seed 42) | 437 | +8.720% |

Decomposing that +62.8pp against the oracle's own trade set:

| set | trades | P&L |
|---|---|---|
| both take | 157 | +7.832pp |
| **only the oracle takes (MISSED)** | **1006** | **+63.720pp** |
| only the model takes (bad picks) | 280 | **−0.216pp** |

**The model's bad picks cost essentially nothing.** Its problem is that it takes
**13.5%** of the available opportunities. Every model-side line this repository
has tried — nine architecture variants, two label changes, COCONUT, GAN
augmentation across eight arms, head weights, task filters — was aimed at making
picks BETTER. Precision was never the problem. Recall is, by a factor of seven.

## But recall cannot be bought at the operating point

Lowering the confidence threshold does raise recall. It does not pay:

| seed | thr | trades | return | recall | precision | missed P&L | bad-pick P&L |
|---|---|---|---|---|---|---|---|
| s1 | 0.80 | 321 | **+5.72%** | 11.3% | 40.8% | +62.52 | −3.32 |
| s1 | 0.75 | 1295 | −9.13% | 41.0% | 36.8% | +43.06 | −37.56 |
| s1 | 0.70 | 1789 | −16.10% | 45.1% | 29.3% | +30.82 | −56.55 |
| s7 | 0.80 | 413 | +2.97% | 13.3% | 37.5% | +66.26 | −2.32 |
| s7 | 0.75 | 1254 | −4.61% | 38.7% | 35.9% | +40.74 | −35.37 |
| s7 | 0.70 | 1807 | −13.72% | 44.9% | 28.9% | +28.56 | −56.49 |

Going 0.80 → 0.70 on seed 1 **recovers +31.7pp of missed opportunity** — the
model genuinely finds more oracle trades — and **pays −53.2pp in bad picks**. Net
−21.5pp, matching the realised move (+5.72 → −16.10) almost exactly. Seed 7
replicates.

**So the +63.7pp is unreachable at ANY threshold.** The binding constraint is the
model's RANKING: it can only find more of the oracle's trades by admitting many
more losers, and the losers cost more than the finds gain.

## The model is not useless — it is ~100x better than random

At threshold 0.70 it achieves 29.3% precision. The base rate is roughly 1163
oracle trades among ~385,000 eligible bars (11 pairs x ~35,000), about **0.3%**.
So the model carries strong signal. It is simply not strong enough to make high
recall profitable, which is a quantitative shortfall rather than an absence of
edge.

## `gbb_path` does not fix it — matched comparison at threshold 0.80

| label | oracle set | recall (s1/s7) | precision (s1/s7) | bad-picks (s1/s7) |
|---|---|---|---|---|
| `gbb` | 1163tr +71.552% | 11.3% / 13.3% | 40.8% / 37.5% | −3.32 / −2.32pp |
| `gbb_path` | 1173tr **+87.192%** | **9.6% / 9.5%** | 41.1% / 38.9% | −12.21 / −7.13pp |

Precision is flat, recall is WORSE, bad-picks are worse — while the opportunity
set is worth 15.6pp MORE. **The path label made the prize bigger without making
it more findable**, which is exactly why it realised −0.091pp (NNMT) and
+0.747pp (NNNC).

**This is the cleanest statement of the repository's recurring result**: label
quality and learnability are independent axes, and every intervention so far has
moved the first.

## What this forecloses, and what it opens

**FORECLOSED**: reaching NNNC's open-gate headroom by tuning the operating point.
Measured across two seeds and three thresholds; the trade is strictly negative
at every point.

**OPEN, and now sharply posed**: anything that improves the model's RANKING —
its ability to order oracle-trades above non-oracle-trades — at fixed recall.
Note this is a different target from every past attempt, which sought better
picks rather than better ordering.

**The first candidate, from the label side rather than the model side**:
`augment_training_signals` dilates every buy label onto the two PRECEDING bars.
That deliberately blurs the decision boundary by labelling bars that do not meet
the condition. Blurring is defensible when precision is comfortable — and this
document shows precision is the expensive axis. **Reducing the dilation width
(2 → 1 → 0) is a direct, cheap test against the measured constraint.**
