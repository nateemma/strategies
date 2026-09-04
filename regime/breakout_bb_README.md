# `labels_breakout_bb` (method 26) — Bollinger breakout label

Added 2026-09-03. Buy when `bb_position` is high (riding the top of the band),
`bb_width` confirms a genuinely wide band rather than a flat squeeze, and there
is forward follow-through. Sell label is the mirror at the lower band.

**Rationale:** both inputs are in the classifier's ACTIVE feature set
(`bb_position`, `bb_width`), which is the property that makes a label learnable
here — the price-based Donchian breakout (method 20) only reached MCC 0.63–0.67
because the model must infer the level indirectly, while gbb and inverse-gbb hit
0.75–0.83 by keying off visible features. Of the candidates considered
(Bollinger / Keltner / Williams %R / Fibonacci), Bollinger was the ONLY one whose
inputs the model already sees.

## Defect found while building it: `bb_position` scale was mis-documented

`DataframePopulator` computes `bb_position` on [0, 1] and then **rescales it to
[−1, 1]** (`2.0 * (x − 0.5)`, line 483) — but two comments (lines 400, 1967) still
described it as "0 to 1". Writing the labeler against the documented scale gave
buy 2.2% vs sell 13.8%; the asymmetry was the tell.

Evidence it is the stored column, not the formula: recomputing from the same
populated frame gives median **0.5042**, while the stored column has median
**0.0083** and a **min of −1.0** despite its creating line ending in `.clip(0, 1)`.

The rescale is deliberate, so the code was right and the comments were wrong.
Both corrected. **+1 = at/above the upper band, 0 = middle, −1 = at/below the
lower band.** This matters to anyone writing a label or threshold against it, and
`bb_position` is one of the 24 live features.

**Threshold default 0.60** = the upper 20% of the band. Requiring the band itself
(+1.0) is the purer breakout definition, but `bb_position` SATURATES — 5.4% of
bars clip to exactly +1.0 — and after the follow-through filter only ~2.2% of
bars survive, below the ~5.5% floor where `aug_risk` flags all-Hold collapse.
At 0.60 the label is symmetric: 5.78% buy / 5.48% sell on SOL.

## Learnability: dominated by `breakout_gbb`

H=48, mean over XRP/SOL/LINK:

| method | thr | MCC | EV/signal | signals | aug_risk |
|---|---|---|---|---|---|
| gbb | 0.005 | **0.832** | 3.72% | 14544 | LOW/MED |
| breakout_gbb | 0.005 | 0.793 | **4.27%** | 14294 | LOW/MED |
| **breakout_bb** | 0.005 | 0.721 | 4.15% | **7102** | **HIGH/MED** |
| breakout_bb | 0.007 | 0.695 | 4.36% | 6728 | HIGH/MED |
| breakout_bb | 0.010 | 0.641 | 4.70% | 6178 | HIGH |

Per-pair at 0.007: LINK 0.722 / SOL 0.739 / XRP 0.626 — same ordering as the
references, XRP weakest for all three.

**Lower MCC than both references, slightly below `breakout_gbb` on EV, HALF the
signal volume, and HIGH collapse risk.** Nothing on the screen favours it.

## How much that verdict is worth

Very little on its own, and this must be said plainly: **the screen does not
predict edge.** `breakout_gbb` screened BETTER than gbb on EV and backtested 43pp
worse. A label that screens lower is not thereby a worse trade.

What the screen DOES reliably predict in this codebase is trade VOLUME — the
3-class classifier predicts Hold most of the time, and only high-MCC labels break
through. At MCC 0.72 with half the labels, `breakout_bb` would likely trade
thinly, which is the failure mode that made the 2026-07 three-arm A/B
inconclusive.

**Recommendation:** do not spend a retrain on this on screen evidence alone. It
is a sixth variant of an entry family that has now failed five different ways
(threshold, ATR band, take-profit/ROI, exit signal, exit label form). If it is
tested, test it for the reason that it is a genuinely DIFFERENT trigger — a band
crossing rather than an oscillator level — not because the screen ranked it.
