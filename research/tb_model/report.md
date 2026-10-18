# ATR-barrier study: S1 screen (2026-10-17)

Pre-registration `preregistration.json` (69d250ae). Arms: cs_tba (ARO on 037 tbatr, volume > 0, 24
inputs), cs_tbb (+ 7 composite inputs). S1 pass = candidate-bar tbatr AUC above ARO's in both halves.

| Median over folds | composite | ARO | cs_tba | cs_tbb |
|---|---|---|---|---|
| candidate AUC early | 0.554 | 0.541 | 0.503 | 0.492 |
| candidate AUC late | 0.552 | 0.539 | 0.523 | 0.525 |
| candidate MCC early | — | 0.007 | −0.003 | 0.006 |
| candidate MCC late | — | 0.004 | 0.057 | 0.064 |
| all-bar AUC early / late | — | 0.518 / 0.497 | 0.497 / 0.505 | 0.493 / 0.510 |

**S1: both arms FAIL; no backtests run.** B > A in both halves: no.

Diagnostic (not part of the gate): both models output near-constant p_buy (mean ≈ 0.36, sd ≈ 0.035,
max ≈ 0.44, e.g. f02 and f13), so they sit near the base rate rather than learning the label. cs_tbb
has the composite's inputs yet scores below the composite's fixed equal-weight readout (0.49 vs 0.55
early), which points at training, not at the inputs. Late folds f12/f13 do reach 0.55–0.61.

## Diagnostics after S1 (2026-10-17; not gates)

**1. What the trained model learned.** Training is 3-class (sell 43% / hold 20% / buy 37% on f02);
val MCC sits at ≈ 0.14 from epoch 1, then early-stops at epoch 26 after two LR cuts. The test windows
show where that MCC comes from:

| CS_TBA, median over folds | early | late |
|---|---|---|
| p_buy, barrier hit vs timeout (all bars) | ≈ 0.70 | ≈ 0.71 |
| p_buy, up vs down (candidates; the S1 score) | 0.503 | 0.523 |
| **p_buy − p_sell**, up vs down (candidates) | **0.542** | **0.563** |
| p_buy − p_sell, up vs down (all bars) | 0.527 | 0.527 |
| corr(p_buy, p_sell) on candidates | 0.84 | 0.84 |

p_buy and p_sell both rise with "a barrier will be hit" (volatility); the direction is in their
difference. S1 scored the wrong readout for a 3-class model. (Full softmax regenerated outside the
repo; `research/nnnc_framework/signals.py` now also writes p_sell / p_hold.)

**2. What is learnable from the same inputs** (`reference.py`, `reference.json`; last bar only,
hourly training rows to 48 bars before each test window):

| Median AUC, candidates | early | late |
|---|---|---|
| direction, A inputs (24): logistic / HGB | 0.552 / 0.538 | 0.560 / 0.555 |
| direction, B inputs (31): logistic / HGB | 0.545 / 0.537 | 0.570 / 0.558 |
| composite (fixed, no fit) | 0.554 | 0.552 |
| trained CS_TBA, p_buy − p_sell | 0.542 | 0.563 |
| hit vs timeout, A / B HGB | 0.679 / 0.692 | 0.670 / 0.684 |

Direction tops out near 0.55–0.57 for every model on these inputs; the trained LSTM, read as
p_buy − p_sell, is at that level. The composite's members add ≤ 0.01 (B ≈ A). Barrier-hit (volatility)
is far more learnable (≈ 0.68 on candidates, 0.73 on all bars).
