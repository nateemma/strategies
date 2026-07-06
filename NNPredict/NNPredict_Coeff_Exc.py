# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W0613
# type: ignore
# pylint: disable=import-error

"""
NNPredict_Coeff_Exc — the validated wavelet-coefficient regression strategy.

Predicts the dominant *capturable move* over the next H bars (forward excursion)
from rolling DWT coefficients of the gain series, and trades only high-conviction
signals. This is the config that survived an 8-window walk-forward (train on
prior data, backtest the unseen next ~55 days, model retrained per window,
coeff-PCA fit oldest-first — lookahead-clean):

    OOS window (Apr 2025 -> Jun 2026):  +0.38 +0.53 +0.17 +1.28 +0.70 +0.16
                                        +0.37 +0.55  %
    8/8 windows positive.  mean +0.52%/window,  sum +4.14% over ~14 months,
    134 trades total (low-frequency, high-conviction).

It was reached only by reformulating the TARGET (excursion, not endpoint return)
and cutting turnover — the prediction-quality levers (regularization, PCA,
horizon, feature smoothing, LSTM capacity) all improved ρ/R² while leaving OOS
P&L flat or worse. Judge any change to this strategy on walk-forward P&L, never
on ρ/R².

Caveats: the edge is thin (~3-4%/yr gross), rests on small per-window trade
counts, and is unproven on costs/breadth (one feature family, 11 pairs, 15m,
binanceus fees). Cost + breadth stress-testing is the next gate.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredict_Coeff import NNPredict_Coeff


class NNPredict_Coeff_Exc(NNPredict_Coeff):
    # --- feature / model ---
    HORIZON = 8                    # forward window for the excursion target
    coeff_pca_components = 16      # reduce the wavelet coeff block to 16 whitened PCs
    ridge_alpha = 10.0             # regularize the (still high-dim) input

    # --- target ---
    target_mode = "excursion"      # predict the dominant capturable move, not the endpoint

    # --- signal gate (high-conviction / low-turnover) ---
    entry_z = 2.5                  # only strongly-ranked predictions
    min_magnitude = 0.20           # only large-magnitude (confident) predictions
