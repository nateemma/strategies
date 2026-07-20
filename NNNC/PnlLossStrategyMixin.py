# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
PnlLossStrategyMixin — P&L-magnitude-weighted training loss probe.

Adds a per-row forward-excursion magnitude as a "%pnl_weight" dataframe column
(auto-sliced with rows, auto-dropped before tensorization so it never leaks as a
feature), and stamps the blend alpha onto the classifier. The classifier's
train() blends it into the focal loss: w = (1-alpha) + alpha*normalised(mag),
so alpha=0 is identical to the standard loss (control).

Mix in *before* the NNNC base (NNNC_MLX for the non-GAN probe):

    class NNNC_MLX_PnlLoss(PnlLossStrategyMixin, NNNC_MLX):
        pnl_loss_alpha = 0.5

Gate on the learnable gbb signal first (not triple-barrier). See
docs/superpowers/specs/2026-07-19-pnl-weighted-loss-probe-design.md.
"""

import numpy as np
from Framework.TrainingSignals import forward_excursion


class PnlLossStrategyMixin:
    # None = off (standard loss). Float in (0, 1] enables P&L weighting.
    pnl_loss_alpha = None
    # Optional training-seed override for robustness sweeps (None = default).
    train_seed = None

    def get_training_labels(self, dataframe):
        labels = super().get_training_labels(dataframe)
        if self.pnl_loss_alpha is not None:
            buy_mfe, sell_mfe = forward_excursion(dataframe, self.HORIZON)
            is_buy = dataframe["%train_buy"].values > 0.5
            is_sell = dataframe["%train_sell"].values > 0.5
            w = np.full(len(dataframe), np.nan, dtype=np.float64)
            w[is_buy] = buy_mfe[is_buy]
            w[is_sell] = sell_mfe[is_sell]
            dataframe["%pnl_weight"] = w
        return labels

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.pnl_loss_alpha = self.pnl_loss_alpha
            clf.train_seed = self.train_seed
        return clf
