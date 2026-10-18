# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study, direction-only training (research/tb_model/preregistration_dironly.json).

The 3-class tbatr model spends its capacity on HOLD (timeout) vs a barrier hit -- volatility -- and
its p_buy tracks that (report.md diagnostics). This mixin keeps only windows whose last bar's label
is BUY or SELL (`%tb_dir_keep`, written beside %train_buy / %train_sell), on top of the class's own
row filter, train and validation alike; HOLD never appears, so p_buy is P(up first | a barrier is hit).
The filter spec is a model-key attribute and get_training_labels a key method, so the weights are new.
"""

import numpy as np


class DirOnlyMixin:
    DIR_KEEP_COL = "%tb_dir_keep"

    def get_training_labels(self, dataframe):
        labels = super().get_training_labels(dataframe)
        hit = (dataframe["%train_buy"].to_numpy() > 0.5) | (dataframe["%train_sell"].to_numpy() > 0.5)
        dataframe[self.DIR_KEEP_COL] = np.where(hit, 1.0, np.nan)
        return labels
