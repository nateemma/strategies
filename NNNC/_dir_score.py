# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study, direction readout (research/tb_model/preregistration_dir.json): a 3-class
model's p_buy rises with "a barrier will be hit" as much as with "up first" (p_buy and p_sell
correlate 0.84 on candidates), so the sizer's agree score is the direction in the softmax,
(1 + p_buy - p_sell) / 2 -- mapped into [0, 1] for prediction_threshold; same ranking as
p_buy - p_sell. A readout only: the weights (and model key) are the arm's own."""

import numpy as np


class DirScoreMixin:
    def sizer_score(self, dataframe):
        p = 0.5 * (1.0 + dataframe["%predict_prob_buy"].to_numpy(float)
                   - dataframe["%predict_prob_sell"].to_numpy(float))
        dataframe["sizer_score"] = p
        return np.nan_to_num(p, nan=0.0)
