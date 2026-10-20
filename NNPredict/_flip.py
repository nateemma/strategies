# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 threshold amendment, option 1 (research/nnpredict/preregistration_threshold.json): trade the
model's prediction with its sign flipped.

The decile check (research/nnpredict/deciles_nnp_ridge.json) found Ridge's predicted gain ranks
outcomes INVERTED on the early folds (Spearman(decile, return) -0.73 at H=4): the lowest predictions
did best. The gate sees the negated prediction, so BUY fires on the most negative predictions and
SELL on the most positive; `%predict_gain` keeps the model's own sign. Post-model only: the regressor's
weights and the model key are unchanged.
"""

import numpy as np


class _Negated:
    def __init__(self, inner):
        self._inner = inner

    def predict(self, x, *args, **kwargs):
        return -np.asarray(self._inner.predict(x, *args, **kwargs), dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class FlipSignMixin:
    def get_predictions(self, dataframe, classifier):
        out = super().get_predictions(dataframe, _Negated(classifier))
        dataframe["%predict_gain"] = -dataframe["%predict_gain"]
        return out
