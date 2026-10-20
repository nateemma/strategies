"""039 threshold amendment, option 1: the flip arm's gate sees the negated prediction, and
%predict_gain keeps the model's own sign."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNPredict"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _flip import FlipSignMixin  # noqa: E402


class _Model:
    def predict(self, x):
        return np.array([0.3, -0.2, 0.0], dtype=np.float32)


class _Gate:  # stands in for NNPredictStrategy.get_predictions: records what the gate saw
    def get_predictions(self, dataframe, classifier):
        g = np.asarray(classifier.predict(None))
        dataframe["%predict_gain"] = g
        self.seen = g
        return np.sign(g)


class _Flip(FlipSignMixin, _Gate):
    pass


def test_gate_sees_negated_prediction_and_column_keeps_model_sign():
    s, df = _Flip(), pd.DataFrame(index=range(3))
    actions = s.get_predictions(df, _Model())
    assert np.allclose(s.seen, [-0.3, 0.2, 0.0])
    assert np.allclose(actions, [-1, 1, 0])
    assert np.allclose(df["%predict_gain"], [0.3, -0.2, 0.0])
