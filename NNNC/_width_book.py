"""Model-free WIDTH BOOK entry for an NNNC model class (research/model_calibration, 2026-10-13).

The model check (research/model_calibration/findings.md) found that, within the gbb rule's
candidates, the model's selection behaves like a sort on Bollinger band width, and a simulator
rebuild of a model-free width book matched the model books. This mixin is the real-backtest
version, to confirm that rebuild:

  entry = the gbb rule's eligibility (guard_metric < 0.3 AND bb_width > 0.035)
          AND bb_width >= WIDTH_CUTS[this fold's MODEL_TRAIN_TIMERANGE]
  (then BaseStrategy's own volume / rvol / guard gates, exactly as for the model entry)

Everything else is the model class's own: exits, stop, family tag ("<label family>:model").
The model still loads and predicts (its exit-side columns need it) but does not select entries.
The held-pair add is OFF, so the book compares with the simulator and the 24 h controls.
"""

import numpy as np
from pandas import Series


class WidthBookMixin:
    #: MODEL_TRAIN_TIMERANGE -> bb_width cut (the previous fold's matched cut; f00 its own).
    WIDTH_CUTS: dict = {}
    GBB_GUARD, GBB_BB_WIDTH = 0.3, 0.035
    HELD_PAIR_ADD = None
    MODEL_KEY_IGNORED = ("WIDTH_CUTS", "GBB_GUARD", "GBB_BB_WIDTH")

    def get_entry_conditions(self, dataframe):
        super().get_entry_conditions(dataframe)  # predictions (exit side) + model tags; discarded
        key = getattr(self, "MODEL_TRAIN_TIMERANGE", None)
        if key not in self.WIDTH_CUTS:
            raise KeyError(f"{type(self).__name__}: no WIDTH_CUTS entry for {key!r}")
        gm = dataframe["guard_metric"].to_numpy(float)
        bw = dataframe["bb_width"].to_numpy(float)
        m = (gm < self.GBB_GUARD) & (bw > self.GBB_BB_WIDTH) & (bw >= self.WIDTH_CUTS[key])
        self._st_tags = self._model_tags(m, len(dataframe))
        return Series(m, index=dataframe.index)
