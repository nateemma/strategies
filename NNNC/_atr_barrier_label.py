"""ATR-scaled triple-barrier label for an NNNC model (research/tb_model/): 037's `tbatr`.

Three classes on the framework's own labelers: BUY = the close reaches +k x ATR(14)/close before
-k x ATR(14)/close within HORIZON bars, SELL = the mirror, HOLD = timeout or same-bar tie. k = 4.0
puts the median barrier at 2% on f00's training span (037 tb.py). No %-floor: min_gain / min_loss are
overridden to None and the post-label peak filter (which keeps only peaks whose future move reaches
MIN_BUY_GAIN_THRESHOLD, a +-2% floor) is OFF -- both would make the label favour volatile bars again.
LABEL_PEAK_FILTER is a model-key attribute. Entry family pinned to reversion, as 034's label arms.
"""

import numpy as np
import pandas as pd

from LabelParamsMixin import LabelParamsMixin


ATR_K = 4.0


class AtrBarrierLabelMixin(LabelParamsMixin):
    TRAINING_TYPE = 1                                   # LabelMethod.triple_barrier
    LABEL_PARAMS = {"pt": 1e-9, "sl": 1e-9, "atr_scale": ATR_K, "min_gain": None}
    LABEL_PARAMS_SELL = {"pt": 1e-9, "sl": 1e-9, "atr_scale": ATR_K, "min_loss": None}
    LABEL_PEAK_FILTER = False
    MODEL_ENTRY_FAMILY = "reversion"
    MODEL_KEY_ATTRIBUTES = ("LABEL_PEAK_FILTER",)

    def _filtered(self, future_df, labels, side: str) -> pd.Series:
        if self.LABEL_PEAK_FILTER:
            return super()._filtered(future_df, labels, side)
        return pd.Series((np.asarray(labels, dtype=float) > 0.5).astype(float))
