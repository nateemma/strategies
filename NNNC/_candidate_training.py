"""Candidate-only training for an NNNC model class (feature 032, contract C3).

The model check (research/model_calibration/findings.md) found that most of an NNNC model's AUC is
the gbb rule itself: within the rule's candidates it ranks at only 0.53-0.62. This mixin trains the
model ONLY on windows whose last bar passes the rule, so its capacity goes to ranking candidates:

  * TRAIN_ROW_FILTER -- the gbb rule as data, applied by Framework/TrainingEngine after windowing
    (train and validation), and part of the 030 model key for classes with this mixin only;
  * entry -- the parent's entry mask AND the same rule, so the model only ranks bars of the kind
    it was trained on (and pick_threshold scores ineligible bars as 0);
  * HELD_PAIR_ADD off, matching the add-off 24 h controls and width books it is compared with.
"""

import numpy as np
from pandas import Series

from Framework.RowFilter import row_filter_mask


class CandidateTrainingMixin:
    TRAIN_ROW_FILTER = (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))
    MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)
    HELD_PAIR_ADD = None

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        if mask is None:
            return None
        m = np.asarray(mask, dtype=bool) & row_filter_mask(dataframe, self.TRAIN_ROW_FILTER)
        self._st_tags = self._model_tags(m, len(dataframe))
        return Series(m, index=dataframe.index)
