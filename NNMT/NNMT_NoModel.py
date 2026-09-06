# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_NoModel — the guard chain ALONE, with the model's entry opinion bypassed.

The difference between this and the control is the model's ENTIRE contribution
to entry selection. Together with NNMT_Oracle it bounds that contribution from
both sides: no model / real model / perfect model.

Lowering ``prediction_threshold`` does NOT do this. ``argmax_with_threshold``
returns the argmax class whenever max_prob exceeds the threshold, so even at 0.0
the model still decides — bars whose argmax is HOLD are still rejected. That
measures the CONFIDENCE gate, not the model.

The parent's ``get_entry_conditions`` MUST be called first: it runs
``process_predictions``, which is what creates the predict_buy / predict_sell
columns. The exit side and the diagnostics block both read them, so skipping it
raises KeyError: 'predict_buy'. We keep the columns and discard the opinion.

The model is still loaded and the exit side is untouched, so only the ENTRY
contribution is isolated. ``get_model_path`` is inherited from NNMT_MLX_Base and
aliases NNMT_MLX's weights, so this arm trains nothing.
"""

import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame, Series

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX_Base import NNMT_MLX_Base


class NNMT_NoModel(NNMT_MLX_Base):

    def get_entry_conditions(self, dataframe: DataFrame):
        super().get_entry_conditions(dataframe)
        return Series(np.ones(len(dataframe), dtype=bool), index=dataframe.index)
