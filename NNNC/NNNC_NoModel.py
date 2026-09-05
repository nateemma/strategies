import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame, Series

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_NoModel(NNNC_MLX):
    """Guard chain ALONE — the model's entry opinion is bypassed entirely.

    Lowering `prediction_threshold` does NOT do this. `argmax_with_threshold`
    returns the argmax class whenever max_prob > threshold, so at threshold 0.0
    the model still decides: bars whose argmax is HOLD are still rejected. That
    measures the CONFIDENCE gate, not the model.

    This subclass overrides `get_entry_conditions` to accept every bar, so the
    entry set is determined by the guard chain and nothing else. The difference
    against NNNC_MLX is the model's entire contribution to entry selection.

    The model is still loaded and the exit side is untouched, so only the ENTRY
    contribution is isolated.
    """

    def get_entry_conditions(self, dataframe: DataFrame):
        # Call the parent first: it runs process_predictions, which is what
        # creates the predict_buy / predict_sell columns. The exit side and the
        # diagnostics block both need them, so skipping it raises
        # KeyError: 'predict_buy'. We keep the columns and discard the opinion.
        super().get_entry_conditions(dataframe)
        return Series(np.ones(len(dataframe), dtype=bool), index=dataframe.index)

    def get_model_path(self) -> str:
        """Reuse NNNC_MLX's weights — nothing here is retrained."""
        root_dir = self.get_storage_location()
        return root_dir + "NNNC_MLX/" + "NNNC_MLX.keras"
