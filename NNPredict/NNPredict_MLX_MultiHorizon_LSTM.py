# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore
# pylint: disable=import-error

"""NNPredict_MLX_MultiHorizon_LSTM — MLX multi-output regressor.

Trains a single LSTM with N output heads (one per horizon in HORIZONS),
combined at signal-time via SIGNAL_COMBINATION rule. Initial config:
HORIZONS=[2, 4, 8], combination="all_agree".
"""

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredictMultiHorizonStrategy import NNPredictMultiHorizonStrategy
import NNPredictRegressorMLX


class NNPredict_MLX_MultiHorizon_LSTM(NNPredictMultiHorizonStrategy):

    SIGNAL_COMBINATION: str = "any_strong"

    def get_classifier_type(self):
        return NNPredictRegressorMLX.RegressorTypeMLX.MULTI_HORIZON_LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        if not (hasattr(mx, "metal") and mx.metal.is_available()):
            print(
                "ERROR: This strategy requires Apple's MLX package, and only "
                "runs on native Apple hardware"
            )
            return None
        reg, _ = NNPredictRegressorMLX.create_regressor_mlx(
            classifier_type, pair, num_features, seq_len
        )
        reg.max_epochs = 300
        # Tell the regressor how many heads to output.
        reg.num_horizons = len(self.HORIZONS)
        return reg
