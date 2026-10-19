# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 (research R4): NNPredict with NNNC's plain LSTM (LSTM_RAW: features -> LSTM -> dense head) as a
regressor, on the RAW input set (raw8 + NNPredict's current_gain / recent_gain). Target, horizon, gate
and exits are NNPredictStrategy's, unchanged."""

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))

from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressorMLX
from _inputs import InputSetMixin


class NNPredict_LSTM_Raw(InputSetMixin, NNPredictStrategy):
    INPUT_SET = "raw"

    def get_classifier_type(self):
        return NNPredictRegressorMLX.RegressorTypeMLX.LSTM_RAW

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> Any:
        if not (hasattr(mx, "metal") and mx.metal.is_available()):
            print("ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware")
            return None
        reg, _ = NNPredictRegressorMLX.create_regressor_mlx(classifier_type, pair, num_features, seq_len)
        reg.max_epochs = 300  # as NNPredict_MLX_LSTM; early stopping ends training sooner
        reg.horizon = int(self.HORIZON)
        return reg
