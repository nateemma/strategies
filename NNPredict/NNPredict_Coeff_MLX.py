# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W0613
# type: ignore
# pylint: disable=import-error

"""
NNPredict_Coeff_MLX — NNPredict_Coeff with an MLX (Apple Silicon) LSTM regressor
instead of the Ridge baseline.

Same wavelet-coefficient feature set as NNPredict_Coeff (inherited); only the
predictor changes. This is the head-to-head that matters: does a nonlinear
sequence model beat the linear floor on the rich (~100-dim) coefficient
representation? If LSTM <= Ridge here too, "simpler wins" holds even on rich
features; if it wins, this is where capacity pays.
"""

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredict_Coeff import NNPredict_Coeff
import NNPredictRegressorMLX


class NNPredict_Coeff_MLX(NNPredict_Coeff):

    def get_classifier_type(self):
        return NNPredictRegressorMLX.RegressorTypeMLX.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> Any:
        if not (hasattr(mx, "metal") and mx.metal.is_available()):
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs "
                "on native Apple hardware"
            )
            return None
        reg, _ = NNPredictRegressorMLX.create_regressor_mlx(
            classifier_type, pair, num_features, seq_len
        )
        reg.max_epochs = 300
        reg.horizon = int(self.HORIZON)
        return reg
