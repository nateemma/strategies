# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""NNPredict_MLX_LSTM — MLX (Apple Silicon) LSTM regressor."""

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressorMLX


class NNPredict_MLX_LSTM(NNPredictStrategy):
    # ModelType stays KERAS (inherited): there is no ModelType.MLX in the
    # enum today. The MLX-vs-Keras dispatch happens via get_classifier_type()
    # returning a RegressorTypeMLX value, the same way NNNC_CGP_MLX_LSTM does
    # it for the classifier family.

    def get_classifier_type(self):
        return NNPredictRegressorMLX.RegressorTypeMLX.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        if not (hasattr(mx, "metal") and mx.metal.is_available()):
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            return None
        reg, _ = NNPredictRegressorMLX.create_regressor_mlx(
            classifier_type, pair, num_features, seq_len
        )
        # Last training run had best_epoch=100 (the prior cap) with ρ still
        # climbing and val_loss still decreasing — model was still learning
        # when training was cut off. Allow a longer leash here.
        reg.max_epochs = 300
        # Pass HORIZON through so the trainer can compute the "predict-current-
        # state" shortcut diagnostic in the per-epoch line.
        reg.horizon = int(self.HORIZON)
        return reg
