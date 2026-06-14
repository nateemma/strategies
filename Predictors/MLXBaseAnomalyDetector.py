from __future__ import annotations

# Base class for MLX-based GANomaly anomaly-detector classifiers.
#
# Combines MLXBasePredictor (model lifecycle: save/load/path) with the
# BaseAnomalyDetector marker, and exposes the predict() contract used by
# the anomaly strategies:
#     predict(data) -> {"reconstruction": ndarray (N, S, F),
#                       "trading":        ndarray (N, 3)}
#
# Concrete subclasses (the 4 GANomaly variants) live in
# Anomaly/NNGANomalyClassifierMLX.py and supply the encoder / decoder /
# discriminator bodies plus the 3-phase train() implementation.
import logging

import mlx.core as mx
import numpy as np
from Predictors.BaseAnomalyDetector import BaseAnomalyDetector
from Predictors.MLXBasePredictor import MLXBasePredictor


log = logging.getLogger(__name__)


class MLXBaseAnomalyDetector(MLXBasePredictor, BaseAnomalyDetector):
    """
    Base for MLX anomaly detectors with a (reconstruction, trading) output.

    Carries no architecture itself — subclasses override create_model() to
    return a unified nn.Module whose __call__(x) yields the tuple
    (reconstruction, trading), and override train() for the 3-phase flow.
    """

    clean_data_required = True

    def predict(self, data):
        # Lazy-load (params can change up to this point).
        if self.model is None:
            self.model = self.load()

        if self.model is None:
            log.error("    ERR: no model for predictions")
            n = np.shape(data)[0]
            return {
                "reconstruction": np.zeros((n, self.seq_len, self.num_features), dtype=np.float32),
                "trading": np.zeros((n, 3), dtype=np.float32),
            }

        if self.dataframeUtils.is_dataframe(data):
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            df_tensor = data

        x = mx.array(np.asarray(df_tensor, dtype=np.float32))

        self.model.eval()
        rec, trd = self.model(x)
        mx.eval(rec, trd)

        return {
            "reconstruction": np.array(rec, dtype=np.float32),
            "trading": np.array(trd, dtype=np.float32),
        }
