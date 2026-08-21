# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_DDPM_MLX - Subclass of NNMTStrategy using DDPM and MLX models
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_DDPM import NNMT_DDPM  # noqa: E402
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, MLXMultiTaskClassifierMixin

# -----------


class NNMT_DDPM_MLX(MLXMultiTaskClassifierMixin, NNMT_DDPM):

    # MLX LSTM (default from MLXMultiTaskClassifierMixin).
    classifier_type = ClassifierTypeMLX.LSTM

    # buy_params = { **NNMT_DDPM.buy_params,
    #     "prediction_threshold": 0.5
    #     }

    gan_target_ratio = 0.3
    gan_run_diagnostics = True
    entry_trend_filter_enable = False
    gan_synth_autoencoder_threshold = 0.005
