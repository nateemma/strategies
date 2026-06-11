# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_WGAN_MLX - Subclass of NNMTStrategy using WGAN and MLX models
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_WGAN import NNMT_WGAN  # noqa: E402
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, MLXMultiTaskClassifierMixin

# -----------


class NNMT_WGAN_MLX(MLXMultiTaskClassifierMixin, NNMT_WGAN):

    # MLX LSTM (default from MLXMultiTaskClassifierMixin).
    classifier_type = ClassifierTypeMLX.LSTM

    # Scalar broadcasts to every task — see balance_multi_task.target_ratios.
    gan_target_ratio = 0.5
    gan_run_diagnostics = True

    use_post_gan_scaling = True

    # Per-class autoencoder filter — trading-head only (Option B).
    # Same setting as NNMT_DDPM_MLX / NNNC_DDPM_MLX.
    gan_synth_autoencoder_threshold = 0.005
