# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore
# pylint: disable=import-error

"""
NNMT_DDPM_MLX_MultiLSTM - Subclass of NNMT_DDPM using MLX MultiLSTM classifier.

Case A applies: NNMT_WGAN hardcodes GANType.MT_WGAN in both the class attribute
and the GANInterface constructor call in preprocess_training_data.  A simple
attribute override is therefore insufficient; NNMT_DDPM.py is the sibling base
that replaces both references with GANType.MT_DDPM.

The MLX classifier factory is provided by MLXMultiTaskClassifierMixin.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_DDPM import NNMT_DDPM
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, MLXMultiTaskClassifierMixin


class NNMT_DDPM_MLX_MultiLSTM(MLXMultiTaskClassifierMixin, NNMT_DDPM):

    classifier_type = ClassifierTypeMLX.Multi_LSTM

    gan_target_ratio = 0.4

    # Per-class autoencoder filter — trading-head only (Option B).
    # Same setting as NNMT_DDPM_MLX / NNNC_DDPM_MLX.
    gan_synth_autoencoder_threshold = 0.005
