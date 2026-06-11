# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_CGP — multi-task NNMT strategy with MT CTAB-GAN+ augmentation.

Just declares ``gan_type = GANType.MT_CTAB_GAN``; everything else (loading
the GAN, validating its metadata, cross-task balanced augmentation,
denormalising) is dispatched by ``BaseNNStrategy.enhance_training_data``
— see ``GANs.balance.balance_multi_task`` for the actual augmentation.
"""

import sys
from pathlib import Path
from typing import Any

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_MLX import NNMT_MLX  # noqa: E402
from GANs.GANType import GANType  # noqa: E402


class NNMT_CGP_MLX(NNMT_MLX):

    buy_params = { **NNMT_MLX.buy_params,
        "prediction_threshold": 0.45,
        }

    # GAN augmentation — multi-task CTAB-GAN+, applied at the 2-D
    # tabular stage via the BaseNN dispatcher.
    gan_type = GANType.MT_CTAB_GAN
    augment_training_data = True

    gan_run_diagnostics = True
    entry_trend_filter_enable = False
    gan_synth_autoencoder_threshold = 0.005

    # Per-task ratio.  Float broadcasts to every task in train_labels;
    # use a Dict[task, float] or Dict[task, Dict[cls, float]] for finer
    # control — same shape as balance_multi_task accepts.
    gan_target_ratio: Any = 0.3

    batch_size = 2048  # bigger since we augmented the data
