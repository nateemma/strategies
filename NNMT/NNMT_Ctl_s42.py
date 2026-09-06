# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Ctl_s42 — pre-inheritance label 0.007, seed 42

Phase 4 of feature 005 compares the PRE-INHERITANCE label (0.007) with the
INHERITED one (0.02) on NNMT. FR-020 requires at least two training seeds before
any acceptance or rejection, and `train_seed` is INERT for multi-task models
(R8), so the seed is supplied by the MLX_SEED environment variable at training
time and distinguished here only by class name.

Why a class per (label, seed): BaseNNStrategy keys the model directory off
`self.__class__.__name__`. These arms MUST train their own models, so unlike
NNMT_MLX_Base / NNMT_Oracle / NNMT_NoModel they deliberately do NOT alias
NNMT_MLX's weights — `get_model_path` is restored to the class-name-keyed
default below. Training any of them therefore writes to its own directory and
CANNOT touch the production NNMT_MLX weights.

Its .json pins the control configuration, including cexit_take_profit 0.032, so
the label is the only thing that varies against the control (Constitution VIII).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX_Base import NNMT_MLX_Base


class NNMT_Ctl_s42(NNMT_MLX_Base):

    # Pre-inheritance label, inherited from NNMT_MLX_Base (0.007). Restated
    # here so the arm is readable on its own.
    MIN_BUY_GAIN_THRESHOLD = 0.007
    MIN_SELL_LOSS_THRESHOLD = 0.007

    def get_model_path(self) -> str:
        """Own weights, NOT NNMT_MLX's — this arm trains its own model."""
        name = self.__class__.__name__
        return self.get_storage_location() + name + "/" + name + ".keras"
