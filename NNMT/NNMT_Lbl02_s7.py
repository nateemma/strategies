# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Lbl02_s7 — inherited label 0.02 with the profit head pinned, seed 7

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


class NNMT_Lbl02_s7(NNMT_MLX_Base):

    # The INHERITED label under test.
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02

    # The profit head's class boundaries are ALSO derived from these two
    # attributes (NNMTStrategy ~438 and ~510, at PROFIT_HORIZON rather than
    # HORIZON), so raising the label would move the buy/sell labels AND an
    # auxiliary head's targets at once. Pinning the profit head to the old value
    # keeps this arm single-variable (research R2, Constitution VIII).
    #
    # T013 established the pin is NOT load-bearing in practice: the profit head
    # carries loss weight 0.0, so it is untrained and the coupling is cosmetic.
    # It is kept because it costs nothing, it makes the arm single-variable by
    # construction rather than by an argument about loss weights, and it stays
    # correct if that head is ever re-enabled.
    PROFIT_HEAD_GAIN_THRESHOLD = 0.007

    def get_profit_target(self, dataframe):
        """Build the PROFIT-head label at the pinned pre-inheritance threshold.

        Both threshold reads sit inside get_profit_target (the peak variant is
        reachable only through it), so restoring the attributes around one call
        pins both sites.
        """
        saved = (self.MIN_BUY_GAIN_THRESHOLD, self.MIN_SELL_LOSS_THRESHOLD)
        pin = self.PROFIT_HEAD_GAIN_THRESHOLD
        self.MIN_BUY_GAIN_THRESHOLD = pin
        self.MIN_SELL_LOSS_THRESHOLD = pin
        try:
            return super().get_profit_target(dataframe)
        finally:
            self.MIN_BUY_GAIN_THRESHOLD, self.MIN_SELL_LOSS_THRESHOLD = saved

    def get_model_path(self) -> str:
        """Own weights, NOT NNMT_MLX's — this arm trains its own model."""
        name = self.__class__.__name__
        return self.get_storage_location() + name + "/" + name + ".keras"
