import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02
from TriggerOnlyMixin import TriggerOnlyMixin


class NNNC_TrigGate(TriggerOnlyMixin, NNNC_MLX_Lbl02):
    """CONTROL for the trigger study: the trigger applied at INFERENCE ONLY.

    Reuses NNNC_MLX_Lbl02's existing weights — NO retraining — so the only
    difference from the control arm is that entries must also satisfy the
    trigger. This separates the inference-side gate from the training-side
    reweighting, which `NNNC_TrigOnly` adds on top (Constitution VIII).

    `pnl_loss_alpha = None` disables the loss masking; this arm never trains.
    """

    pnl_loss_alpha = None

    def get_model_path(self) -> str:
        return self.get_storage_location() + "NNNC_MLX_Lbl02/" + "NNNC_MLX_Lbl02.keras"
