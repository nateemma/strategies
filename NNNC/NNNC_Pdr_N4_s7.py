import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNClassifierMLX import ClassifierTypeMLX
from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02
from PonderStrategyMixin import PonderStrategyMixin


class NNNC_Pdr_N4_s7(PonderStrategyMixin, NNNC_MLX_Lbl02):
    """COCONUT stage-2 PONDER at the new operating point — 4 refinement steps.

    Subclasses NNNC_MLX_Lbl02 (label 0.02) rather than NNNC_MLX_Open, because
    Open ALIASES Lbl02's weights via get_model_path and a ponder model has new
    parameters that must be trained. The guards-open / threshold-0.80 operating
    point is pinned in NNNC_Pdr_N4_s7.json instead.

    ponder_steps=0 is the MATCHED CONTROL: the loop is skipped so the forward
    pass is identical to the production _LSTMModel, but the code path, parameter
    tree and init RNG are the same as the N>0 arms.

    WHY RE-RUN. The ponder edge (+0.9pp, 4/4 seeds) was RETRACTED as a DDPM-base
    artefact after the predict-time scaling bug was fixed, and was never re-run
    on the non-GAN base. It also could not have been measured at the old
    operating point, where a PERFECT model was worth 0.68pp. Trained at
    MLX_SEED=7.
    """

    classifier_type = ClassifierTypeMLX.LSTM_PONDER
    ponder_steps = 4
