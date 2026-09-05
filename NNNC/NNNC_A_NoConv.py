import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNClassifierMLX import ClassifierTypeMLX
from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02


class NNNC_A_NoConv(NNNC_MLX_Lbl02):
    """Architecture study — STRAIGHT LSTM: resize -> LSTM. Conv1d + residual removed.

    Control: `r02t08` (NNNC_MLX_Lbl02, full stack), W1 431tr +6.94%,
    W2 763tr +16.91%, W3 295tr +10.25%, guards off, threshold 0.80.
    """

    classifier_type = ClassifierTypeMLX.LSTM_NOCONV
