import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNClassifierMLX import ClassifierTypeMLX
from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02


class NNNC_A_K5(NNNC_MLX_Lbl02):
    """Architecture study arm — RECEPTIVE FIELD: conv kernel 2 -> 5.

    Inherits NNNC_MLX_Lbl02, so it carries the 0.02 label and is evaluated at
    the configuration where model quality is worth ~25pp (guards off, threshold
    0.80). Everything except the conv kernel is identical to that control
    (Constitution VIII, one parameter space at a time).

    Control for this arm: `r02t08` (NNNC_MLX_Lbl02, stock LSTM), W1 431tr
    +6.94%, W2 763tr +16.91%, W3 295tr +10.25%.
    """

    classifier_type = ClassifierTypeMLX.LSTM_K5
