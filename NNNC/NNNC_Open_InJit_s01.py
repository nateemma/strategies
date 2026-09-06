import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNClassifierMLX import ClassifierTypeMLX
from NNNC_MLX_Open import NNNC_MLX_Open
from NoisyCoconutStrategyMixin import NoisyCoconutStrategyMixin


class NNNC_Open_InJit_s01(NoisyCoconutStrategyMixin, NNNC_MLX_Open):
    """COCONUT re-run at the NEW operating point — INPUT-jitter voting, sigma 0.1

    TRAINING-FREE: reuses NNNC_MLX_Lbl02's weights, so this is a backtest only.

    WHY RE-RUN. Noise-voting was REJECTED in 2026-08 for monotonic harm, but
    that was measured at the OLD operating point where the guard chain selected
    ~99% of entries and a PERFECT model was worth 0.68pp — nothing could have
    registered. With the guards open there is ~25pp of headroom and 4.5x the
    trades, so a test-time intervention can finally be seen.

    Control is `open` (NNNC_MLX_Open, sigma 0): W1 431tr +6.94%,
    W2 763tr +16.91%, W3 295tr +10.25%.
    """

    classifier_type = ClassifierTypeMLX.LSTM_INJIT
    reuse_model_from = "NNNC_MLX_Lbl02"
    noisy_sigma = 0.1
    noisy_k = 16
