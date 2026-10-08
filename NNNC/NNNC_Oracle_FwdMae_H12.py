import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_FwdMae_H12(NNNC_Oracle):
    """ORACLE on forward_mae, horizon 12. LOOKAHEAD CEILING -- never promotable.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    Answers in minutes whether this label can pay at all.
    """

    TRAINING_TYPE = LabelMethod.forward_mae
    HORIZON = 12
    MIN_BUY_GAIN_THRESHOLD = 0.015
    MIN_SELL_LOSS_THRESHOLD = 0.015
    LABEL_MAX_DRAWDOWN = 0.10
