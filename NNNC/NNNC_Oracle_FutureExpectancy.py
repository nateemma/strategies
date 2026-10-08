import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_FutureExpectancy(NNNC_Oracle):
    """ORACLE on future_expectancy. LOOKAHEAD CEILING -- never promotable.

    Buy where the future-window expectancy (win_rate * avg_win - loss_rate *
    avg_loss) for going long beats the equivalent short expectancy
    (``Framework.TrainingSignals.labels_future_expectancy``).

    HORIZON and MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD are left at
    NNNC_Oracle's inherited defaults (48 / 0.02) -- the same operating point
    as the incumbent gbb_path label -- so the only variable under test is the
    return DEFINITION, not the horizon or threshold. ``min_expectancy``
    (default 0.01) has no existing strategy-attribute plumbing; it is left at
    the labeler's own default rather than invented.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.future_expectancy
