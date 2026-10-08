import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_LocalMinFollowthrough(NNNC_Oracle):
    """ORACLE on local_min_followthrough. LOOKAHEAD CEILING -- never promotable.

    Buy only at a local price minimum (within a +/-``window_k`` bar window)
    that is followed by a forward gain clearing MIN_BUY_GAIN_THRESHOLD while
    the max adverse excursion stays under a drawdown cap
    (``Framework.TrainingSignals.labels_local_min_followthrough``).

    HORIZON and MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD are left at
    NNNC_Oracle's inherited defaults (48 / 0.02) -- the same operating point
    as the incumbent gbb_path label. ``window_k`` (default 3) has no existing
    strategy-attribute plumbing and is left at the labeler's own default.
    The labeler DOES accept ``max_drawdown`` (default 0.1) via the existing
    LABEL_MAX_DRAWDOWN opt-in hook; left unset here so the labeler's own
    default (0.1) applies.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.local_min_followthrough
