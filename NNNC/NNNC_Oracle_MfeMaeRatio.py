import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_MfeMaeRatio(NNNC_Oracle):
    """ORACLE on mfe_mae_ratio. LOOKAHEAD CEILING -- never promotable.

    Buy where max favourable excursion clears MIN_BUY_GAIN_THRESHOLD AND the
    MFE/MAE ratio clears ``min_ratio`` -- path-aware in spirit to forward_mae
    (both cap by the adverse excursion), but via a RATIO rather than an
    absolute drawdown cap
    (``Framework.TrainingSignals.labels_mfe_mae_ratio``).

    HORIZON and MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD are left at
    NNNC_Oracle's inherited defaults (48 / 0.02) -- the same operating point
    as the incumbent gbb_path label. ``min_ratio`` (default 2.0) has no
    existing strategy-attribute plumbing; it is left at the labeler's own
    default rather than invented.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.mfe_mae_ratio
