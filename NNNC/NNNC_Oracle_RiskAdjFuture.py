import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_RiskAdjFuture(NNNC_Oracle):
    """ORACLE on risk_adj_future. LOOKAHEAD CEILING -- never promotable.

    Buy where the future-window Sharpe proxy clears ``min_sharpe`` AND the
    max adverse excursion stays under a drawdown cap -- path-aware in spirit
    to forward_mae (``Framework.TrainingSignals.labels_risk_adjusted_future_return``).

    HORIZON and MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD are left at
    NNNC_Oracle's inherited defaults (48 / 0.02) -- the same operating point
    as the incumbent gbb_path label. ``min_sharpe`` (default 0.5) and
    ``window_stats`` (default 16) have no existing strategy-attribute
    plumbing and are left at the labeler's own defaults. The labeler DOES
    accept ``max_drawdown`` (default 0.1) via the existing LABEL_MAX_DRAWDOWN
    opt-in hook; left unset here so the labeler's own default (0.1) applies,
    matching the rationale this arm is screening rather than a chosen value.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.risk_adj_future
