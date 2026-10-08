import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from LabelParamsMixin import LabelParamsMixin
from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_TripleBarrier_H12(LabelParamsMixin, NNNC_Oracle):
    """ORACLE on triple_barrier, horizon 12, pt/sl = 1.5%. LOOKAHEAD CEILING --
    never promotable.

    The study's other (never P&L-tested) learnability winner. Buy only if the
    +1.5% profit-take barrier is touched strictly before the -1.5% stop-loss
    barrier within 12 bars (``Framework.TrainingSignals.labels_triple_barrier``).

    PLUMBING NOTE: ``labels_triple_barrier``'s ``pt``/``sl`` default to
    0.03/0.02 and are only ever RAISED via ``max(pt, min_gain)`` /
    ``max(sl, min_loss)`` -- they can never be LOWERED through
    MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD alone. Reaching 1.5%
    therefore needs BOTH: LABEL_PARAMS to set pt/sl directly (via the
    existing ``NNNC.LabelParamsMixin`` extension point -- no Framework/
    change), AND lowering MIN_BUY_GAIN_THRESHOLD/MIN_SELL_LOSS_THRESHOLD to
    1.5% too, or the floor re-raises pt/sl straight back to the 0.02 defaults
    inherited from NNNC_Oracle.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.triple_barrier
    HORIZON = 12
    MIN_BUY_GAIN_THRESHOLD = 0.015
    MIN_SELL_LOSS_THRESHOLD = 0.015
    LABEL_PARAMS = {"pt": 0.015, "sl": 0.015}
    LABEL_PARAMS_SELL = None  # mirror LABEL_PARAMS on the sell side
