import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_GbbFwdMae_H48(NNNC_Oracle):
    """ORACLE on gbb_fwdmae (the HYBRID label), horizon 48. LOOKAHEAD CEILING --
    never promotable.

    Tests the hypothesis that gbb's edge is its regime TRIGGER, not its return
    definition: this label keeps gbb's trigger (``guard_metric`` /
    ``bb_width`` / ``rvol``) but replaces gbb's bare forward-gain floor with
    forward_mae's drawdown-capped return test (see
    ``Framework.TrainingSignals.labels_gbb_fwdmae``).

    Same HORIZON/threshold/drawdown operating point as
    ``NNNC_Oracle_FwdMae_H48`` -- the only variable under test against that
    arm is whether adding gbb's trigger recovers the ~0%, never-trades result
    forward_mae produced alone.

    Applies the standing oracle gate (regime/ORACLE_GATE.md) to a label instead of
    training a model on it: PASS = W1 guards-off return > 0 AND n_trades >= 200.
    """

    TRAINING_TYPE = LabelMethod.gbb_fwdmae
    HORIZON = 48
    MIN_BUY_GAIN_THRESHOLD = 0.015
    MIN_SELL_LOSS_THRESHOLD = 0.015
    LABEL_MAX_DRAWDOWN = 0.10
