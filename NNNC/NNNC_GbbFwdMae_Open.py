import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Framework.TrainingSignals import LabelMethod
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_GbbFwdMae_Open(NNNC_A_Raw_Open):
    """HYBRID label arm: gbb's regime trigger AND forward_mae's capped return test.

    One-variable change from the control `NNNC_A_Raw_Open` -- same LSTM_RAW, same
    24-indicator set, same seq_len 16, same guards-open gate, same exit. Only the
    label's FORM differs.

    Why this arm exists. On 15-fold pooled P&L, gbb returns +39.98pp and forward_mae
    -0.83pp, and the three obvious explanations are all eliminated: the gate (both
    guards-open), the features (raw8 vs ind24 was a null, -1.23pp/yr CI spanning
    zero), and the exit (swapping it back moved 0.6pp; a take-profit ladder from 1.5%
    to 5% left the control UNCHANGED, CI spanning zero). The oracle gate cannot
    separate them either -- all 11 labels screened PASS, and gbb ranks second from
    LAST on ceiling (+2,524% vs forward_mae's +6,779%).

    So the distinguishing feature is not ceiling, hit rate, learnability or payoff
    geometry. The one structural thing gbb has that forward_mae lacks is its REGIME
    TRIGGER (`guard_metric < 0.3 AND bb_width > 0.035`), which selects moments when a
    forward move is collectable rather than merely moments when one occurred. This arm
    adds that trigger to forward_mae's return test and asks whether it recovers the
    result.

    `LABEL_MAX_DRAWDOWN = 0.04` is stop-matched rather than the study's learnability-
    optimal 0.10: a 10% cap admits paths that the ~4% ATR stop would close at a loss
    before the gain arrives. Measured BUY share is 7.38% at 0.04 against 7.13% at 0.10,
    so the trigger dominates the balance and this choice is nearly free.

    Reference balances: gbb 10.77%, this arm 7.38%, forward_mae H48 33.03% -- the last
    marking a third of all bars as buys, which is itself part of why it carries so
    little information per signal.
    """

    TRAINING_TYPE = LabelMethod.gbb_fwdmae
    LABEL_MAX_DRAWDOWN = 0.04
