import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _hold_window import HoldWindowMixin
from NNNC_BreakoutOpen import NNNC_BreakoutOpen


class NNNC_Brk_L2(HoldWindowMixin, NNNC_BreakoutOpen):
    """Feature 003 arm L2 -- the PACKAGE: tail label + hold window (US2, FR-005).

    Candidate C3_h96_t03_TAIL, the best-economics cell admitted by screening:

        density 5.92% of bars   (floor 5.5%)
        median labelled gain    6.41%   vs the control's 2.83%
        clears the -5% stop     66.8%   vs 26.0%
        reward:risk             1.28    vs 0.57
        median time to peak     17.0h   vs 6.8h  -- OUTSIDE the 12h forced exit
        stopped before peak     11.2%   vs 4.5%

    The last two lines are why this arm MUST carry the hold-window override.
    A 17.0h median peak against a 12h forced exit would be harvested early by
    construction, so the label would be measured through an exit rule that
    forbids its payoff. research.md R4 accepts the consequence: this feature
    cannot say whether the tail label ALONE would have sufficed, because that
    configuration is knowably compromised and would yield a false negative
    rather than a test.

    Contributions separate against H1, which carries the SAME hold window on
    the control's label (FR-011, T028):

        hold window alone            = H1 - control
        tail label given the window  = L2 - H1

    HORIZON 96 at the 15m timeframe is a 24h label horizon, which is exactly
    HOLD_WINDOW_HOURS -- the exit window is set to the label's own horizon, not
    swept.
    """

    HORIZON = 96
    MIN_BUY_GAIN_THRESHOLD = 0.03
    MIN_SELL_LOSS_THRESHOLD = 0.03

    HOLD_WINDOW_HOURS = 24.0
