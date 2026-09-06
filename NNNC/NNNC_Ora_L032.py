import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Oracle import NNNC_Oracle


class NNNC_Ora_L032(NNNC_Oracle):
    """ALIGNMENT test — label target 3.2% against cexit_take_profit 2.4%.

    The label currently certifies a +2.0% move while the exit requires +2.4%, so
    12.6% of labelled bars never reach the level the strategy can monetise and
    are held past the point the label promised. This arm raises the LABEL to
    meet the exit.

    LOOKAHEAD CEILING arm, never promotable.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.032
    MIN_SELL_LOSS_THRESHOLD = 0.032
