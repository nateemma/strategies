import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Oracle import NNNC_Oracle


class NNNC_Ora_StopMid(NNNC_Oracle):
    """T019 — ATR stop band variant. LOOKAHEAD CEILING arm, never promotable.

    The diagnosis (T012) named the ATR adaptive stop as the dominant destroyer
    of the oracle's guards-off P&L: 717 of 1760 trades, -11967.2, against
    +8258.3 earned by the take-profit. T017 then REFUTED the other half of the
    hypothesis — raising the take-profit cap made things monotonically worse, so
    winners are not being cut short. That leaves the stop side.

    NNNC's band is multiplier 4.0, floor -0.06, cap -0.04. These are CLASS
    attributes, unreachable via --set, and `--stoploss` is inert under
    use_atr_adaptive_stoploss — hence a subclass (research.md R6).

    This arm: multiplier 3.0, floor -0.045, cap -0.03.
    """

    atr_stoploss_multiplier = 3.0
    atr_stoploss_floor = -0.045
    atr_stoploss_cap = -0.03
