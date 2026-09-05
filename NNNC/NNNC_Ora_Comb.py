import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Oracle import NNNC_Oracle


class NNNC_Ora_Comb(NNNC_Oracle):
    """T022 — COMBINED variant. LOOKAHEAD CEILING arm, never promotable.

    Run only after T018 (exit side) and T021 (label side) reported separately,
    so each contribution is attributable. Compared against arms differing from
    it in exactly ONE respect (FR-015, SC-005).

    Tight ATR band (2.0 / -3% .. -2%), the best exit-side result at -13.58%,
    combined with the 0.02 label threshold. Its one-respect comparators are
    ora_stight (same stop, baseline label) and ora_lbl02 (same label, baseline stop).
    """

    atr_stoploss_multiplier = 2.0
    atr_stoploss_floor = -0.03
    atr_stoploss_cap = -0.02

    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
