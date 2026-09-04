import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_BreakoutOpen import NNNC_BreakoutOpen


class NNNC_Brk_L1(NNNC_BreakoutOpen):
    """Feature 003 arm L1 -- the label effect in ISOLATION (US1, FR-004, FR-008).

    Candidate C1_h48_t02, admitted by regime/screen_candidates.py:

        density 5.81% of bars   (floor 5.5%)
        median labelled gain    4.29%   vs the control's 2.83%
        clears the -5% stop     40.6%   vs 26.0%
        reward:risk             0.86    vs 0.57
        median time to peak     8.2h    vs 6.8h -- INSIDE the 12h unclog window

    The last line is why this candidate goes first. It raises the gain threshold
    4x without moving the horizon, so the payoff still peaks before the 12h
    forced exit at Framework/BaseStrategy.py:1040 and NO exit-side change is
    needed. Whatever this arm shows is attributable to the label alone
    (Constitution VIII -- one parameter space at a time).

    C3_h96_t03_TAIL has better economics (r:r 1.28) but peaks at 17.0h, outside
    the hold window; it belongs to Phase 5 and must be measured together with a
    hold-window override so the two contributions separate (FR-011).

    Everything else -- guards, prediction_threshold, ATR stop band, exit config
    -- is inherited unchanged from the host, and NNNC_Brk_L1.json pins the same
    effective configuration as NNNC_BreakoutOpen.json (FR-016).
    """

    # C1_h48_t02: the gain threshold rises 0.005 -> 0.02; HORIZON stays at 48.
    HORIZON = 48
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
