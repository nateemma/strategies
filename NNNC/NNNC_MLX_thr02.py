# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error

"""
NNNC_MLX_thr02 — gbb label with the gain threshold raised 0.007 -> 0.02.

Motivated by the 2026-09-04 tail analysis (regime/breakout_tail_analysis_README.md):
the gbb forward-gain distribution is heavily fat-tailed (skew ~3.8, excess
kurtosis ~30, top 10% of bars carrying ~38% of all upside), and at 0.007 the
threshold accepts nearly everything — the median labelled gain is 2.81% against a
2.4% take-profit, so only 57.8% of labelled bars clear the target that actually
harvests them.

At 0.02 the labelled population improves materially and stays inside every
constraint:

    threshold   %bars   median   clears TP 2.4%   time-to-peak
    0.007       8.75%    2.81%            57.8%           7.2h
    0.02        5.83%    4.05%            86.7%           8.5h

5.83% is just above the ~5.5% floor where aug_risk flags all-Hold collapse, and
8.5h is still inside the 12h unclog window — so unlike the breakout tail-targeting
package, this needs NO exit changes.

Carries a copy of NNNC_MLX.json so it runs the PRODUCTION exit config rather than
class defaults (a variant without its own params file silently falls back to
class defaults — notably cexit_take_profit 0.032 instead of the tuned 0.024,
which makes absolute numbers non-comparable).

Delete after the study.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_MLX_thr02(NNNC_MLX):
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
