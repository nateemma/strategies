# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_LblMatchG_s7 — buy label guards MATCHED to the live entry gate. Seed 7.

The production buy label and the post-prediction entry gate filter the SAME two
quantities in the SAME direction, at DIFFERENT thresholds:

                    LABEL requires      ENTRY gate requires     binds
    guard_metric    < -0.2              < 0.3                   LABEL
    bb_width        > 0.035             > 0.06                  ENTRY

The mismatch runs in opposite directions. On guard_metric the label is stricter
by 0.5, so `entry_guard_threshold` is INERT (no labelled bar can fail it) and
the model is never taught about bars in -0.2..0.3 that the gate would happily
accept. On bb_width the gate is stricter, so the model IS taught about bars it
can never act on.

Measured over 1,813,739 bars / 11 pairs, against the live NNNC gate
(guard<0.3, bb_width>0.06, rvol>2.0, volume>0), which passes 10,380 bars:

    label setting              labelled   % bars   & passes gate   % of label
    current    (-0.2/0.035)     129,196    7.12%           5,374        4.2%
    match gm   ( 0.3/0.035)     181,192    9.99%           6,589        3.6%
    match both ( 0.3/0.06 )      65,107    3.59%           6,589       10.1%

Matching guard_metric raises TRADEABLE labelled bars 5,374 -> 6,589 (+22.6%):
the share of gate-passing bars carrying a buy label goes 51.8% -> 63.5%.
Matching bb_width as well adds NO further tradeable bars (identical 6,589) and
only removes 64,089 the gate rejects anyway -- denoising, not signal.

THIS ARM matches guard_metric ONLY (0.3 / 0.035).
Label population 181,192 bars (9.99%).

This is NOT the unguarded experiment (NNNC_Unguarded_s*), which deleted both
conditions, grew the label to 41.3% of bars and collapsed completely: class-1
precision and recall both exactly 0.000 on a 25.5% base rate, val_mcc 0.09,
zero trades at threshold 0.8 and 0.5, and +1.075%/-0.333% when forced to trade
at 0.35 against the guarded baseline's +9.523%. That closed the question of
whether the label guards are redundant: they are not, they carry the signal.
This arm keeps them and only re-points them at the gate the model actually
trades through.

Only the BUY label moves. The sell label carries the mirrored asymmetry
(label guard_metric > 0.2 vs exit_guard_threshold -0.2) and is deliberately
left alone so the arm moves exactly one mechanism.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from LabelParamsMixin import LabelParamsMixin
from NNNC_MLX import NNNC_MLX


class NNNC_LblMatchG_s7(LabelParamsMixin, NNNC_MLX):

    LABEL_PARAMS = {"guard_threshold": 0.3, "bb_width_threshold": 0.035}

    def get_model_path(self) -> str:
        name = self.__class__.__name__
        return self.get_storage_location() + name + "/" + name + ".keras"
