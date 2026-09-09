# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_Band30 — ATR stop-band arm. multiplier 3.0, floor -0.05, cap -0.03.
Midpoint between the two families.

WHY THIS SWEEP. On 20240819-20260809 every loss in BOTH families was a stop
exit, and the two families sit at different points on the same trade-off:

    family  band (mult/floor/cap)  stop exits        avg    total   Sharpe  DD
    NNMT    2.5 / -0.04 / -0.02    187/529 (35.3%)  -3.20%  -33.43pp  1.96  5.18%
    NNNC    4.0 / -0.06 / -0.04     66/276 (23.9%)  -5.17%  -24.05pp  1.51  5.89%

Tighter band => more stops, less damage each. Both returned ~+19.4%. The band
is therefore worth measuring rather than inheriting -- NNNC's 4.0 and NNMT's
2.5 were each set before the exit signal was enabled and before the current
label.

The band is a CLASS ATTRIBUTE, unreachable via run_arm.py --set, and
--stoploss is inert under use_atr_adaptive_stoploss (BaseNNStrategy sets it
True) -- hence a subclass.

NOT re-testing multiplier 15, which regressed ~3pp on NNNC and is closed.

get_model_path is pinned to NNNC_MLX so this pure re-parameterisation LOADS that
model instead of training a fresh one inside its own backtest (the base
implementation keys off self.__class__.__name__).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_Band30(NNNC_MLX):

    atr_stoploss_multiplier = 3.0
    atr_stoploss_floor = -0.05
    atr_stoploss_cap = -0.03

    def get_model_path(self) -> str:
        # PIN to the parent's trained weights -- exit-only re-parameterisation.
        return self.get_storage_location() + "NNNC_MLX/" + "NNNC_MLX.keras"
