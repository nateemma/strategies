# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC reference book since 2026-10-24 (user decision after 040): cs_aro (width book + ARO p_buy sizer)
with the regime-transition exposure scaler (040 tr_scale, research/nnmt/report.md section 6): stake
x1.333 when the stand-in's P(UP, up_pair_96) >= its fold tail p80, x0.667 when P(DOWN, down_pair_672)
>= p80 (both: unchanged). Walk-forward only: the predictions are the per-fold stand-in tables
(user_data/data/nnmt_transition/pred_fNN.parquet), read by fold id from the class name, so a class
with no fold id (live) raises -- a live stand-in is a separate build. Same weights as cs_aro.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _transition_signal import TrScaleMixin
from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_TR(TrScaleMixin, NNNC_CS_ARO):
    TR_UP_CELL = "up_pair_96"
    TR_DOWN_CELL = "down_pair_672"
