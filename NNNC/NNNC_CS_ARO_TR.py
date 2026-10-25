# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC reference book since 2026-10-24 (user decision after 040): cs_aro (width book + ARO p_buy sizer)
with the regime-transition exposure scaler (040 tr_scale, research/nnmt/report.md section 6): stake
x1.333 when the stand-in's P(UP, up_pair_96) >= its fold tail p80, x0.667 when P(DOWN, down_pair_672)
>= p80 (both: unchanged). Same weights as cs_aro.

Since 2026-10-24 the stand-in is scored INSIDE the strategy (TR_SOURCE "standin",
NNNC/_transition_standin.py + utils/CloseHistoryStore.py), the same code a live bot runs: the fold's
fitted models are saved_data/transition/<train end>/. Verified before the switch: the strategy-side refit
reproduces the per-fold tables bit for bit (15/15 folds, both cells); cs_aro_trs (this path) made the
tables run's exact trades on 15/15 folds (65.351 both); on 920-candle live buffers the fire flags agree
600/600 (research/nnmt/live_check.json). Live folds: research/live_pnl/live_fold.py --book cs_aro_tr.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _transition_signal import TrScaleMixin
from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_TR(TrScaleMixin, NNNC_CS_ARO):
    TR_UP_CELL = "up_pair_96"
    TR_DOWN_CELL = "down_pair_672"
    TR_SOURCE = "standin"
    TR_STANDIN_ROOT = "transition_v2"  # v2 inputs (history store OHLCV): = the tables bit for bit, 15/15
