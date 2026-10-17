# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study, arm A (research/tb_model/): ARO's architecture and inputs trained on the ATR-scaled
triple barrier (`_atr_barrier_label`, 037's tbatr) on windows whose last bar has volume > 0, used as the
capital-neutral width-book sizer (agree = p_buy above the fold's 0.5-share threshold). Entries,
stakes and threshold rule are cs_aro's."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _atr_barrier_label import AtrBarrierLabelMixin
from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_TBA(AtrBarrierLabelMixin, NNNC_CS_ARO):
    TRAIN_ROW_FILTER = (("volume", "gt", 0.0),)
    MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)
