# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book `NNNC_W_ARO` with the reversion take-profit at 2 x ATR(14)/close of the signal bar,
unclamped (`_atr14_take_profit`); stop, grace and every other exit unchanged.
Pre-registered in research/tb_model/preregistration_tp2.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _atr14_take_profit import Atr14TakeProfitMixin
from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_ARO_TP2(Atr14TakeProfitMixin, NNNC_W_ARO):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.0}}
