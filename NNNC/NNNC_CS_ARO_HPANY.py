# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO` with 031's ANY-signal held-pair add (no in-profit condition): a fresh width-book
signal on a held pair adds one 1x stake whatever the trade's P&L (max 1 add, last fill >= 1 h old,
clocks restart from the add). Same mechanics and fresh-candle liquidity as cs_aro_hp; isolates the
in-profit condition. Pre-registered in research/tb_model/preregistration_lev2.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_HPANY(NNNC_CS_ARO):
    HELD_PAIR_ADD = {"in_profit_only": False}
