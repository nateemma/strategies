# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO` (ARO as a capital-neutral width-book sizer) with 031's held-pair add switched back
on as the NNNC default has it: a fresh width-book signal on a pair holding a trade in profit adds one
stake (max 1 add, last fill >= 1 h old, clocks restart from the add). The add is 1x the proposed
stake: freqtrade does not call custom_stake_amount for it, so the sizer's weight sets only the first
fill. Pre-registered in research/tb_model/preregistration_stack.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_HP(NNNC_CS_ARO):
    HELD_PAIR_ADD = {"in_profit_only": True}
