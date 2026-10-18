# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_W_ARO` with 031's held-pair add switched back on as the NNNC default has it: a fresh entry
signal on a pair holding a trade that is in profit adds one stake (max 1 add, last fill >= 1 h old,
clocks restart from the add). Pre-registered in research/tb_model/preregistration_reentry.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_ARO_HP(NNNC_W_ARO):
    HELD_PAIR_ADD = {"in_profit_only": True}
