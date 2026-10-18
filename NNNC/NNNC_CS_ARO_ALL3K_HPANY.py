# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""All-available entry (limit 3,000 x weight) plus 031's any-signal held-pair add: a fresh width-book
signal on a held pair adds one stake on that later candle, at min(900 proposed, its own fillable cap)
(HELD_PAIR_ADD add_fraction is capped at 1, so the add is not raised to the 3,000 limit). Max 1 add.
Pre-registered in research/tb_model/preregistration_seed2.json (Q6)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_ALL3K import NNNC_CS_ARO_ALL3K


class NNNC_CS_ARO_ALL3K_HPANY(NNNC_CS_ARO_ALL3K):
    HELD_PAIR_ADD = {"in_profit_only": False}
