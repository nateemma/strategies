# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Both capital routes stacked: `NNNC_CS_ARO_HPANY` (any-signal held-pair add) with first-fill stakes
x1.7432 as in `NNNC_CS_ARO_LEV2`. The add stays 1x proposed (HELD_PAIR_ADD add_fraction is capped at 1),
so each route runs at its tested setting. Pre-registered in research/tb_model/preregistration_cap.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO_HPANY import NNNC_CS_ARO_HPANY


class NNNC_CS_ARO_HPANY_LEV2(NNNC_CS_ARO_HPANY):
    STAKE_SCALE = 1.7432
    MODEL_KEY_IGNORED = ("STAKE_SCALE",)

    def stake_weight(self, entry_tag) -> float:
        return super().stake_weight(entry_tag) * self.STAKE_SCALE
