# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Leverage control for cs_aro_hp: `NNNC_CS_ARO` (add OFF) with every stake scaled by STAKE_SCALE,
cs_aro_hp's committed capital over cs_aro's (time-weighted, 2023-26 pooled: 1.3187). Same entries,
exits and sizer weights; freqtrade's max_stake and the fillable (liquidity) cap still apply, so the
wallet binds at peaks as it does for the add. Pre-registered in research/tb_model/preregistration_lev.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_LEV(NNNC_CS_ARO):
    STAKE_SCALE = 1.3187
    MODEL_KEY_IGNORED = ("STAKE_SCALE",)

    def stake_weight(self, entry_tag) -> float:
        return super().stake_weight(entry_tag) * self.STAKE_SCALE
