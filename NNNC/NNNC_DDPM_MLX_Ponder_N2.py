# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Stage-2 ponder A/B variant: ponder_steps=2 (N=0 is the matched control).
Trains its own model on the pinned window; reuses shared TabDDPM + scalers."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_DDPM_MLX_Ponder import NNNC_DDPM_MLX_Ponder


class NNNC_DDPM_MLX_Ponder_N2(NNNC_DDPM_MLX_Ponder):
    ponder_steps = 2
