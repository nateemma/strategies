# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_InJit_P0 — Stage-0 input jitter at the LOOSENED (powered) operating point.

Same training-free input-jitter mechanism as NNNC_MLX_InJit (reuses NNNC_MLX
weights), but at prediction_threshold=0.45 so the marginal / high-entropy decisions
the voting actually acts on are TRADED — the tight-guard sweeps only traded the
confident tail where voting is inert (docs/GAN_TODO.md methodology note;
docs/coconut_study.md). Re-run of the Stage-0 sweep at power.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX_InJit import NNNC_MLX_InJit


class NNNC_MLX_InJit_P0(NNNC_MLX_InJit):

    buy_params = {**NNNC_MLX_InJit.buy_params, "prediction_threshold": 0.45}
