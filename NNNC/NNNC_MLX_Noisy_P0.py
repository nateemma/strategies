# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_Noisy_P0 — Stage-1 latent jitter at the LOOSENED (powered) operating point.

Same training-free latent-jitter (COCONUT) mechanism as NNNC_MLX_Noisy (reuses
NNNC_MLX weights), but at prediction_threshold=0.45. The hypothesis (user, 2026-07-20):
the tight-guard sweeps found latent voting near-inert because they traded only the
confident tail, filtering out the marginal decisions where the voting has leverage.
At 0.45 those marginal decisions trade, so any effect becomes measurable.
Re-run of the Stage-1 sweep at power.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX_Noisy import NNNC_MLX_Noisy


class NNNC_MLX_Noisy_P0(NNNC_MLX_Noisy):

    buy_params = {**NNNC_MLX_Noisy.buy_params, "prediction_threshold": 0.45}
