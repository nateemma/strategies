# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P2_aeoff — AE-filter-OFF control for the dispersion probe (GAN_TODO #5).

The AE filter culls off-manifold synth; it also rejected the widened synth ~98%,
crashing the balance loop. To test whether WIDENING recovers P&L we turn the AE
off — but AE-off ALSO changes the synth (the AE was the only ingredient that ever
beat no-GAN, project_ae_filter_win). So this is the AE-off control (dispersion 1.0,
no widening): the widened test (aeoff_disp20) is judged RELATIVE to this, isolating
the pure widening effect. prediction_threshold 0.45 (powered).
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P2_aeoff(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_synth_autoencoder_threshold = None   # AE filter OFF
    gan_inference_dispersion_scale = 1.0      # no widening (control)
