# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_DDPM_MLX_P2_aeoff_disp20 — the widening test (GAN_TODO #5, option 1).

AE filter OFF (so the widened, off-manifold synth isn't culled) + 2.0x output
dispersion (σ_ratio ~0.5 -> ~1.0, matching real variance; correlations preserved).
Judged RELATIVE to NNNC_DDPM_MLX_P2_aeoff (AE-off, no widen): the delta is the pure
effect of dispersing the synth to match real σ.
  - If aeoff_disp20 > aeoff (and toward/past non-GAN 21.24%) -> dispersion IS the
    lever; a Phase-3 on-manifold retrain is justified.
  - If <= aeoff -> off-manifold widening is junk; the AE was right; dead end.
Draw cap (balance.py, 400K) prevents the Metal-limit crash. pred_thr 0.45.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX


class NNNC_DDPM_MLX_P2_aeoff_disp20(NNNC_DDPM_MLX):

    buy_params = {**NNNC_DDPM_MLX.buy_params, "prediction_threshold": 0.45}

    gan_synth_autoencoder_threshold = None   # AE filter OFF
    gan_inference_dispersion_scale = 2.0      # widen synth to ~real variance
