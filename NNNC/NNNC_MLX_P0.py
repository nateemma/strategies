# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NNNC_MLX_P0 — Phase-0 powered baseline (NON-GAN control) for the GAN sample-quality
plan (docs/GAN_TODO.md #5).

Reuses the trained NNNC_MLX (no GAN) weights verbatim and only LOOSENS the entry
gate (prediction_threshold) for trade volume, matching NNNC_DDPM_MLX_P0 so the
GAN-vs-non-GAN A/B is statistically powered. prediction_threshold is inference-time
-> no retrain. Isolated as its own class so it never reads/writes NNNC_MLX.json
(which the running hyperopt owns).
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX


class NNNC_MLX_P0(NNNC_MLX):

    # Loosened entry gate for a powered baseline (was 0.6). Keep IDENTICAL to
    # NNNC_DDPM_MLX_P0 so the only difference between the two is GAN augmentation.
    buy_params = {**NNNC_MLX.buy_params, "prediction_threshold": 0.45}

    def get_model_path(self) -> str:
        # Reuse the production NNNC_MLX weights, not this class's own name.
        root_dir = self.get_storage_location()
        return root_dir + "NNNC_MLX/NNNC_MLX.keras"
