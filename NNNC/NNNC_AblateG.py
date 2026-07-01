"""NNNC_AblateG — GAN-chain ablation consumer (throwaway).

Inherits the full production NNNC_DDPM_MLX GAN config (gan_type=TAB_DDPM,
gan_target_ratio, inference overrides, autoencoder filter, post-GAN
scaling) for parity, but selects the per-rung include_list (ABLATION_RUNG)
and isolates all storage under saved_data/_ablate/. Loads the rung's GAN
trained by CreateTabDDPMAblateG. NOT for production use.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_DDPM_MLX import NNNC_DDPM_MLX
from _ablation_config import current_include_list, AblateGANMixin


class NNNC_AblateG(AblateGANMixin, NNNC_DDPM_MLX):
    include_list = current_include_list()

    # MEASUREMENT EXERCISE ONLY — not a ship config. The strategy is
    # structurally low-frequency (28 trades / 720d in-sample), so at the
    # production prediction_threshold=0.75 + guards-on a 30d OOS yields ~3
    # trades — too few to discriminate feature sets. Lower the threshold and
    # disable entry guards to raise trade count. Applied identically to every
    # rung, so the RELATIVE comparison stays fair.
    import os as _os
    _thr = float(_os.environ.get("ABLATE_PRED_THR", "0.6"))
    _guards = _os.environ.get("ABLATE_GUARDS", "off").lower() in ("1", "true", "on", "yes")
    buy_params = {
        **NNNC_DDPM_MLX.buy_params,
        "prediction_threshold": _thr,
        "entry_enable_guards": _guards,
    }

    # Production passthrough is ["atr_norm", "spread_ma"]; some rungs drop
    # those, and passing a non-include_list column to GAN passthrough is
    # meaningless. Filter to the rung's actual features. (current_include_list
    # is referenced as a module global so it's visible in the comprehension
    # scope — class-body names are not.)
    gan_passthrough_columns = [
        c for c in NNNC_DDPM_MLX.gan_passthrough_columns
        if c in current_include_list()
    ]
