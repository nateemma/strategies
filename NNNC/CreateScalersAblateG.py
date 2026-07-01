"""CreateScalersAblateG — isolated scaler builder for the GAN-chain phase.

Same as CreateScalersAblate but isolates via get_storage_location
(saved_data/_ablate/) rather than scaler-name overrides, so the whole GAN
chain (scaler + GAN + model) shares one isolated root. Default scaler
names are fine since the directory is already isolated.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.CreateScalers import CreateScalers
from _ablation_config import current_include_list, AblateGANMixin


class CreateScalersAblateG(AblateGANMixin, CreateScalers):
    include_list = current_include_list()
