"""CreateTabDDPMAblateG — isolated TabDDPM builder for the ablation GAN chain.

Trains a TabDDPM GAN on the per-rung include_list (ABLATION_RUNG) into the
isolated saved_data/_ablate/ root, so it never overwrites the global
production tab_ddpm GAN. Inherits CreateTabDDPM's config (use_post_gan_scaling,
class_balanced_sampling) for parity.
"""

import sys
from pathlib import Path

gan_dir = Path(__file__).parent
sys.path.append(str(gan_dir))                  # GANs (CreateTabDDPM / CreateGAN)
sys.path.append(str(gan_dir.parent / "NNNC"))  # _ablation_config

from CreateTabDDPM import CreateTabDDPM
from _ablation_config import current_include_list, AblateGANMixin


class CreateTabDDPMAblateG(AblateGANMixin, CreateTabDDPM):
    include_list = current_include_list()
