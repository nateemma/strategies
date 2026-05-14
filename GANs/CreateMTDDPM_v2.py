# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore
"""
CreateMTDDPM_v2 — MT_DDPM creator that uses the post-GAN scaling pipeline.
GAN trains on RAW tensors; internal z-score handles model-side normalisation.
Saved separately from v1 via gan_save_path(post_gan_scaling=True).
"""

from __future__ import annotations
import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateMTDDPM import CreateMTDDPM


class CreateMTDDPM_v2(CreateMTDDPM):
    use_post_gan_scaling = True
