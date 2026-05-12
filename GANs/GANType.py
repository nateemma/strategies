"""
GANType — Enum identifying the GAN backend to use.

This file is intentionally self-contained: it imports nothing from the
strategies framework so the GAN subsystem can be used and tested
independently of Freqtrade strategy classes.
"""

from __future__ import annotations

from enum import Enum, auto


class GANType(Enum):
    """Identifies which GAN implementation to use.

    Naming convention:
        MT_*  — Multi-Task variant (labels are a dict of one-hot arrays)
        *_WGAN — WGAN-GP family (function-based, trains + augments in one call)
        *_CTAB_GAN — CTAB-GAN+ family (model-based: fit → generate → save/load)
    """

    NONE = auto()         # No GAN augmentation
    WGAN = auto()         # WGAN-GP, single-task, 2-D tabular (TF or MLX)
    MT_WGAN = auto()      # WGAN-GP, multi-task, 3-D sequential (TF or MLX)
    CTAB_GAN = auto()     # CTAB-GAN+, single-task, tabular (TF or MLX)
    MT_CTAB_GAN = auto()  # CTAB-GAN+, multi-task, tabular (TF)
    CGAN = auto()         # Conditional GAN, single-task, sequential (TF)
    BOTH = auto()         # WGAN pre-processing + CTAB-GAN augmentation
    TAB_DDPM = auto()     # TabDDPM (tabular diffusion, MLX-only, continuous-only, single-task)
    MT_DDPM = auto()      # DDPM, multi-task, 3-D sequential (MLX-only, continuous-only)
