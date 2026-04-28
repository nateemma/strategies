"""
Shared GAN storage layout.

All GAN models for a strategy live under one parent directory
(``GANs/`` or ``GANs_PCA/`` for the PCA-reduced variant), with a
per-type subdirectory keyed by ``GANType.name``.  Putting WGAN, CTAB-GAN,
MT-WGAN, etc. side-by-side under one parent removes the historical
``GANs/`` vs ``CTABGANs/`` vs ``MTCTABGANs/`` split and lets a strategy
that consumes multiple GAN types (e.g. ``BOTH`` mode) keep them all in
one place without filename collisions.

This module is the single source of truth for the layout — every caller
that needs a GAN save/load path goes through ``gan_save_subdir`` rather
than constructing a string locally.  When the convention changes, this
function is the only thing that has to change.
"""
from __future__ import annotations

import os

from GANs.GANType import GANType


GAN_PARENT_DIR: str = "GANs"
GAN_PCA_PARENT_DIR: str = "GANs_PCA"


def gan_save_subdir(gan_type: GANType, *, use_pca: bool = False) -> str:
    """Return the per-type subdir under the strategy's GANs parent.

    Examples (with ``use_pca=False``):
        GANType.WGAN         → "GANs/wgan"
        GANType.CTAB_GAN     → "GANs/ctab_gan"
        GANType.MT_WGAN      → "GANs/mt_wgan"
        GANType.MT_CTAB_GAN  → "GANs/mt_ctab_gan"

    With ``use_pca=True`` the parent is ``GANs_PCA/`` instead.

    The returned path is relative — combine with the strategy's
    storage location to get an absolute path:

        os.path.join(strategy.get_storage_location(), gan_save_subdir(...))
    """
    parent = GAN_PCA_PARENT_DIR if use_pca else GAN_PARENT_DIR
    return os.path.join(parent, gan_type.name.lower())


def gan_save_path(
    storage_location: str,
    gan_type: GANType,
    *,
    use_pca: bool = False,
) -> str:
    """Convenience: full absolute path = storage_location / GANs / <type>."""
    return os.path.join(storage_location, gan_save_subdir(gan_type, use_pca=use_pca))
