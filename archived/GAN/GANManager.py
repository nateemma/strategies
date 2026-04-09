from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from user_data.strategies.GAN.WGANGP import SingleTaskWGANGP
from user_data.strategies.GAN.MTWGANGP import MultiTaskWGANGP


class GANType(Enum):
    WGAN_GP = "wgan_gp"
    MT_WGAN_GP = "mt_wgan_gp"


class GANManager:
    """Factory that produces configured GAN instances."""

    DEFAULT_ROOT = Path("./saved_data/GANs")

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else self.DEFAULT_ROOT
        self._registry = {
            GANType.WGAN_GP: SingleTaskWGANGP,
            GANType.MT_WGAN_GP: MultiTaskWGANGP,
        }

    def list_types(self) -> Tuple[GANType, ...]:
        return tuple(self._registry.keys())

    def get_gan(self, gan_type: GANType, identifier: str, config: Optional[Dict[str, Any]] = None):
        gan_cls = self._require_impl(gan_type)
        default_config = {
            "root_dir": str(self.root_dir),
            "identifier": identifier,
        }
        if config:
            default_config.update(config)
        return gan_cls(**default_config)

    def exists(self, gan_type: GANType, identifier: str, config: Optional[Dict[str, Any]] = None) -> bool:
        return self.get_gan(gan_type, identifier, config).exists()

    def _require_impl(self, gan_type: GANType):
        if gan_type not in self._registry:
            raise ValueError(f"Unknown GAN type {gan_type!r}")
        return self._registry[gan_type]


__all__ = ["GANManager", "GANType"]
