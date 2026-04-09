# pragma pylint: disable=C0114, C0115, C0116
"""
Factory helper to create specific GAN creators with a unified API.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple


class CreateGANFactory:
    """
    Lightweight factory for instantiating GAN creator strategies. New GAN types
    can be registered dynamically to extend the available catalogue.
    """

    _REGISTRY: Dict[str, Tuple[str, str]] = {
        "wgan": ("user_data.strategies.GANs.CreateWGAN", "CreateWGAN"),
        "mt_wgan": ("user_data.strategies.GANs.CreateMTWGAN", "CreateMTWGAN"),
    }

    @classmethod
    def register(cls, key: str, module_path: str, class_name: str) -> None:
        cls._REGISTRY[key] = (module_path, class_name)

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        return tuple(cls._REGISTRY.keys())

    @classmethod
    def create(cls, key: str, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        if key not in cls._REGISTRY:
            available = ", ".join(cls.available())
            raise ValueError(f"Unknown GAN type '{key}'. Available options: {available}")

        module_path, class_name = cls._REGISTRY[key]
        module = importlib.import_module(module_path)
        gan_class = getattr(module, class_name)
        return gan_class(gan_config=gan_config, **kwargs)

