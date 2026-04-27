"""
GAN backend implementations.

Importing this package registers every concrete backend in the global
``GANs.GANBackend`` registry — that's how ``resolve_backend()`` finds them.

Adding a new backend:
  1. Create ``backends/<name>.py`` exporting a subclass of GANBackend.
  2. Decorate the class with ``@register_backend``.
  3. Add ``from . import <name>  # noqa`` below so it's loaded at import time.
"""

from . import ctab_gan  # noqa: F401  — registers CTAB-GAN + MT-CTAB-GAN backends
from . import cgan      # noqa: F401  — registers CGAN backend
from . import wgan      # noqa: F401  — registers WGAN TF + MLX backends
