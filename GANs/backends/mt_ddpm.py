"""MT-DDPM (MLX) backend — adapter onto the GANBackend registry.

Mirrors MTWGANMLXBackend's structure. There is no TF counterpart;
production runs MLX, and the single-task TabDDPM has no TF implementation either.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType
from GANs.backends.mt_wgan import _build_mt_balance_config, _mlx_available


_MLX_META_PATTERN = "mt_ddpm_meta_mlx*.pkl"


@register_backend
class MTDDPMMLXBackend(GANBackend):
    """MLX backend for multi-task DDPM."""

    GAN_TYPE = GANType.MT_DDPM
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ---------- fit / generate ----------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_mt_ddpm_mlx import balance_with_mt_ddpm_mlx  # noqa: E402

        if not isinstance(labels, dict):
            raise ValueError(
                "MT_DDPM expects labels as Dict[str, np.ndarray] "
                f"(one entry per task); got {type(labels).__name__}"
            )

        data_f32 = np.asarray(data, dtype=np.float32)
        config = _build_mt_balance_config(kwargs, labels)

        _, _, gan = balance_with_mt_ddpm_mlx(
            data_f32, labels, _return_model=True, **config
        )
        self._model = gan

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "MTDDPMMLXBackend.generate called before fit/load — no model"
            )
        task_labels = kwargs.get("task_labels")
        if task_labels is None:
            raise ValueError(
                "generate() for MT_DDPM requires keyword argument task_labels=<dict>"
            )
        return self._model.generate(n, task_labels)

    # ---------- save / load ----------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("MTDDPMMLXBackend.save called before fit — no model")
        # MTDDPMMLX.save takes extra_metadata as a single dict kwarg.
        self._model.save(save_path, extra_metadata=extra_metadata or None)

    @classmethod
    def load(cls, save_path: str) -> Tuple["MTDDPMMLXBackend", Dict[str, Any]]:
        candidates = sorted(glob.glob(os.path.join(save_path, _MLX_META_PATTERN)))
        if not candidates:
            raise FileNotFoundError(
                f"No MLX-format MT_DDPM model at {save_path} "
                f"(no files matching {_MLX_META_PATTERN})"
            )
        from GANs.df_mt_ddpm_mlx import MTDDPMMLX  # noqa: E402

        instance = cls()
        instance._model, metadata = MTDDPMMLX.load_from(save_path)
        return instance, metadata
