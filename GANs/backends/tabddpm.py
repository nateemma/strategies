"""TabDDPM (MLX) backend — adapter onto the GANBackend registry."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType


# Reuse the WGAN helper to coerce 3-D (N, 1, F) → 2-D (N, F) — same
# shape conventions across the single-task GAN backends.
from GANs.backends.wgan import _data_to_2d, _mlx_available


# Kwargs the TabDDPMMLX constructor accepts. Anything not in this set
# (e.g. CTAB-specific keys callers might forward via a single config
# dict) is silently dropped — same pattern as _CTAB_MLX_CTOR_KEYS.
_TABDDPM_CTOR_KEYS: frozenset = frozenset({
    "d_model", "d_layers", "dropout",
    "num_timesteps", "num_sample_steps",
    "epochs", "batch_size",
    "learning_rate", "weight_decay",
    "ema_decay", "eval_frequency", "verbose",
})


_META_FILENAME = "tabddpm_metadata.pkl"
_WEIGHTS_FILENAME = "tabddpm_gen_mlx.safetensors"


@register_backend
class TabDDPMMLXBackend(GANBackend):
    """MLX backend for TabDDPM. No TF counterpart — production runs MLX."""

    GAN_TYPE = GANType.TAB_DDPM
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ---------- lifecycle ---------- #

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_tabddpm_mlx import TabDDPMMLX  # noqa: E402

        data_2d = _data_to_2d(data)
        labels_f32 = np.asarray(labels, dtype=np.float32)
        ctor_kwargs = {k: v for k, v in kwargs.items() if k in _TABDDPM_CTOR_KEYS}

        self._model = TabDDPMMLX(
            num_features=data_2d.shape[1],
            num_classes=labels_f32.shape[1],
            **ctor_kwargs,
        )
        self._model.fit(
            data_2d, labels_f32,
            categorical_columns=categorical_columns or [],
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "TabDDPMMLXBackend.generate called before fit/load — no model"
            )
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError(
                "generate() for TAB_DDPM requires keyword argument one_hot=<np.ndarray>"
            )
        return self._model.generate(n, one_hot)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("TabDDPMMLXBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["TabDDPMMLXBackend", Dict[str, Any]]:
        meta_p = os.path.join(save_path, _META_FILENAME)
        weights_p = os.path.join(save_path, _WEIGHTS_FILENAME)
        if not (os.path.exists(meta_p) and os.path.exists(weights_p)):
            raise FileNotFoundError(
                f"No MLX-format TabDDPM model at {save_path} "
                f"(needs {_META_FILENAME} + {_WEIGHTS_FILENAME})"
            )
        from GANs.df_tabddpm_mlx import TabDDPMMLX  # noqa: E402

        instance = cls()
        instance._model, metadata = TabDDPMMLX.load_from(save_path)
        return instance, metadata
