from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class GANBase:
    """Shared functionality for GAN implementations."""

    MODEL_FILENAME: str = ""
    METADATA_FILENAME: str = ""
    DEFAULT_CONFIG: Dict[str, Any] = {
        "train_kwargs": {
            "epochs": 100,
            "batch_size": 256,
            "n_critic": 5,
            "verbose": True,
            "assess_quality": False,
        },
        "augment_kwargs": {
            "epochs": 100,
            "batch_size": 256,
            "n_critic": 5,
            "verbose": True,
            "assess_quality": True,
        },
    }

    def __init__(
        self,
        identifier: str,
        root_dir: str,
        train_kwargs: Dict[str, Any] | None = None,
        augment_kwargs: Dict[str, Any] | None = None,
        save_path: str | None = None,
        **_: Any,
    ) -> None:
        config = _deep_merge(
            copy.deepcopy(self.DEFAULT_CONFIG),
            {
                "train_kwargs": train_kwargs or {},
                "augment_kwargs": augment_kwargs or {},
            },
        )
        self.identifier = identifier
        root = Path(root_dir)
        default_save = Path(save_path) if save_path else Path(identifier)
        self.save_path = default_save if default_save.is_absolute() else root / default_save
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.train_kwargs = config["train_kwargs"]
        self.augment_kwargs = config["augment_kwargs"]
        self.train_kwargs.setdefault("save_path", str(self.save_path))
        self.augment_kwargs.setdefault("save_path", str(self.save_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def exists(self) -> bool:
        model_path = self.save_path / self.MODEL_FILENAME
        meta_path = self.save_path / self.METADATA_FILENAME
        return model_path.exists() and meta_path.exists()

    def train(self, dataframe: pd.DataFrame, labels: Any, **kwargs: Any) -> Dict[str, Any]:
        options = self._merge_kwargs(self.train_kwargs, kwargs)
        trained = self._train_impl(dataframe, labels, options)
        return {
            "identifier": self.identifier,
            "trained_samples": trained,
            "save_path": str(self.save_path),
        }

    def augment(self, dataframe: pd.DataFrame, labels: Any, **kwargs: Any) -> Tuple[pd.DataFrame, Any]:
        if not self.exists():
            raise FileNotFoundError(
                f"No trained GAN found for identifier '{self.identifier}'. Train the model before augmenting."
            )
        options = self._merge_kwargs(self.augment_kwargs, kwargs)
        return self._augment_impl(dataframe, labels, options)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_kwargs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        merged.update(overrides)
        return merged

    @staticmethod
    def _extract_dataframe(dataframe: pd.DataFrame) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("GAN operations expect pandas DataFrame inputs")
        return dataframe.to_numpy(dtype=np.float32, copy=True), tuple(dataframe.columns)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _train_impl(self, dataframe: pd.DataFrame, labels: Any, kwargs: Dict[str, Any]) -> int:
        raise NotImplementedError

    def _augment_impl(
        self,
        dataframe: pd.DataFrame,
        labels: Any,
        kwargs: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Any]:
        raise NotImplementedError


__all__ = ["GANBase"]
