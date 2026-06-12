"""Label-shape wrappers around a GANInterface, used by the multi-task GAN
augmentation path. Kept in GANs/ (a neutral utility location) so both the
single-task pipeline base and the multi-task strategy import them from here —
the base must not import from a subclass module.
"""
from typing import Dict

import numpy as np
import pandas as pd


class _UnflattenedGenerateWrapper:
    """Wraps a GANInterface so generate() returns (n, T, F) ndarrays.

    Multi-task tabular GAN backends (e.g. MT_CTAB_GAN) are trained on
    flattened sequence windows and natively produce (n, T*F) DataFrames
    with column names like ``<feature>_<t>``. preprocess_training_data
    needs to mix those with the real (n, T, F) tensor that arrives from
    prepare_training_data, so we reshape the synth output here once,
    then balance_multi_task and swap_passthrough_columns can run on the
    3-D path uniformly.

    The reshape uses C-order, which is the same convention used to
    flatten during GAN training: index ``t*F + f`` on the flat axis
    maps to ``(t, f)`` on the un-flattened tensor.
    """

    def __init__(self, interface, T: int, F: int):
        self._interface = interface
        self._T = T
        self._F = F
        # Mirror commonly-inspected attributes so callers can still read
        # gan_type / save_path / model through the wrapper.
        self.gan_type = getattr(interface, "gan_type", None)

    def __getattr__(self, name):
        return getattr(self._interface, name)

    def generate(self, n, **kwargs):
        result = self._interface.generate(n=n, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            data, labels = result
            return self._unflatten(data), labels
        return self._unflatten(result)

    def _unflatten(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if not isinstance(data, np.ndarray):
            return data
        if data.ndim == 2 and data.shape[1] == self._T * self._F:
            return data.reshape(data.shape[0], self._T, self._F)
        if data.ndim == 3:
            return data
        # Anything else -- leave as-is and let the caller's shape check
        # surface the mismatch with an informative error.
        return data


class _PadMissingTaskLabelsWrapper:
    """Wraps a GANInterface so generate() pads missing task labels with
    uniform-random one-hot encodings.

    Used by single-task NNNC strategies running against a multi-task GAN:
    the strategy provides only ``{"trading": one_hot}``, but the loaded GAN
    was trained conditioned on N tasks. Calling generate() with only one
    task is an OOD input regime for the model — diagnostics showed this
    degrades sample quality (synth lag-1 autocorrelation flips negative
    against a real autocorrelation of +0.98 — see the NNNC_DDPM_MLX_LSTM_MT
    diagnostic at 2026-05-13). Filling the missing task slots with
    uniform-random one-hots restores in-distribution conditioning.

    The "uniform-random" choice is a deliberate non-informative prior on
    the auxiliary tasks; we don't want their values to systematically bias
    the trading-task generation.
    """

    def __init__(self, interface, expected_task_label_dims: Dict[str, int]):
        self._interface = interface
        self._task_dims = dict(expected_task_label_dims)
        self.gan_type = getattr(interface, "gan_type", None)

    def __getattr__(self, name):
        return getattr(self._interface, name)

    def generate(self, n, **kwargs):
        task_labels = dict(kwargs.get("task_labels", {}))
        for task, dim in self._task_dims.items():
            if task in task_labels:
                continue
            idx = np.random.randint(0, dim, size=n)
            task_labels[task] = np.eye(dim, dtype=np.float32)[idx]
        kwargs["task_labels"] = task_labels
        return self._interface.generate(n, **kwargs)
