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
    """Wraps a GANInterface so generate()'s ``task_labels`` are reconciled to
    EXACTLY the task set the loaded GAN was trained on: tasks the GAN expects
    but the caller didn't supply are padded with uniform-random one-hots, and
    tasks the caller supplied that the GAN was NOT trained on are dropped.

    Two callers rely on this:

    * Single-task NNNC strategies running against a multi-task GAN provide
      only ``{"trading": one_hot}``, but the loaded GAN was trained
      conditioned on N tasks. Calling generate() with only one task is an OOD
      input regime for the model — diagnostics showed this degrades sample
      quality (synth lag-1 autocorrelation flips negative against a real
      autocorrelation of +0.98 — see the NNNC_DDPM_MLX_LSTM_MT diagnostic at
      2026-05-13). Filling the MISSING task slots with uniform-random one-hots
      restores in-distribution conditioning.

    * Multi-task classifiers running against a REDUCED-task GAN (one trained
      with ``gan_condition_tasks`` restricting conditioning to a subset)
      supply the full label dict (all classifier heads), but the GAN only
      knows the subset. Passing a task the GAN's ``_TaskLabelEmbed`` has no
      embedding for would KeyError, so the EXTRA tasks are dropped here before
      generate(). The classifier's synth labels for those dropped tasks are
      still produced by the balancer (from its own ``labels`` bookkeeping),
      so the classifier keeps all its heads — those heads just condition on
      labels the GAN didn't shape (harmless when they're weight-0).

    The "uniform-random" choice for padding is a deliberate non-informative
    prior on the auxiliary tasks; we don't want their values to systematically
    bias the trading-task generation. When the supplied task set already
    matches the GAN's exactly (the common case), this is a no-op.
    """

    def __init__(self, interface, expected_task_label_dims: Dict[str, int]):
        self._interface = interface
        self._task_dims = dict(expected_task_label_dims)
        self.gan_type = getattr(interface, "gan_type", None)

    def __getattr__(self, name):
        return getattr(self._interface, name)

    def generate(self, n, **kwargs):
        supplied = dict(kwargs.get("task_labels", {}))
        # Pad tasks the GAN expects but the caller didn't supply.
        for task, dim in self._task_dims.items():
            if task in supplied:
                continue
            idx = np.random.randint(0, dim, size=n)
            supplied[task] = np.eye(dim, dtype=np.float32)[idx]
        # Drop tasks the caller supplied that the GAN was not trained on —
        # the GAN's label embedding has no slot for them.
        task_labels = {t: v for t, v in supplied.items() if t in self._task_dims}
        kwargs["task_labels"] = task_labels
        return self._interface.generate(n, **kwargs)
