"""Brief 07 — do label-free learned embeddings carry information the hand-built features lack?

DESIGNED TO BE KILLED CHEAPLY. A Spearman screen gates everything: no backtest runs unless the
embedding beats the hand-built features on nested out-of-sample data. That ordering is the
brief's main virtue.

TS2VEC ONLY. Kronos-small and Chronos-Bolt are out of scope -- neither `transformers` nor any
pretrained weights are installed, and adding them means new dependencies in the production venv
plus an external fetch, for a brief whose own expected result is null. Nothing here installs a
package or downloads a model.

MLX, NOT TORCH. torch is present but only transitively; this repository's NN stack is MLX
throughout -- nine MLX GAN implementations, MLXBaseClassifier/Predictor/AnomalyDetector, and
BaseNNStrategy itself.

THE BAR IS MEASURED, NEVER QUOTED. rho ~ 0.15 is widely cited here as "the OHLCV information
ceiling". Measured across five folds and six horizons, it is a HORIZON-3-TO-5 number:

    H        3        6       12       24       48       96
    median  0.1717   0.1127   0.0944   0.0720   0.0826   0.0583

Predictability decays with forward horizon, so a bar quoted at one horizon is the wrong bar at
another. Each screen carries its own bar, computed by the same ridge on the same split.

THE SCREEN MEASURES LEARNABILITY, WHICH IS NOT EDGE. This repository has a recorded case of a
+0.155 MCC producing identical trades. Passing the screen EARNS A BACKTEST; it is never an
acceptance.
"""
from __future__ import annotations

_VENV = "freqtrade/.venv/bin/python"
_missing = []
for _m in ("numpy", "pandas", "scipy", "sklearn", "mlx"):
    try:
        __import__(_m)
    except ImportError:                                        # pragma: no cover
        _missing.append(_m)
if _missing:                                                   # pragma: no cover
    import sys as _sys
    raise ImportError(
        f"research.labelfree.ssl needs {', '.join(_missing)}, missing from {_sys.executable}.\n"
        f"Run under {_VENV}. Note ~/.pyenv/shims/python3 is a different 3.12.13 carrying neither "
        f"pandas nor scipy, and MLX is required here — this feature cannot run without it."
    )
del _m, _missing
