"""Independent corroboration against `jumpmodels` 0.1.1 (T022, T023).

The in-repo model is PRIMARY and carries the verdict (FR-014); the package corroborates and
never overrides.

THE AUDIT IS THE POINT, NOT A FORMALITY. Audited 2026-09-15 in site-packages/jumpmodels/jump.py:

    dp(...)                    :135-139  traceback  -> SMOOTHED
    JumpModel.predict()        :606-616  use_viterbi=True BY DEFAULT -> SMOOTHED
    JumpModel.predict_online() :566-581  value_mx.argmin(axis=1)     -> FILTERED

So the package's obvious method is the LEAK. Reaching for `predict()` would have silently
reintroduced exactly what this feature exists to measure, and the cross-check would then have
agreed with our smoothed reference while disagreeing with our honest one -- a failure that
reads as a bug in OUR code. This module uses `predict_online` and asserts the audit first.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np

from . import AGREEMENT_FLOOR, JUMPMODELS_VERSION


def audit_package_inference() -> dict:
    """Confirm the package exposes a genuinely filtered predictor. Its default is NOT assumed."""
    import importlib.metadata as md
    from jumpmodels.jump import JumpModel

    ver = md.version("jumpmodels")
    online_src = inspect.getsource(JumpModel.predict_proba_online)
    default_src = inspect.getsource(JumpModel.predict_proba)

    filtered = "argmin(axis=1)" in online_src and "traceback" not in online_src
    default_is_viterbi = "use_viterbi" in default_src or "viterbi" in default_src.lower()

    return {
        "version": ver,
        "version_pinned_ok": ver == JUMPMODELS_VERSION,
        "filtered": bool(filtered),
        "online_method": "predict_online",
        "default_predict_is_smoothed": bool(default_is_viterbi),
        "interpreter": sys.executable,
        "evidence": ("predict_proba_online uses value_mx.argmin(axis=1) (forward-only); "
                     "predict/predict_proba default to the Viterbi traceback"),
    }


def _sparse_model(xs: np.ndarray, k: int, lam: float, kappa: float, seed: int):
    """Construct `SparseJumpModel` with OUR conventions translated into THEIRS.

    Three translations, each verified against the package source (2026-09-15):
      max_feats    = kappa**2          (sparse_jump.py:366, `norm_ub = sqrt(max_feats)`)
      jump_penalty = lam * sqrt(d)     (sparse_jump.py init_jm divides by sqrt(n_features))
      mode_loss    = irrelevant at k=2 (jump.py:426-429 -- for a symmetric penalty both rows
                                        give the same logsumexp and the correction cancels)
    """
    from jumpmodels.sparse_jump import SparseJumpModel
    d = xs.shape[1]
    return SparseJumpModel(n_components=k, max_feats=kappa ** 2,
                           jump_penalty=lam * np.sqrt(d), random_state=seed)


def _vol_order(xs: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort([xs[s == c, 0].mean() if (s == c).any() else np.inf for c in range(k)])
    lut = np.empty(k, dtype=int)
    for rank, raw in enumerate(order):
        lut[raw] = rank
    return lut[s]


def package_states(xs: np.ndarray, k: int, lam: float, seed: int = 0) -> np.ndarray:
    """Fit and infer with `jumpmodels`, using the FILTERED predictor only.

    `xs` MUST ALREADY BE STANDARDISED, with the same statistics our own fit used.

    Why that is not a detail: `jumpmodels.jump` does no scaling of its own -- it ships a
    separate `preprocess.StandardScalerPD` for the caller to apply -- while `model.fit` here
    standardises internally. Passing raw features to one and standardised features to the
    other compares a penalty of 785 against a loss of order 1 in one case and order 1e-6 in
    the other. Measured 2026-09-15: doing so collapsed the package to a SINGLE state (high-vol
    share 0.000) and the cross-check read 0.4863, tripping the FR-016 stop on what was an
    adapter bug rather than a disagreement between the models.

    Their penalty matrix is `jump_penalty * (ones - eye)` (jump.py:47), identical to ours, so
    on a common scale the two are directly comparable.
    """
    import pandas as pd

    a = audit_package_inference()
    if not a["filtered"]:
        raise SystemExit(f"jumpmodels inference audit FAILED: {a}")

    from .model import default_kappa
    X = pd.DataFrame(xs, columns=[f"f{i}" for i in range(xs.shape[1])])
    m = _sparse_model(xs, k, lam, default_kappa(xs.shape[1]), seed)
    m.fit(X)
    s = np.asarray(m.predict_online(X)).astype(int)
    # Match our volatility ordering so the comparison is about STATES, not label permutations.
    return _vol_order(xs, s, k)


def matched_parameter_agreement(xs: np.ndarray, k: int, lam: float, seed: int = 0) -> dict:
    """FR-016a, the HARD gate -- does our inference recursion match theirs?

    Runs OUR filtered recursion using THEIR fitted weights and centroids. This isolates the
    inference from the fit, which matters because the two are wrong in completely different
    ways: a bad recursion is a defect, a different local optimum is a tie-break.

    Measured 2026-09-15 on SOL/USDT: 0.8935 independent, 0.9360 with their weights, 0.9708
    with their weights and centroids. The recursion is right.
    """
    import dataclasses

    import pandas as pd

    from .model import default_kappa, fit
    from .online import filtered_states

    X = pd.DataFrame(xs, columns=[f"f{i}" for i in range(xs.shape[1])])
    m = _sparse_model(xs, k, lam, default_kappa(xs.shape[1]), seed)
    m.fit(X)
    theirs = _vol_order(xs, np.asarray(m.predict_online(X)).astype(int), k)

    w = np.asarray(m.w, dtype=float)
    centres = np.asarray(m.centers_, dtype=float) / np.sqrt(np.maximum(w, 1e-12))
    ours_fit = fit(xs, k, lam, seed=seed, n_init=1)
    matched = dataclasses.replace(
        ours_fit, weights=w, centroids=centres,
        state_order=tuple(int(i) for i in np.argsort(centres[:, 0])))

    return {
        "matched": agreement(filtered_states(xs, matched), theirs),
        "independent": agreement(filtered_states(xs, ours_fit), theirs),
        "our_weights": ours_fit.weights.tolist(),
        "their_weights": w.tolist(),
        "our_l1": float(np.abs(ours_fit.weights).sum()),
        "their_l1": float(np.abs(w).sum()),
    }


def agreement(ours: np.ndarray, theirs: np.ndarray) -> float:
    n = min(len(ours), len(theirs))
    return float((np.asarray(ours[:n]) == np.asarray(theirs[:n])).mean())


def verdict(matched_rate: float) -> str:
    """FR-016a. Only the MATCHED-parameter rate gates; the independent rate is advisory."""
    return "agree" if matched_rate >= AGREEMENT_FLOOR else "DISAGREE_STOP"


def describe_divergence(res: dict) -> str:
    """FR-016b -- name the cause rather than thresholding a coin flip."""
    same_budget = abs(res["our_l1"] - res["their_l1"]) < 1e-3
    return ("different local optima of the same objective at the same L1 budget "
            f"({res['our_l1']:.3f})" if same_budget
            else f"different L1 budgets ({res['our_l1']:.3f} vs {res['their_l1']:.3f}) "
                 "-- a convention mismatch, not multi-modality")
