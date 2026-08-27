"""Downstream-utility probe: does this GAN's synth actually help a classifier?

The missing axis. The GAN family has fidelity machinery (marginal / joint /
temporal moments via ``GANs.diagnostics``) but nothing that answers "is the synth
USEFUL". That distinction is not academic here: GAN_TODO §5 recorded TabDDPM synth
that was high-fidelity and still added no edge, and separately synth that raised
val_mcc while WORSENING P&L. Fidelity, learnability and P&L are three different
things and this module measures the middle one.

Deliberately cheap and deterministic: it runs once per registered backend, and its
job is to RANK variants, not to settle P&L. Anything it flags still needs the
powered paired-seed A/B protocol from GAN_TODO §5 before any production claim.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _flatten_2d(x: np.ndarray) -> np.ndarray:
    """(n, w, f) -> (n, w*f); (n, f) unchanged."""
    x = np.asarray(x)
    return x.reshape(x.shape[0], -1) if x.ndim > 2 else x


def _as_1d_labels(y: Any) -> np.ndarray:
    """Accept 1-D ints, one-hot, or a multi-task dict (primary task wins)."""
    if isinstance(y, dict):
        y = y[sorted(y)[0]]
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return y.reshape(-1, y.shape[-1]).argmax(axis=1)
    return y.reshape(-1)


def delta_val_mcc(
    real_x: np.ndarray,
    real_y: Any,
    synth_x: Optional[np.ndarray],
    synth_y: Any = None,
    *,
    seed: int = 42,
    val_fraction: float = 0.3,
    n_repeats: int = 10,      # k=1 FLIPS SIGN across seeds; see the docstring
) -> Dict[str, Any]:
    """MCC of a classifier trained on real vs real+synth, evaluated on real only.

    The validation split NEVER sees synthetic data -- otherwise a GAN that
    memorises the training set scores well for the wrong reason.

    Returns a dict with ``delta`` = mcc_aug - mcc_real, averaged over
    ``n_repeats`` PAIRED repeats (each repeat uses one split for both arms, so
    the split cancels). ``delta_sd`` reports the spread; a delta smaller than its
    own sd is not resolvable and must not be acted on.

    The single-repeat version of this probe was NOT resolvable: a 3-seed
    replication flipped the SIGN on 10 of 11 GAN variants, because one 300-sample
    draw against ~2800 training rows moves MCC by less than the classifier's own
    run-to-run noise. Repeats shrink that as 1/sqrt(k).

    ``delta`` is None with a ``reason`` when the probe cannot run; a broken
    variant must still yield a scorecard row rather than aborting the sweep.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import train_test_split

    out: Dict[str, Any] = {"mcc_real": None, "mcc_aug": None, "delta": None,
                           "delta_sd": None, "resolvable": None,
                           "n_synth": 0, "n_repeats": n_repeats, "reason": ""}

    rx, ry = _flatten_2d(real_x), _as_1d_labels(real_y)
    if len(rx) != len(ry):
        out["reason"] = f"real X/y length mismatch {len(rx)} vs {len(ry)}"
        return out
    if len(np.unique(ry)) < 2:
        out["reason"] = "real labels are single-class"
        return out

    def _one(rep: int):
        rs = seed + 1000 * rep
        strat = ry if np.min(np.bincount(ry.astype(int))) >= 2 else None
        xtr, xva, ytr, yva = train_test_split(
            rx, ry, test_size=val_fraction, random_state=rs, stratify=strat)

        def _fit_mcc(x, y):
            clf = HistGradientBoostingClassifier(max_iter=60, random_state=rs)
            clf.fit(x, y)
            return float(matthews_corrcoef(yva, clf.predict(xva)))

        base = _fit_mcc(xtr, ytr)
        if sx is None:
            return base, None
        return base, _fit_mcc(np.vstack([xtr, sx]), np.concatenate([ytr, sy]))

    sx = sy = None
    if synth_x is not None and len(synth_x) > 0:
        cand = _flatten_2d(synth_x)
        if cand.shape[1] != rx.shape[1]:
            out["reason"] = f"synth feature dim {cand.shape[1]} != real {rx.shape[1]}"
        elif not np.isfinite(cand).all():
            out["reason"] = "synthetic samples contain non-finite values"
        else:
            cy = _as_1d_labels(synth_y) if synth_y is not None else None
            if cy is None or len(cy) != len(cand):
                out["reason"] = "synthetic labels missing or mis-shaped"
            else:
                sx, sy = cand, cy
                out["n_synth"] = int(len(sx))
    elif synth_x is None or len(synth_x) == 0:
        out["reason"] = "no synthetic samples"

    bases, augs = [], []
    for rep in range(max(1, n_repeats)):
        b, a = _one(rep)
        bases.append(b)
        if a is not None:
            augs.append(a)

    out["mcc_real"] = float(np.mean(bases))
    if not augs:
        return out
    out["mcc_aug"] = float(np.mean(augs))
    deltas = np.asarray(augs) - np.asarray(bases)
    out["delta"] = float(deltas.mean())
    out["delta_sd"] = float(deltas.std(ddof=1)) if len(deltas) > 1 else None
    # A delta smaller than its own spread is not resolvable at this budget.
    out["resolvable"] = (out["delta_sd"] is not None
                         and abs(out["delta"]) > out["delta_sd"])
    return out
