"""The deployability gate (T024, T024a, T025).

AGENT_GUIDE.md:545 prohibits features whose ranges differ by pair. A TS2Vec embedding carries no
such guarantee, so it is measured:

    ratio(d) = std_over_pairs( mean_within_pair(d) ) / mean_over_pairs( std_within_pair(d) )
    reject if max over d of ratio(d) > 0.30

A RATIO rather than an absolute, because embedding dimensions have arbitrary units.

0.30 IS ANCHORED ON PRODUCTION, not chosen. Measured across four folds, the production feature
set's worst single dimension reaches 0.2297 (f10), so 0.30 leaves ~1.3x headroom:

    fold   median    p90      max
    f05    0.0306   0.0563   0.0992
    f07    0.0260   0.0732   0.1203
    f10    0.0410   0.1425   0.2297
    f13    0.0383   0.1580   0.1792

An earlier draft used 0.5, which the pre-implementation analyze pass found to be over 4x
production's typical maximum -- a gate nothing plausible could fail, and therefore worse than no
gate, since a decorative PASS reads as corroboration beside the screen doing the real work.

A REJECTION IS A FINDING (FR-012b): an embedding whose scale depends on the pair cannot be
deployed here however well it screens. It is never normalised away silently.
"""
from __future__ import annotations

import numpy as np

THRESHOLD = 0.30


def ratios(per_pair: dict) -> np.ndarray:
    """One ratio per embedding dimension."""
    means = np.array([np.nanmean(v, axis=0) for v in per_pair.values()])
    stds = np.array([np.nanstd(v, axis=0) for v in per_pair.values()])
    return np.nanstd(means, axis=0) / np.maximum(np.nanmean(stds, axis=0), 1e-12)


def check(per_pair: dict, threshold: float = THRESHOLD) -> dict:
    r = ratios(per_pair)
    return {"threshold": threshold,
            "median": round(float(np.nanmedian(r)), 4),
            "p90": round(float(np.nanpercentile(r, 90)), 4),
            "max": round(float(np.nanmax(r)), 4),
            "n_dims_over": int(np.nansum(r > threshold)),
            "n_dims": int(r.size),
            "passed": bool(np.nanmax(r) <= threshold),
            "anchored_on": ("production's worst dimension across f05/f07/f10/f13 is 0.2297; "
                            "0.30 leaves ~1.3x headroom")}
