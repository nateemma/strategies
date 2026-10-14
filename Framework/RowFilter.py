"""A declared per-row predicate over a dataframe's columns (feature 032, contract C1).

A strategy may declare a row filter as DATA -- a tuple of `(column, op, value)` conditions, ANDed --
so it hashes deterministically into a model key and holds no strategy-specific code here. The same
mask is used wherever the filter applies (training rows, entry, threshold picking), so they all
select the same bars.
"""

from __future__ import annotations

import numpy as np


_OPS = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal}


def validate(spec) -> tuple:
    """The spec as a tuple of (str, str, float) conditions; raises ValueError if malformed."""
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError(f"row filter must be a non-empty tuple of (column, op, value): {spec!r}")
    out = []
    for cond in spec:
        if not isinstance(cond, (tuple, list)) or len(cond) != 3:
            raise ValueError(f"row filter condition must be (column, op, value): {cond!r}")
        col, op, value = cond
        if not isinstance(col, str) or op not in _OPS:
            raise ValueError(f"row filter condition has a bad column or op "
                             f"(ops: {sorted(_OPS)}): {cond!r}")
        out.append((col, op, float(value)))
    return tuple(out)


def row_filter_mask(df, spec) -> np.ndarray:
    """Boolean array, len(df): the AND of the conditions. NaN fails. A missing column raises
    KeyError. `df` is not modified."""
    mask = np.ones(len(df), dtype=bool)
    for col, op, value in validate(spec):
        if col not in df.columns:
            raise KeyError(f"row filter column {col!r} is not in the dataframe")
        x = df[col].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            mask &= _OPS[op](x, value) & ~np.isnan(x)
    return mask
