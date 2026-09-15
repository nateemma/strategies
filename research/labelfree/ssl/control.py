"""The random-Gaussian control (T021-T023).

Separates INFORMATION from CAPACITY. A ridge's capacity scales with its input width, so a
320-dimensional embedding and a 24-column feature matrix are not comparable through a shared
projection — part of any difference would be width. The control is the same width as the
embedding, through the same ridge on the same splits, carrying nothing.

If the control approaches the embedding, the result is capacity rather than information (FR-012)
and the verdict is a KILL. If the control approaches the HAND-BUILT bar, the screen itself is
broken and nothing it reports is usable.
"""
from __future__ import annotations

import numpy as np


def gaussian_block(n_rows: int, width: int, seed: int = 0) -> np.ndarray:
    """Exactly the embedding's width, so the comparison isolates information."""
    return np.random.default_rng(seed).standard_normal((n_rows, width)).astype(np.float32)
