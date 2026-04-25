# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
Create_CGP_PCA — backwards-compatibility shim.

The only difference from ``CreateCtabGanPlus`` is ``use_pca_reduction``.
Kept so existing freqtrade configs continue to work; the actual
implementation lives in ``CreateGAN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateCtabGanPlus import CreateCtabGanPlus  # noqa: E402


class Create_CGP_PCA(CreateCtabGanPlus):
    use_pca_reduction = True
