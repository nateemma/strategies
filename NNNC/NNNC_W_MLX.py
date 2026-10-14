# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book for MLX (research/model_calibration/build_width_arms.py; GENERATED): the gbb rule +
bb_width >= this fold's cut instead of the model's prediction; everything else is `NNNC_FS_MLX_GR`
(control `fs_mlx_gr`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _width_book import WidthBookMixin
from NNNC_FS_MLX_GR import NNNC_FS_MLX_GR


class NNNC_W_MLX(WidthBookMixin, NNNC_FS_MLX_GR):
    WIDTH_CUTS = {
        '20210101-20230101': 0.09448985210273025,
        '20210101-20230401': 0.09448985210273025,
        '20210101-20230630': 0.06769427470368643,
        '20210101-20230928': 0.0600827729821402,
        '20210101-20231227': 0.06197169707892488,
        '20210101-20240326': 0.060256738576390245,
        '20210101-20240624': 0.06012475296572082,
        '20210101-20240922': 0.06006213186314775,
        '20210101-20241221': 0.06017449819991742,
        '20210101-20250321': 0.06074177273908371,
        '20210101-20250619': 0.0607756721219391,
        '20210101-20250917': 0.06312844890736422,
        '20210101-20251216': 0.06180249249998485,
        '20210101-20260316': 0.06049342655715235,
        '20210101-20260614': 0.06165033007445035,
    }
