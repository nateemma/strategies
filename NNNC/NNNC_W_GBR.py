# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book for GBR (research/model_calibration/build_width_arms.py; GENERATED): the gbb rule +
bb_width >= this fold's cut instead of the model's prediction; everything else is `NNNC_FS_GBR_GR`
(control `fs_gbr_gr`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _width_book import WidthBookMixin
from NNNC_FS_GBR_GR import NNNC_FS_GBR_GR


class NNNC_W_GBR(WidthBookMixin, NNNC_FS_GBR_GR):
    WIDTH_CUTS = {
        '20210101-20230101': 0.12949822504088457,
        '20210101-20230401': 0.12949822504088457,
        '20210101-20230630': 0.2045258371956995,
        '20210101-20230928': 0.1150673691690824,
        '20210101-20231227': 0.0824100257192934,
        '20210101-20240326': 0.09950144610867244,
        '20210101-20240624': 0.10715937963550863,
        '20210101-20240922': 0.09241125706696328,
        '20210101-20241221': 0.10201362377554862,
        '20210101-20250321': 0.09197410047012389,
        '20210101-20250619': 0.10919673308977804,
        '20210101-20250917': 0.08705661422928966,
        '20210101-20251216': 0.09213719624815676,
        '20210101-20260316': 0.08618340620792839,
        '20210101-20260614': 0.09946384151284596,
    }
