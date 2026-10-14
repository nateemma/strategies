# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book for GFM (research/model_calibration/build_width_arms.py; GENERATED): the gbb rule +
bb_width >= this fold's cut instead of the model's prediction; everything else is `NNNC_FS_GFM_GR`
(control `fs_gfm_gr`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _width_book import WidthBookMixin
from NNNC_FS_GFM_GR import NNNC_FS_GFM_GR


class NNNC_W_GFM(WidthBookMixin, NNNC_FS_GFM_GR):
    WIDTH_CUTS = {
        '20210101-20230101': 0.10539052636216215,
        '20210101-20230401': 0.10539052636216215,
        '20210101-20230630': 0.14759362457940767,
        '20210101-20230928': 0.08674418511417488,
        '20210101-20231227': 0.09594577772312164,
        '20210101-20240326': 0.07863884568058015,
        '20210101-20240624': 0.11828397082954109,
        '20210101-20240922': 0.10463773339564683,
        '20210101-20241221': 0.08816107264393575,
        '20210101-20250321': 0.09362036297244672,
        '20210101-20250619': 0.10641549086107117,
        '20210101-20250917': 0.07352005326191725,
        '20210101-20251216': 0.11838515760830623,
        '20210101-20260316': 0.10892918959435993,
        '20210101-20260614': 0.07645728390692727,
    }
