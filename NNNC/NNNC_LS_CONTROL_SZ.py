# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""034 label screen (ARO; GENERATED): the CONTROL `NNNC_A_Raw_Open` as a width-book sizer, for the S1 agree-share reference pick only (never backtested)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _width_book import WidthBookMixin
from _width_sizer import WidthSizerMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_LS_CONTROL_SZ(WidthSizerMixin, WidthBookMixin, NNNC_A_Raw_Open):
    WIDTH_CUTS = {
        '20210101-20230101': 0.12333389858998543,
        '20210101-20230401': 0.12333389858998543,
        '20210101-20230630': 0.14759362457940767,
        '20210101-20230928': 0.12765635235335906,
        '20210101-20231227': 0.08846952778279098,
        '20210101-20240326': 0.10649954455323984,
        '20210101-20240624': 0.10447016087339737,
        '20210101-20240922': 0.11175517782213314,
        '20210101-20241221': 0.09017314371743745,
        '20210101-20250321': 0.09279669497215012,
        '20210101-20250619': 0.107773276722266,
        '20210101-20250917': 0.08364406381408043,
        '20210101-20251216': 0.1042117985830952,
        '20210101-20260316': 0.10302842321761797,
        '20210101-20260614': 0.07839188593577942,
    }
