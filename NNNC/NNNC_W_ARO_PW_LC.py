# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""T053 DIAGNOSTIC: the width book on the PURGED ARO folds (exits) with the LEGACY width cuts (mapped
fold by fold onto the purged train_timerange keys). Against w_aro_pw it isolates the cut change; against
w_aro the model's exit-side change."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_ARO_PW_LC(NNNC_W_ARO):
    WIDTH_CUTS = {
        "20210101-20221231": 0.12333389858998543,
        "20210101-20230331": 0.12333389858998543,
        "20210101-20230629": 0.14759362457940767,
        "20210101-20230927": 0.12765635235335906,
        "20210101-20231226": 0.08846952778279098,
        "20210101-20240325": 0.10649954455323984,
        "20210101-20240623": 0.10447016087339737,
        "20210101-20240921": 0.11175517782213314,
        "20210101-20241220": 0.09017314371743745,
        "20210101-20250320": 0.09279669497215012,
        "20210101-20250618": 0.107773276722266,
        "20210101-20250916": 0.08364406381408043,
        "20210101-20251215": 0.1042117985830952,
        "20210101-20260315": 0.10302842321761797,
        "20210101-20260613": 0.07839188593577942
}
