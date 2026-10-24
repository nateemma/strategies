# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC_W_ARO on PURGED training windows (040 T053, research/train_window/rebuild.py; GENERATED): only the
width cuts differ -- keyed by the purged train_timerange, from width_cuts_aro_pw.json (causal). Same
class otherwise; the weights are the purged ARO folds' (model key from MODEL_TRAIN_TIMERANGE)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_ARO_PW(NNNC_W_ARO):
    WIDTH_CUTS = {
        "20210101-20221231": 0.1311432723233846,
        "20210101-20230331": 0.1311432723233846,
        "20210101-20230629": 0.1891973819038705,
        "20210101-20230927": 0.1653902810416832,
        "20210101-20231226": 0.09753794374141654,
        "20210101-20240325": 0.10369379141800907,
        "20210101-20240623": 0.128639811886972,
        "20210101-20240921": 0.12212899556601416,
        "20210101-20241220": 0.08809381808800318,
        "20210101-20250320": 0.08947598106275807,
        "20210101-20250618": 0.11689357671349858,
        "20210101-20250916": 0.08224300685674554,
        "20210101-20251215": 0.10534396804380293,
        "20210101-20260315": 0.10373607390792716,
        "20210101-20260613": 0.0968513952991455
}
