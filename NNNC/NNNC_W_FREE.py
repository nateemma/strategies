# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book, cut = trailing-90-day candidate quantile at the fixed share S (research/train_window/width_free.py; GENERATED, preregistration_width_free.json): w_aro with
model-free cuts; only WIDTH_CUTS differ from NNNC_W_ARO."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_FREE(NNNC_W_ARO):
    WIDTH_CUTS = {
        '20210101-20230101': 0.13610597964784274,
        '20210101-20230401': 0.10164748441325241,
        '20210101-20230630': 0.12679443955386563,
        '20210101-20230928': 0.09157984745082653,
        '20210101-20231227': 0.09058871156243349,
        '20210101-20240326': 0.09598034667088509,
        '20210101-20240624': 0.10703307208903658,
        '20210101-20240922': 0.10308430143217649,
        '20210101-20241221': 0.10440032188264751,
        '20210101-20250321': 0.10021204843845957,
        '20210101-20250619': 0.09417106145742835,
        '20210101-20250917': 0.07297239171626291,
        '20210101-20251216': 0.11980835184589314,
        '20210101-20260316': 0.09618593701562088,
        '20210101-20260614': 0.0945329947441521,
    }
