# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC_LIVE_live1 — walk-forward fold live1 (feature 007). GENERATED, do not edit.

Train 2021-01-01..2026-10-15   Test 2026-10-16..2027-01-14 (90d, OUT OF SAMPLE)

Its own class identity gives it its own saved_data/ directory. Two folds sharing a class would
overwrite each other's weights, and a class with no model of its own would train inside its own
backtest. Its .json is generated from the run's params SNAPSHOT, never re-read from the live
working file, so a mid-run edit cannot hand later folds a different entry gate.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_LIVE_live1(NNNC_A_Raw_Open):
    MODEL_TRAIN_TIMERANGE = "20210101-20261016"
    MODEL_SEED = 1
