# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ARO volume-filter retrain (research/aro_volume/): `NNNC_CS_ARO` with the ARO weights retrained on
windows whose last bar has volume > 0. 24% of ARO's BUY positives sat on zero-volume (gap-filled)
bars no strategy can trade. The filter is data (feature 032's TRAIN_ROW_FILTER, applied by
Framework/TrainingEngine to the raw frame after windowing) and part of the model key, so the weights
are new; entries, stakes and the 0.5-share threshold rule are cs_aro's."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_AROV(NNNC_CS_ARO):
    TRAIN_ROW_FILTER = (("volume", "gt", 0.0),)
    MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)
