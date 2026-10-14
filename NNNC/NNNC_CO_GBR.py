# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Candidate-only training for GBR (feature 032; research/candidate_training/build_arms.py;
GENERATED): the model is trained only on windows whose last bar passes the gbb rule, and enters
only on such bars. Everything else is `NNNC_FS_GBR_GR` (control `fs_gbr_gr`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _candidate_training import CandidateTrainingMixin
from NNNC_FS_GBR_GR import NNNC_FS_GBR_GR


class NNNC_CO_GBR(CandidateTrainingMixin, NNNC_FS_GBR_GR):
    pass
