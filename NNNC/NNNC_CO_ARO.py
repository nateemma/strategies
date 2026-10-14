# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Candidate-only training for ARO (feature 032; research/candidate_training/build_arms.py;
GENERATED): the model is trained only on windows whose last bar passes the gbb rule, and enters
only on such bars. Everything else is `NNNC_A_Raw_Open` (control `wf_aro_g24`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _candidate_training import CandidateTrainingMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_CO_ARO(CandidateTrainingMixin, NNNC_A_Raw_Open):
    pass
