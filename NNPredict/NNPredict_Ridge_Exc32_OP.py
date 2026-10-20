# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 follow-up 3 (preregistration_followups.json): NNPredict_Ridge at the operating point, trained
on the `excursion` target at HORIZON 32 (8 h) -- the one horizon where the decile check ranked
outcomes the right way (excursion H=4: Spearman +0.61 at 8 h). New weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _operating_point import OperatingPointMixin
from NNPredict_Ridge import NNPredict_Ridge


class NNPredict_Ridge_Exc32_OP(OperatingPointMixin, NNPredict_Ridge):
    target_mode = "excursion"
    HORIZON = 32
