# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 threshold amendment, option 2a: NNPredict_Ridge at the operating point, trained on the
`excursion` target (the dominant signed move within the next HORIZON candles: what a trade could
capture) instead of the point return. New weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _operating_point import OperatingPointMixin
from NNPredict_Ridge import NNPredict_Ridge


class NNPredict_Ridge_Exc_OP(OperatingPointMixin, NNPredict_Ridge):
    target_mode = "excursion"
