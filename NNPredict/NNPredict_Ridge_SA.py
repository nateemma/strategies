# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 (research R5): NNPredict_Ridge with the path-aware `stop_aware` target -- the realised stop
replaces the point return when the exit stack's ATR stop (with grace) would have fired first. The
untested option recorded in memory (project_nnpredict_stop_aware_target); everything else unchanged."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNPredict_Ridge import NNPredict_Ridge


class NNPredict_Ridge_SA(NNPredict_Ridge):
    target_mode = "stop_aware"
