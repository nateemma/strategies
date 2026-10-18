# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study (research/tb_model/preregistration_dir.json): NNNC_CS_TBB's weights with the
direction readout (`_dir_score.DirScoreMixin`) as the width-book sizer's agree score."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _dir_score import DirScoreMixin
from NNNC_CS_TBB import NNNC_CS_TBB


class NNNC_CS_TBB_D(DirScoreMixin, NNNC_CS_TBB):
    pass
