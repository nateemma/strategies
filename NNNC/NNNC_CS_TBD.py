# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study, direction-only arm (research/tb_model/preregistration_dironly.json): arm A
(NNNC_CS_TBA: ARO's 24 inputs, tbatr label, volume > 0) trained only on windows whose last bar hit a
barrier (`_dir_only.DirOnlyMixin`), used as the width-book sizer on p_buy."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _dir_only import DirOnlyMixin
from NNNC_CS_TBA import NNNC_CS_TBA


class NNNC_CS_TBD(DirOnlyMixin, NNNC_CS_TBA):
    TRAIN_ROW_FILTER = (("volume", "gt", 0.0), ("%tb_dir_keep", "gt", 0.5))
