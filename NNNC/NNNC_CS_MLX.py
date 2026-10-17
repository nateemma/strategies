# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Cross-model sizer study (research/xmodel_sizer/): the MLX control `NNNC_FS_MLX_GR` weights as a
capital-neutral stake sizer on the MLX width book (`NNNC_W_MLX`), agree = p_buy above the fold's
0.5-share threshold -- the MLX twin of `NNNC_CS_ARO` / `NNNC_CS_GBR`."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _width_sizer import WidthSizerMixin
from NNNC_W_MLX import NNNC_W_MLX


class NNNC_CS_MLX(WidthSizerMixin, NNNC_W_MLX):
    AGREE_RULE = "p_buy"
