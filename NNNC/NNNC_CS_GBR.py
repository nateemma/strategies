# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""035 GBR extension (research/new_inputs/preregistration_gbr.json): the GBR control
`NNNC_FS_GBR_GR` weights as a capital-neutral stake sizer on the GBR width book (`NNNC_W_GBR`),
agree = p_buy above the fold's 0.5-share threshold -- the GBR twin of `NNNC_CS_ARO`. Its picks
give GBR's control-at-share reference; it is the S3 comparator if a GBR survivor wins S2."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _width_sizer import WidthSizerMixin
from NNNC_W_GBR import NNNC_W_GBR


class NNNC_CS_GBR(WidthSizerMixin, NNNC_W_GBR):
    AGREE_RULE = "p_buy"
