# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Cross-model sizer study (research/xmodel_sizer/): `NNNC_CS_GBR` with the mixed score -- mean of the
GBR p_buy percentile and the 037 composite percentile (`_mix_sizer.MixSizerMixin`), the GBR twin of
`NNNC_CS_MIX`. Entries, weights (model key = cs_gbr's), stakes and threshold rule are cs_gbr's."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _mix_sizer import DATA, MixSizerMixin
from NNNC_CS_GBR import NNNC_CS_GBR


class NNNC_CS_GBR_MIX(MixSizerMixin, NNNC_CS_GBR):
    MIX_KNOTS = DATA / "mix_sizer_gbr"
