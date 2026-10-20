# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 combined sizer (research/nnpredict/mix.py, preregistration_mix.json): `NNNC_CS_ARO` with the mixed
score -- mean of ARO's p_buy percentile and NNPredict's excursion-H32 Ridge predicted-move percentile
(`_mix_sizer.MixSizerMixin`; the exc32 score from user_data/data/nnp_exc32_sizer tables). Entries,
weights (model key = cs_aro's), stakes and the 0.5-share threshold rule are cs_aro's."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _mix_sizer import DATA, MixSizerMixin
from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_ARO_NNP(MixSizerMixin, NNNC_CS_ARO):
    CMP_TABLES = DATA / "nnp_exc32_sizer"
    MIX_KNOTS = DATA / "mix_sizer_nnp_exc32"
