# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""ATR-barrier study, arm B (research/tb_model/): arm A plus the 037 composite's 7 member columns as
pre-normalised inputs (`_tb_inputs.TbInputMixin`, cache built by research/tb_model/inputs.py). The
control's scaler is read unchanged for ARO's own columns."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _tb_inputs import TbInputMixin
from NNNC_CS_TBA import NNNC_CS_TBA


class NNNC_CS_TBB(TbInputMixin, NNNC_CS_TBA):
    TB_INPUTS_SHA = "d84f8123ea593215"
    TB_INPUT_COLUMNS = ("tbi_temacross_tema_long_close", "tbi_ehlers_dc_dc_upper_close",
                        "tbi_ehlers_lw_lw_trend_buy_close", "tbi_raw8_ret1", "tbi_uo_uo",
                        "tbi_raw8_log_hc", "tbi_ehlersbase_cg_slope")
    include_list = list(NNNC_CS_TBA.include_list) + list(TB_INPUT_COLUMNS)
    pre_normalized_columns = list(NNNC_CS_TBA.pre_normalized_columns) + list(TB_INPUT_COLUMNS)
