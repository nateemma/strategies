# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Cap40 — CAPACITY probe: QUOTE_VOLUME_HEADROOM_MULT 40.0 (default 10).

`custom_stake_amount` caps a trade's stake at 1/MULT of the candle's quote
volume, so raising MULT is equivalent to trading a proportionally LARGER book
against the same market: MULT 20 ~ double the wallet, MULT 40 ~ quadruple it.
That gives the capacity curve without touching config, the wallet, or the model.

Subclasses NNMT_Ctl_s42 (the feature-005 control: label 0.007, take-profit
0.032) and aliases its weights, so this differs from its parent in the CAP
ALONE. Without the alias, get_model_path would key off this class's name and it
would train a model inside its own backtest.

Run twice — plain for the production exit configuration, and with
    --set-sell enable_exit_signal=true --set-sell exit_rvol_threshold=1.0
for the candidate — so both are measured on the same liquidity ladder.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_Ctl_s42 import NNMT_Ctl_s42


class NNMT_Cap40(NNMT_Ctl_s42):

    QUOTE_VOLUME_HEADROOM_MULT = 40.0

    def get_model_path(self) -> str:
        """Alias the control's weights — the cap is the only difference."""
        return self.get_storage_location() + "NNMT_Ctl_s42/" + "NNMT_Ctl_s42.keras"
