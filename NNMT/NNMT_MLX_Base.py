# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_MLX_Base — the PRE-INHERITANCE control for feature 005.

"NNMT today" is not a valid control for anything. Its models on disk were
trained with a label gain threshold of 0.007 while the class now reads 0.020,
and its take-profit moved 0.032 -> 0.022. Both changes arrived by INHERITANCE
from work done for NNNC, neither was tested on the multi-task family, and
measuring that mixture would confound a stale model, a changed label and a
changed exit inside a single number (research R1).

This class pins the configuration as it stood BEFORE that work:

  * label gain/loss threshold 0.007   (class now says 0.020)
  * cexit_take_profit        0.032    (pinned in NNMT_MLX_Base.json)

Everything else is NNMT's own — the six swept entry guards, the 0.5 confidence
threshold, the 21-day maximum hold and the 2.0 exit-side relative-volume gate
are inherited from NNMT_MLX untouched, because NNMT overrides them itself and
they were never affected by the NNNC work.

``buy_params`` / ``sell_params`` are deliberately NOT redefined here: a fresh
literal silently drops every inherited key (recorded failure — a strategy ran
with its stoploss grace disabled, undetected). The two pinned values arrive
through NNMT_MLX_Base.json, which freqtrade merges OVER the class dicts.

``get_model_path`` aliases NNMT_MLX's weights. Without it BaseNNStrategy keys
the model directory off ``self.__class__.__name__`` and this class would train
its own model INSIDE the backtest, fitting the evaluation window and returning
different numbers with no error at all.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX import NNMT_MLX


class NNMT_MLX_Base(NNMT_MLX):

    # Pre-inheritance label. Every NNMT model on disk was trained against this
    # value; the framework default moved to 0.02 for NNNC's benefit.
    MIN_BUY_GAIN_THRESHOLD = 0.007
    MIN_SELL_LOSS_THRESHOLD = 0.007

    def get_model_path(self) -> str:
        """Reuse NNMT_MLX's weights — the control retrains nothing."""
        return self.get_storage_location() + "NNMT_MLX/" + "NNMT_MLX.keras"
