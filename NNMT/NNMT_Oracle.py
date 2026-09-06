# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Oracle — PERFECT entry signal on NNMT's own label. A LOOKAHEAD CEILING.

Replaces the model's entry opinion with the TRAINING LABEL itself (lookahead,
deliberately). Everything else — the six swept entry guards, the ATR stoploss,
custom_exit, ROI, take-profit, the 21-day unclog — is inherited from
NNMT_MLX_Base unchanged. The gap between this and the control is therefore
exactly what a PERFECT model would buy on this family at this gate.

NOTHING MEASURED WITH THIS CLASS IS PROMOTABLE. Every figure it produces is a
ceiling, not an achievable return.

Why overriding ``get_entry_conditions`` is the right hook (research R3): NNMT
does not define it — it inherits BaseNNStrategy's, which calls
``get_predictions`` then ``process_predictions``, both of which NNMT overrides
to consume its six-task prediction dict. Overriding the entry conditions
therefore bypasses the whole multi-task predict path, ``apply_task_filters``
and the forward-peak-regressor filter. That is correct and intended: this arm
measures what a perfect ENTRY signal is worth, not what the task machinery does.

ONE DEVIATION FROM THE NNNC PATTERN, found by reading the code rather than
assumed: NNNC's ``get_training_labels`` returns a flat array of TradingAction
values, so NNNC_Oracle wraps it in ``np.asarray`` directly. NNMT's returns a
DICT of six per-task label arrays. The entry signal is the ``trading`` head —
the one carrying TradingAction.BUY/SELL/HOLD — so it must be selected by key.
Using the dict as-is would silently produce garbage.

The model is still loaded (and never consulted) via the inherited
``get_model_path``, which aliases NNMT_MLX's weights so this arm trains nothing.
"""

import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))

from Framework.BaseStrategy import TradingAction
from NNMT_MLX_Base import NNMT_MLX_Base


class NNMT_Oracle(NNMT_MLX_Base):

    def get_entry_conditions(self, dataframe: DataFrame):
        labels = self.get_training_labels(dataframe)
        # NNMT returns a per-task dict; the entry signal is the trading head.
        targets = np.asarray(labels["trading"] if isinstance(labels, dict) else labels)

        n = len(dataframe)
        if len(targets) != n:                      # labels can be short at the head
            padded = np.full(n, TradingAction.HOLD)
            padded[n - len(targets):] = targets
            targets = padded

        dataframe["predict_buy"] = np.where(targets == TradingAction.BUY, 1, 0)
        dataframe["predict_sell"] = np.where(targets == TradingAction.SELL, 1, 0)
        return dataframe["predict_buy"] > 0.5
