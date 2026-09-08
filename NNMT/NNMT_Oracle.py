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
        targets = np.asarray(labels["trading"] if isinstance(labels, dict) else labels)

        n = len(dataframe)
        if len(targets) != n:                      # labels can be short at the head
            padded = np.full(n, TradingAction.HOLD)
            padded[n - len(targets):] = targets
            targets = padded

        # ------------------------------------------------------------------ #
        # STRIP THE TRAINING AUGMENTATION (feature 006, T003).                #
        #                                                                    #
        # BaseNNStrategy.augment_training_signals runs inside                 #
        # get_trading_classes and its "Trick 1" DILATES every buy label onto  #
        # the TWO PRECEDING bars:                                            #
        #                                                                    #
        #     for i in range(1, 3): buys[valid_indices - i] = 1.0            #
        #                                                                    #
        # Those bars never satisfied the label's own condition, so reading    #
        # the augmented labels made this arm enter up to two bars EARLY --    #
        # measured at only 84.1% compliance with its own label, peaking at    #
        # 88.3% one bar FORWARD of the entry, with 96.6% having a genuine     #
        # label within the two-bar dilation window. Entering early against a  #
        # -2%..-4% stop is materially worse than entering on the signal, so   #
        # every figure this arm produced was biased LOW.                     #
        #                                                                    #
        # The augmentation is correct for TRAINING and is untouched there.    #
        # An oracle must enter only where the label genuinely holds, so the   #
        # augmented trading class is ANDed with the UNAUGMENTED buy signal.   #
        # get_train_buy_signals() is the pre-augmentation source (its CALLER  #
        # applies the augmentation), so the other label logic survives --     #
        # sells overriding buys, and the buy-followed-by-sell override --     #
        # while only the dilated bars are removed.                           #
        # ------------------------------------------------------------------ #
        raw = np.asarray(self.get_train_buy_signals(dataframe), dtype=float)
        if len(raw) != n:
            padded = np.zeros(n)
            padded[n - len(raw):] = raw
            raw = padded

        dataframe["predict_buy"] = np.where(
            (targets == TradingAction.BUY) & (raw > 0.5), 1, 0
        )
        dataframe["predict_sell"] = np.where(targets == TradingAction.SELL, 1, 0)
        return dataframe["predict_buy"] > 0.5
