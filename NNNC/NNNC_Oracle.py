import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))

from Framework.BaseStrategy import TradingAction
from NNNC_MLX import NNNC_MLX


class NNNC_Oracle(NNNC_MLX):
    """PERFECT prediction on NNNC's own label — the ceiling of the execution stack.

    Replaces the model's entry opinion with the TRAINING LABEL itself (lookahead,
    deliberately). Everything else -- guard chain, ATR stoploss, custom_exit,
    ROI, take-profit, unclog -- is inherited from NNNC_MLX unchanged. The gap
    between this and production is therefore exactly what a perfect model would
    buy, and the gap between this and a profitable target is what the execution
    machinery throws away even when the signal is perfect.

    WHY NOT Debug/DebugNNStrategy: it carries its own populate_entry_trend whose
    guard block has DRIFTED from BaseStrategy -- it omits the `rvol` guard
    entirely and applies `atr_pct` outside the `entry_enable_guards` block. rvol
    is worth -3.42pp on W1 when removed, so an oracle built on that path is not
    comparable with production. Subclassing NNNC_MLX reuses the identical entry
    path and changes only the prediction source.
    """

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

    def get_model_path(self) -> str:
        """Reuse NNNC_MLX's weights — the model is loaded but never consulted."""
        return self.get_storage_location() + "NNNC_MLX/" + "NNNC_MLX.keras"
