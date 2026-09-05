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
        targets = np.asarray(self.get_training_labels(dataframe))
        n = len(dataframe)
        if len(targets) != n:                      # labels can be short at the head
            padded = np.full(n, TradingAction.HOLD)
            padded[n - len(targets):] = targets
            targets = padded
        dataframe["predict_buy"] = np.where(targets == TradingAction.BUY, 1, 0)
        dataframe["predict_sell"] = np.where(targets == TradingAction.SELL, 1, 0)
        return dataframe["predict_buy"] > 0.5

    def get_model_path(self) -> str:
        """Reuse NNNC_MLX's weights — the model is loaded but never consulted."""
        return self.get_storage_location() + "NNNC_MLX/" + "NNNC_MLX.keras"
