import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))

from Framework.BaseStrategy import TradingAction
from NNNC_BreakoutOpen import NNNC_BreakoutOpen


class NNNC_BrkOra005(NNNC_BreakoutOpen):
    """ORACLE on the BREAKOUT host — was feature 003's tail reasoning right?

    Feature 003 raised the breakout label's gain threshold, improved its
    reward:risk from 0.57 to 1.28, and realised return did NOT follow. It could
    not tell whether the label was wrong or the model simply could not learn it,
    because it never measured its ceiling.

    This arm measures that ceiling. If the breakout ceiling ALSO improves with
    threshold, 003's economic reasoning was sound and its null was a learnability
    or absorption failure, not a refutation of the idea.

    LOOKAHEAD CEILING arm. Never promotable. Threshold here: 0.005.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.005
    MIN_SELL_LOSS_THRESHOLD = 0.005

    def get_entry_conditions(self, dataframe: DataFrame):
        targets = np.asarray(self.get_training_labels(dataframe))
        n = len(dataframe)
        if len(targets) != n:
            padded = np.full(n, TradingAction.HOLD)
            padded[n - len(targets):] = targets
            targets = padded
        dataframe["predict_buy"] = np.where(targets == TradingAction.BUY, 1, 0)
        dataframe["predict_sell"] = np.where(targets == TradingAction.SELL, 1, 0)
        return dataframe["predict_buy"] > 0.5

    def get_model_path(self) -> str:
        """Reuse the host's weights — loaded, never consulted."""
        return self.get_storage_location() + "NNNC_BreakoutOpen/NNNC_BreakoutOpen.keras"
