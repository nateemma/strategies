import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))

from Framework.BaseStrategy import TradingAction
from NNNC_MLX import NNNC_MLX


class NNNC_Ora_Prec(NNNC_MLX):
    """PARTIAL oracle — how accurate must a model be before the label pays?

    The gate showed the 0.02 label has a +26.54% ceiling guards-off while the
    trained model delivers -12.71%. That gap is either a model problem (the
    required accuracy is reachable) or a data problem (it is not). Nothing so
    far distinguishes them, and the answer decides where effort should go.

    This arm degrades perfect prediction to a chosen PRECISION and measures the
    return. Sweeping precision maps the requirement directly: the crossing point
    where return turns positive is the accuracy a real model must beat.

    Method: take the label's BUY bars. Keep a random `ORACLE_PRECISION` fraction
    of them; replace the remainder with randomly chosen non-BUY bars, so the
    TRADE COUNT is held constant and only the hit rate changes. Precision is the
    metric that matters for trading — a strategy needs some good trades, not all
    of them — and holding count fixed keeps cost and exposure comparable across
    the sweep.

    Seeded, so an arm is reproducible and its trade hash is meaningful.

    LOOKAHEAD CEILING arm. Never promotable.
    """

    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02

    ORACLE_PRECISION = 1.0
    ORACLE_SEED = 17

    # Fraction of the (possibly corrupted) BUY set to keep. Used to match a
    # comparator to the REAL model's trade volume: headroom measured against a
    # 1333-trade oracle when the model takes 431 trades would confound
    # selection quality with how many slots each arm consumes.
    ORACLE_SUBSAMPLE = 1.0

    def get_entry_conditions(self, dataframe: DataFrame):
        targets = np.asarray(self.get_training_labels(dataframe))
        n = len(dataframe)
        if len(targets) != n:
            padded = np.full(n, TradingAction.HOLD)
            padded[n - len(targets):] = targets
            targets = padded

        buy = (targets == TradingAction.BUY)
        p = float(self.ORACLE_PRECISION)
        if p < 1.0:
            rng = np.random.default_rng(self.ORACLE_SEED)
            idx_true = np.flatnonzero(buy)
            idx_false = np.flatnonzero(~buy)
            n_keep = int(round(p * len(idx_true)))
            n_swap = len(idx_true) - n_keep
            keep = rng.choice(idx_true, size=n_keep, replace=False) if n_keep else np.array([], int)
            add = (rng.choice(idx_false, size=min(n_swap, len(idx_false)), replace=False)
                   if n_swap else np.array([], int))
            buy = np.zeros(n, dtype=bool)
            buy[keep] = True
            buy[add] = True

        if float(self.ORACLE_SUBSAMPLE) < 1.0:
            rng2 = np.random.default_rng(self.ORACLE_SEED + 1000)
            idx = np.flatnonzero(buy)
            keep = rng2.choice(idx, size=int(round(float(self.ORACLE_SUBSAMPLE) * len(idx))),
                               replace=False)
            buy = np.zeros(n, dtype=bool)
            buy[keep] = True

        dataframe["predict_buy"] = buy.astype(int)
        dataframe["predict_sell"] = np.where(targets == TradingAction.SELL, 1, 0)
        return dataframe["predict_buy"] > 0.5

    def get_model_path(self) -> str:
        return self.get_storage_location() + "NNNC_MLX/NNNC_MLX.keras"
