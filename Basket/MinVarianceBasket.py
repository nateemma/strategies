"""
MinVarianceBasket — long-only minimum-variance weighting.

Weights that minimise portfolio variance from the trailing return covariance:
    w ∝ Σ⁻¹·1   (clipped to long-only, renormalised)
It leans into low-vol AND low-correlation coins, so it accounts for
diversification (unlike inverse-vol, which ignores correlations).

Covariance from few crypto samples is noisy, so a shrinkage term pulls Σ toward
its diagonal for a stable inverse (0 = raw sample cov, 1 = pure inverse-vol).
The most sophisticated variant — and the most estimation-sensitive; treat its
hyperopt results with extra out-of-sample suspicion.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class MinVarianceBasket(BasketStrategy):

    startup_candle_count = 250  # covers cov_lookback (max 200) + BB window

    cov_lookback = IntParameter(30, 200, default=60, space="buy")
    # Shrink Σ toward its diagonal: 0 = raw cov, 1 = diagonal (→ inverse-vol).
    cov_shrinkage = DecimalParameter(0.0, 0.9, default=0.3, decimals=2, space="buy")

    def _fallback_weight(self) -> float:
        return (1.0 - self.cash_target_weight.value) / self._n_coins()

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        rm = self._return_matrix(self.cov_lookback.value, current_time)
        if rm is None or rm.shape[1] < 2 or pair not in rm.columns or len(rm) < 3:
            return self._fallback_weight()  # not enough coins/data → equal weight

        pairs = list(rm.columns)
        cov = np.cov(rm.values, rowvar=False)
        # Shrink toward the diagonal, then add a tiny ridge for invertibility.
        s = self.cov_shrinkage.value
        cov = (1.0 - s) * cov + s * np.diag(np.diag(cov))
        cov += np.eye(cov.shape[0]) * 1e-10
        try:
            w = np.linalg.inv(cov) @ np.ones(len(pairs))
        except np.linalg.LinAlgError:
            return self._fallback_weight()

        w = np.clip(w, 0.0, None)  # long-only
        if w.sum() <= 0:
            return self._fallback_weight()
        w = w / w.sum()
        weights = dict(zip(pairs, w))
        return (1.0 - self.cash_target_weight.value) * float(weights.get(pair, 0.0))
