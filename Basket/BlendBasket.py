"""
BlendBasket — compose an EXPOSURE rule with a SELECTION rule.

A basket allocation is really two separate decisions, and the other variants
answer both at once (e.g. CPPI decides exposure but splits it equally). Blend
factors them apart and lets you pick each independently:

    target_weight(coin) = exposure_fraction × selection_weight(coin)

  * EXPOSURE — how much of the portfolio to deploy (rest is cash):
      "cppi"      cushion above a floor × multiplier      (defends capital)
      "voltarget" scale to hit a target portfolio vol     (de-risks in turbulence)
      "fixed"     a constant `fixed_exposure`             (always deployed)
  * SELECTION — how to distribute that exposure across coins (weights sum to 1):
      "momentum"    ∝ excess trailing return, gated to cash below threshold
      "inverse_vol" ∝ 1/vol
      "min_variance" ∝ Σ⁻¹·1 (long-only)
      "equal"       1/n

So e.g. exposure="cppi" + selection="momentum" = "ride the strongest coins but
cut exposure to protect a floor" — the best-of-both a hard regime switch is
reaching for, but smooth (no discrete flips) and with fewer thresholds.

Cash is the dynamic residual `1 − exposure_fraction`, so `cash_target_weight`
is unused here (fixed, not optimised).

STILL LONG-ONLY: no exposure/selection combo makes a falling market positive —
the win is better risk-adjusted return across a full cycle, using the right
tool per regime automatically.

NOTE: this class exposes a large hyperopt space (both mode switches + every
mode's params). For efficiency, fix `exposure_mode`/`selection_mode` to the
combo you want and tune the rest, or run more epochs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class BlendBasket(BasketStrategy):

    startup_candle_count = 250  # covers the longest lookback (cov/mom = 200) + BB

    # Cash is the residual of the exposure decision → not a fixed reserve.
    cash_target_weight = 0.0

    # --- the two composition switches ---------------------------------
    exposure_mode = CategoricalParameter(
        ["cppi", "voltarget", "fixed"], default="cppi", space="buy"
    )
    selection_mode = CategoricalParameter(
        ["momentum", "inverse_vol", "min_variance", "equal"],
        default="momentum", space="buy",
    )

    # --- exposure params ----------------------------------------------
    cppi_floor_mode = CategoricalParameter(
        ["ratchet", "fixed"], default="ratchet", space="buy"
    )
    cppi_floor_ratio = DecimalParameter(0.50, 0.95, default=0.80, decimals=2, space="buy")
    cppi_multiplier = DecimalParameter(1.0, 6.0, default=3.0, decimals=1, space="buy")
    target_vol = DecimalParameter(0.2, 1.5, default=0.6, decimals=2, space="buy")  # annualised
    fixed_exposure = DecimalParameter(0.3, 1.0, default=0.8, decimals=2, space="buy")

    # --- selection params ---------------------------------------------
    vol_lookback = IntParameter(10, 100, default=30, space="buy")   # inverse_vol + voltarget
    mom_lookback = IntParameter(20, 200, default=60, space="buy")
    mom_threshold = DecimalParameter(-0.10, 0.20, default=0.0, decimals=2, space="buy")
    cov_lookback = IntParameter(30, 200, default=60, space="buy")
    cov_shrinkage = DecimalParameter(0.0, 0.9, default=0.3, decimals=2, space="buy")

    # CPPI reference state (lazily initialised)
    _hwm: float
    _initial_pv: float

    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # Causal per-coin stats the selection rules read (via _cross_section).
        dataframe["vol"] = (
            dataframe["close"].pct_change()
            .rolling(self.vol_lookback.value, min_periods=self.vol_lookback.value)
            .std()
        )
        dataframe["mom"] = dataframe["close"].pct_change(self.mom_lookback.value)
        return dataframe

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        exposure = self._exposure(current_time)
        if exposure <= 0:
            return 0.0
        return exposure * self._selection(current_time).get(pair, 0.0)

    # ---- exposure: how much to deploy --------------------------------
    def _exposure(self, current_time: datetime) -> float:
        mode = self.exposure_mode.value
        if mode == "cppi":
            return self._cppi_fraction()
        if mode == "voltarget":
            return self._voltarget_exposure(current_time)
        return float(self.fixed_exposure.value)

    def _cppi_fraction(self) -> float:
        pv = self._portfolio_value()
        if pv <= 0:
            return 0.0
        if not hasattr(self, "_initial_pv"):
            self._initial_pv = pv
            self._hwm = pv
        if pv > self._hwm:
            self._hwm = pv
        if self.cppi_floor_mode.value == "ratchet":
            floor = self.cppi_floor_ratio.value * self._hwm
        else:
            floor = self.cppi_floor_ratio.value * self._initial_pv
        cushion = max(pv - floor, 0.0)
        return min(1.0, max(0.0, self.cppi_multiplier.value * cushion / pv))

    def _voltarget_exposure(self, current_time: datetime) -> float:
        rm = self._return_matrix(self.vol_lookback.value, current_time)
        if rm is None or rm.empty or rm.shape[1] == 0:
            return 0.0
        per_period = float(rm.mean(axis=1).std())
        if per_period <= 0:
            return 1.0
        annualised = per_period * (self._periods_per_year() ** 0.5)
        return min(1.0, max(0.0, self.target_vol.value / annualised))

    # ---- selection: how to distribute it (weights sum to 1) ----------
    def _selection(self, current_time: datetime) -> dict[str, float]:
        mode = self.selection_mode.value
        if mode == "momentum":
            return self._sel_momentum(current_time)
        if mode == "inverse_vol":
            return self._sel_inverse_vol(current_time)
        if mode == "min_variance":
            return self._sel_min_variance(current_time)
        return self._sel_equal()

    def _sel_equal(self) -> dict[str, float]:
        wl = self.dp.current_whitelist()
        n = max(1, len(wl))
        return {p: 1.0 / n for p in wl}

    def _sel_momentum(self, current_time: datetime) -> dict[str, float]:
        moms = self._cross_section("mom", current_time)
        thr = self.mom_threshold.value
        strong = {p: (m - thr) for p, m in moms.items() if m > thr}
        total = sum(strong.values())
        if not total:
            return {}  # nobody qualifies → all cash
        return {p: v / total for p, v in strong.items()}

    def _sel_inverse_vol(self, current_time: datetime) -> dict[str, float]:
        vols = self._cross_section("vol", current_time)
        inv = {p: 1.0 / v for p, v in vols.items() if v and v > 0}
        total = sum(inv.values())
        if not total:
            return {}
        return {p: v / total for p, v in inv.items()}

    def _sel_min_variance(self, current_time: datetime) -> dict[str, float]:
        rm = self._return_matrix(self.cov_lookback.value, current_time)
        if rm is None or rm.shape[1] < 2 or len(rm) < 3:
            return self._sel_equal()
        pairs = list(rm.columns)
        cov = np.cov(rm.values, rowvar=False)
        s = self.cov_shrinkage.value
        cov = (1.0 - s) * cov + s * np.diag(np.diag(cov))
        cov += np.eye(cov.shape[0]) * 1e-10
        try:
            w = np.linalg.inv(cov) @ np.ones(len(pairs))
        except np.linalg.LinAlgError:
            return self._sel_equal()
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            return {}
        w = w / w.sum()
        return dict(zip(pairs, w.tolist()))
