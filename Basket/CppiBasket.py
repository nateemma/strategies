"""
CppiBasket — Constant Proportion Portfolio Insurance.

The OPPOSITE stance to constant-mix: it BUYS strength and SELLS weakness to
protect a floor. Trend-following; outperforms in sustained trends (rising
markets let the cushion grow, so exposure grows) and protects capital in
drawdowns (as value falls toward the floor, exposure is cut). Underperforms
constant-mix in choppy markets (it can buy high / sell low on whipsaws).

Mechanics
---------
    cushion        = portfolio_value - floor
    risky_exposure = m * cushion            (capped to [0, portfolio_value])
    risky_fraction = risky_exposure / portfolio_value      in [0, 1]
    per-coin target = risky_fraction / n_coins
    cash weight     = 1 - risky_fraction    (DYNAMIC — see note below)

Cash handling vs constant-mix
-----------------------------
Constant-mix holds a FIXED cash_target_weight. CPPI's cash is a DYNAMIC
residual of the cushion: when the portfolio is well above the floor, cash
shrinks (more deployed); as it falls toward the floor, cash grows (de-risked).
So this class IGNORES the base `cash_target_weight` and lets cash emerge as
`1 - risky_fraction`. The shared cash-bucket handling still applies — cash is
just undeployed stake — only the TARGET is computed differently here.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
from freqtrade.strategy import CategoricalParameter, DecimalParameter

sys.path.append(str(Path(__file__).parent))
from BasketStrategy import BasketStrategy


class CppiBasket(BasketStrategy):

    # =====================================================================
    # TUNABLE PARAMETERS  (CPPI specific; shared ones are in base)
    # =====================================================================

    # Floor definition:
    #   "ratchet" — floor = cppi_floor_ratio * high-water-mark (locks in gains)
    #   "fixed"   — floor = cppi_floor_ratio * initial portfolio value
    # CPPI derives cash as a DYNAMIC residual (1 - risky_fraction) and never
    # reads cash_target_weight, so we shadow the base hyperopt Parameter with a
    # plain attr — this removes it from CPPI's hyperopt search space (it would
    # otherwise be optimised over a value that has no effect).
    cash_target_weight = 0.20

    cppi_floor_mode = CategoricalParameter(
        ["ratchet", "fixed"], default="ratchet", space="buy"
    )
    cppi_floor_ratio = DecimalParameter(
        0.50, 0.95, default=0.80, decimals=2, space="buy"
    )  # protect this fraction of the reference value

    # Multiplier m: higher = more aggressive risk-on per unit of cushion.
    # m=3 with an 80% floor implies full deployment once the cushion reaches
    # ~6.7% of portfolio value (0.20/3), and de-risking below that.
    cppi_multiplier = DecimalParameter(1.0, 6.0, default=3.0, decimals=1, space="buy")

    # rebalance_band (inherited) applies to the CPPI target the same way.

    # instance state (lazily initialised)
    _hwm: float
    _initial_pv: float

    def _update_reference(self, pv: float) -> None:
        # LOOKAHEAD: pv is the CURRENT wallet value — no future information.
        if not hasattr(self, "_initial_pv"):
            self._initial_pv = pv
            self._hwm = pv
        if pv > getattr(self, "_hwm", pv):
            self._hwm = pv  # ratchet the high-water mark upward only

    def _floor_value(self) -> float:
        if self.cppi_floor_mode.value == "ratchet":
            return self.cppi_floor_ratio.value * self._hwm
        return self.cppi_floor_ratio.value * self._initial_pv

    def _risky_fraction(self) -> float:
        # Mark-to-market portfolio value (NOT get_total_stake_amount, which is
        # the ~constant stake budget — the CPPI cushion must track live value).
        pv = self._portfolio_value()
        if pv <= 0:
            return 0.0
        self._update_reference(pv)
        cushion = max(pv - self._floor_value(), 0.0)
        frac = (self.cppi_multiplier.value * cushion) / pv
        return min(1.0, max(0.0, frac))  # cap 100%, floor 0%

    def get_target_weight(
        self, pair: str, current_time: datetime, dataframe: DataFrame
    ) -> float:
        # Risky exposure split equally across the eligible coins; the rest is
        # the dynamic cash residual.
        return self._risky_fraction() / self._n_coins()
