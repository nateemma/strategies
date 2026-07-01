"""
WalletCalmarHyperOptLoss

Calmar ratio computed on a RECONSTRUCTED mark-to-market equity curve, for
hold-and-rebalance / basket strategies.

Why reconstruct?
----------------
Freqtrade only captures the true daily wallet-balance curve in BACKTEST
runmode (``Backtesting._capture_wallet`` early-returns when runmode !=
BACKTEST), so it is NOT available to a loss function during hyperopt —
``backtest_stats["wallet_stats"]`` is empty there. And the closed-trade
metrics are degenerate for a basket (one trade per coin, all closing at the
final force-exit → phantom ~0 drawdown). So neither the built-in wallet
metrics nor the built-in trade metrics work during hyperopt.

This loss rebuilds an approximate equity curve from the trades and the price
data in ``processed``:

    equity(t) = starting_balance
              + Σ realized profit of trades closed before t
              + Σ over trades OPEN at t of  amount · (price(t) − open_rate)

i.e. cash plus the mark-to-market value of open positions. Its Calmar
(CAGR ÷ max drawdown) then reflects the real portfolio equity path — the
intra-hold drawdown that the trade-based Calmar misses.

APPROXIMATION: it holds each trade's FINAL ``amount`` for the whole life of
the trade and folds all realized P&L in at close, so it ignores the
intermediate size changes from rebalancing adjustments. For a slow basket
(few adjustments) this is close; for a heavily-adjusted strategy treat it as
a proxy and confirm the winner with a normal backtest (where the TRUE wallet
Calmar is computed).

Set WALLET_METRIC = "sharpe"/"sortino" style is not provided here — this file
targets Calmar (CAGR/maxDD). To deploy: copy to <freqtrade>/user_data/hyperopts/
    freqtrade hyperopt ... --hyperopt-loss WalletCalmarHyperOptLoss
"""
from datetime import datetime
from typing import Any, Dict

import numpy as np
from pandas import DataFrame, Series, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss

# Returned when the equity curve can't be built (no trades / no price data),
# so hyperopt steers away from these configs.
UNDESIRED_SOLUTION = 999.0


class WalletCalmarHyperOptLoss(IHyperOptLoss):
    """Optimise Calmar on a reconstructed mark-to-market equity curve."""

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        if results is None or len(results) == 0:
            return UNDESIRED_SOLUTION

        start_balance = float(
            backtest_stats.get("starting_balance")
            or config.get("dry_run_wallet")
            or 0.0
        )
        if start_balance <= 0:
            return UNDESIRED_SOLUTION

        # Daily equity grid over the backtest span.
        idx = date_range(start=min_date, end=max_date, freq="1D", normalize=True)
        if len(idx) < 3:
            return UNDESIRED_SOLUTION

        # Per-pair daily close (forward-filled) from the processed dataframes.
        price: Dict[str, Series] = {}
        for pair, df in (processed or {}).items():
            if df is None or len(df) == 0 or "close" not in df:
                continue
            s = df.set_index("date")["close"] if "date" in df else df["close"]
            price[pair] = s.reindex(idx, method="ffill")

        if not price:
            return UNDESIRED_SOLUTION  # price data cleared → can't reconstruct

        equity = Series(start_balance, index=idx, dtype="float64")
        for _, tr in results.iterrows():
            p = price.get(tr["pair"])
            if p is None:
                continue
            od = tr["open_date"].normalize()
            cd = tr["close_date"].normalize()
            amt = float(tr["amount"])
            open_rate = float(tr["open_rate"])
            pabs = float(tr["profit_abs"])
            # unrealized mark-to-market while the position is open
            open_mask = (idx >= od) & (idx < cd)
            equity.loc[open_mask] += amt * (p.loc[open_mask] - open_rate)
            # realized P&L booked from close onward
            equity.loc[idx >= cd] += pabs

        equity = equity.ffill().fillna(start_balance)

        # Calmar = CAGR / max drawdown of the reconstructed curve.
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = abs(float(drawdown.min()))
        if max_dd < 1e-6:
            max_dd = 1e-6  # avoid divide-by-zero blow-up

        days = max((max_date - min_date).days, 1)
        cagr = (equity.iloc[-1] / start_balance) ** (365.0 / days) - 1.0
        if not np.isfinite(cagr):
            return UNDESIRED_SOLUTION

        calmar = cagr / max_dd
        return -float(calmar)
