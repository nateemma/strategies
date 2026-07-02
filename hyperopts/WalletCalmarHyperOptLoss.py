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

This loss rebuilds the equity curve by walking each trade's ORDER LEDGER and
marking the true holdings to market:

    equity(t) = cash(t) + Σ_pair holdings_pair(t) · price_pair(t)

where cash and holdings are updated at every fill (buy: cash down, holdings
up; sell: the reverse). Its Calmar (CAGR ÷ max drawdown) then reflects the
real portfolio equity path — including the intra-hold drawdown that the
trade-based Calmar misses.

WHY THE LEDGER: an earlier version held each trade's FINAL ``amount`` for the
whole life of the trade. For a heavily-adjusted strategy (skim / rebalance
trims the position to a few % of its peak size) that collapses the
mark-to-market swing and hides most of the drawdown — e.g. it reported a 4%
drawdown on a position path whose true drawdown was 36%, so hyperopt happily
chose buy-and-hold-the-winner configs. The order-ledger reconstruction fixes
this. If a trade has no usable order list, we fall back to the old
final-amount proxy.

Set WALLET_METRIC = "sharpe"/"sortino" style is not provided here — this file
targets Calmar (CAGR/maxDD). To deploy: copy to <freqtrade>/user_data/hyperopts/
    freqtrade hyperopt ... --hyperopt-loss WalletCalmarHyperOptLoss
"""
from datetime import datetime
from typing import Any, Dict

import numpy as np
from pandas import DataFrame, Series, Timestamp, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss

# Returned when the equity curve can't be built (no trades / no price data),
# so hyperopt steers away from these configs.
UNDESIRED_SOLUTION = 999.0


def _equity_from_orders(results: DataFrame, price: dict, idx, start_balance: float):
    """Cash + mark-to-market of the TRUE holdings, walking every trade's order
    ledger fill-by-fill. Returns the equity Series, or None if any trade lacks
    a usable order list (caller then falls back to the final-amount proxy)."""
    tz = idx.tz
    cash = Series(start_balance, index=idx, dtype="float64")
    holdings: dict = {}
    for _, tr in results.iterrows():
        pair = tr["pair"]
        if price.get(pair) is None:
            continue
        orders = tr.get("orders")
        if orders is None or len(orders) == 0:
            return None
        h = holdings.setdefault(pair, Series(0.0, index=idx, dtype="float64"))
        for o in orders:
            ts = o.get("order_filled_timestamp")
            if ts is None:
                continue  # unfilled order contributes nothing
            try:
                amt = float(o["amount"])
                rate = float(o["safe_price"])
                side = o["ft_order_side"]
            except (KeyError, TypeError, ValueError):
                return None  # unexpected order shape → use the proxy instead
            day = Timestamp(ts, unit="ms", tz="UTC").normalize()
            if tz is None:
                day = day.tz_localize(None)
            signed = amt if side == "buy" else -amt
            mask = idx >= day
            cash.loc[mask] -= signed * rate
            h.loc[mask] += signed
    equity = cash
    for pair, h in holdings.items():
        equity = equity + (h * price[pair]).fillna(0.0)
    return equity


def _equity_from_final_amount(results: DataFrame, price: dict, idx, start_balance: float):
    """Legacy proxy: hold each trade's FINAL amount for its whole life and book
    realized P&L at close. Under-captures drawdown for heavily-adjusted
    strategies; used only when order ledgers are unavailable."""
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
        open_mask = (idx >= od) & (idx < cd)
        equity.loc[open_mask] += amt * (p.loc[open_mask] - open_rate)
        equity.loc[idx >= cd] += pabs
    return equity


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

        # Primary: reconstruct from the order ledger (true holdings marked to
        # market). Fall back to the final-amount proxy only if orders are
        # unavailable / unexpectedly shaped.
        equity = _equity_from_orders(results, price, idx, start_balance)
        if equity is None:
            equity = _equity_from_final_amount(results, price, idx, start_balance)

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
