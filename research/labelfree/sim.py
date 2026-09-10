"""Vectorised trading simulator — accounting layer and replay path (T012-T016).

Contract: specs/009-labelfree-direct-policy/contracts/simulator.md

Scope note (FR-008a, clarification Q1=B): the simulator does NOT implement the production
exit stack (take-profit, adaptive stop, stop grace, model sell signal). Calibration replays
the entry/exit series the arbiter actually produced and validates the ACCOUNTING layer only:
fee application, fill timing, the liquidity cap, stake sizing and the portfolio layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import BPS, CostModel

LIQUIDITY_FRACTION = 0.10      # <=10% of a candle's quote volume (FR-005)

# Exits decided by the DEPLOYMENT stack fill at a stack-computed price (the stop level),
# not at a bar open. FR-008a puts that stack out of scope, so for these the replay takes
# the arbiter's realised exit rate as given data. Entries are always derived independently,
# which is what keeps fill timing under test. Measured 2026-09-10: with this rule the other
# 74 of 99 control trades reconcile at exactly 0.0 bps.
# 'roi' is freqtrade's built-in minimal_roi exit (BaseStrategy sets {"0": 0.03}); it prices
# at the ROI level, not a bar open, so it is stack-priced too. The strategy's own
# 'take_profit' and 'model_exit' are band-like and DO fill at the bar open - they reconcile
# at exactly 0.0 bps without any special case.
STACK_EXIT_REASONS = {"stop_loss", "trailing_stop_loss", "roi", "liquidation",
                      "emergency_exit"}


@dataclass(frozen=True)
class Portfolio:
    max_open_trades: int = 10
    stake_amount: float = 900.0
    tradable_balance_ratio: float = 0.99
    wallet: float = 10000.0


@dataclass
class Trade:
    pair: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    open_rate: float
    close_rate: float
    amount: float
    profit_abs: float
    exit_reason: str = "band"


@dataclass
class SimulatedRun:
    trades: list[Trade] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def total_profit(self) -> float:
        return float(sum(t.profit_abs for t in self.trades))

    def to_frame(self) -> pd.DataFrame:
        """Backtest-archive shape, so regime/wf pooling works unchanged (FR-007)."""
        return pd.DataFrame([{
            "pair": t.pair, "open_date": t.open_date, "close_date": t.close_date,
            "open_rate": t.open_rate, "close_rate": t.close_rate, "amount": t.amount,
            "profit_abs": t.profit_abs, "exit_reason": t.exit_reason,
        } for t in self.trades])


def liquidity_cap(volume_quote: float, requested: float,
                  fraction: float = LIQUIDITY_FRACTION) -> float:
    """Hard cap for evaluation (FR-005)."""
    return float(min(requested, fraction * volume_quote))


def soft_liquidity_cap(volume_quote, requested, fraction: float = LIQUIDITY_FRACTION,
                       tau: float = 0.05):
    """Smooth min for training, so the gradient survives where the cap binds (FR-005).

    A hard min zeroes the gradient on the capped branch, so the policy would never learn
    that requesting more than the book can fill is futile.
    """
    cap = fraction * np.asarray(volume_quote, dtype=float)
    r = np.asarray(requested, dtype=float)
    return -tau * np.logaddexp(-r / tau, -cap / tau)


def replay(trade_series: pd.DataFrame, ohlcv: dict[str, pd.DataFrame],
           costs: CostModel, portfolio: Portfolio,
           *, use_archive_rates: bool = False) -> SimulatedRun:
    """Replay a FIXED entry/exit series and recompute P&L from our own accounting.

    `trade_series` needs pair, open_date, close_date, amount (and, when
    `use_archive_rates`, open_rate/close_rate). Fill prices are otherwise derived
    independently from OHLCV at the bar stamped by the arbiter, which is what makes this
    a test rather than a restatement.
    """
    run = SimulatedRun()
    fee = costs.fee_per_side
    miss_open = miss_close = 0
    capped = 0
    stack_priced = 0

    for _, r in trade_series.iterrows():
        df = ohlcv.get(r["pair"])
        if df is None:
            continue
        od, cd = r["open_date"], r["close_date"]

        if use_archive_rates:
            o_rate, c_rate = float(r["open_rate"]), float(r["close_rate"])
        else:
            if od not in df.index:
                miss_open += 1
                continue
            o_rate = float(df.loc[od, "open"])
            reason = str(r.get("exit_reason", "band")).strip()
            if reason in STACK_EXIT_REASONS:
                c_rate = float(r["close_rate"])          # stack-priced, out of scope
                stack_priced += 1
            elif cd in df.index:
                c_rate = float(df.loc[cd, "open"])
            else:
                miss_close += 1
                c_rate = float(r["close_rate"])

        amount = float(r["amount"])
        # Liquidity is assessed on the DECISION bar (t), not the fill bar (t+1).
        # BaseStrategy.custom_stake_amount reads `dataframe.iloc[-1]` - the last analysed
        # candle, i.e. the bar the decision is made on - and caps stake at
        # quote_volume / QUOTE_VOLUME_HEADROOM_MULT. Measuring against the fill bar instead
        # reports 72% of the control's trades as over-cap when the true figure is 0%.
        if od in df.index:
            i = df.index.get_loc(od)
            bar = df.iloc[i - 1] if i > 0 else df.iloc[i]
            vq = float(bar["volume"]) * float(bar["close"])
            if amount * o_rate > LIQUIDITY_FRACTION * vq:
                capped += 1

        profit = amount * c_rate * (1 - fee) - amount * o_rate * (1 + fee)
        run.trades.append(Trade(r["pair"], od, cd, o_rate, c_rate, amount, profit,
                                str(r.get("exit_reason", "band"))))

    run.diagnostics = {
        "n_trades": len(run.trades),
        "missing_open_bar": miss_open,
        "missing_close_bar": miss_close,
        "liquidity_cap_would_bind": capped,
        "max_concurrent": max_concurrent(run.to_frame()) if run.trades else 0,
        "max_open_trades": portfolio.max_open_trades,
        "fee_per_side_bps": fee / BPS,
        "stack_priced_exits": stack_priced,
        "band_priced_exits": len(run.trades) - stack_priced,
    }
    return run


def max_concurrent(tr: pd.DataFrame) -> int:
    """Peak simultaneous open positions (FR-006)."""
    if tr.empty:
        return 0
    ev = ([(t, 1) for t in tr["open_date"]] + [(t, -1) for t in tr["close_date"]])
    # Closes sort BEFORE opens at the same timestamp: a trade closing on bar t frees its
    # slot for one opening on bar t, which is how freqtrade enforces max_open_trades.
    # Sorting opens first over-counts concurrency and would over-report against the cap.
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak
