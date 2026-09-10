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


# ---------------------------------------------------------------------------
# Differentiable rollout (T024, T025) — the training path.
# ---------------------------------------------------------------------------

def bar_returns(open_px) -> "object":
    """Log return earned by a position filled at t+1 open and held to t+2 open.

    Decide at bar t using bars <= t; fill at t+1 open (FR-003). The position established
    at t+1 therefore earns log(open[t+2] / open[t+1]). Element i of the result aligns with
    a decision made at bar i.
    """
    import mlx.core as mx
    o = mx.array(open_px)
    return mx.log(o[2:]) - mx.log(o[1:-1])


def net_returns(w, open_px, cost_per_unit, *, volume_quote=None,
                soft_cap: bool = True, tau: float = 0.05):
    """Net-of-cost log return per bar for an exposure path (FR-004, FR-005).

    w[i] is the exposure decided at bar i. Costs are charged on the executed change in
    exposure, which the liquidity cap may limit; in training the cap is a soft-min so the
    gradient survives where it binds, in evaluation it is hard.

    Liquidity is assessed on the DECISION bar, matching BaseStrategy.custom_stake_amount.
    """
    import mlx.core as mx
    r = bar_returns(open_px)
    n = min(w.shape[-1], r.shape[-1])
    w, r = w[..., :n], r[..., :n]
    c = mx.array(cost_per_unit)[:n] if not isinstance(cost_per_unit, float) else cost_per_unit

    prev = mx.concatenate([mx.zeros(w.shape[:-1] + (1,)), w[..., :-1]], axis=-1)
    requested = mx.abs(w - prev)
    if volume_quote is not None:
        cap = LIQUIDITY_FRACTION * mx.array(volume_quote)[:n]
        executed = (soft_liquidity_cap(cap / LIQUIDITY_FRACTION, requested, tau=tau)
                    if soft_cap else mx.minimum(requested, cap))
    else:
        executed = requested
    return w * r - c * executed


def shift_features_leaky(x, k: int = 1):
    """Deliberately leak the future by k bars (FR-010, T025).

    Used ONLY to prove the learning path can exploit lookahead. SC-002 is a PAIRED test: on
    the same range and seed, the leaky policy must exceed the clean one by >=2.0 log units of
    cumulative net return. A fixed multiple of some other window's return does not work -
    a 4-year training range and a 90-day test range differ in buy-and-hold by orders of
    magnitude, so no single absolute bar is meaningful across both. A policy that cannot
    separate here has a broken gradient path, and any later null result would be meaningless.
    """
    import mlx.core as mx
    return mx.concatenate([x[k:], mx.repeat(x[-1:], k, axis=0)], axis=0)
