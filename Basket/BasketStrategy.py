"""
BasketStrategy — shared base for slow, portfolio-level basket rebalancing.

This is NOT a fast trading scheme. It holds a fixed basket of coins (the
whitelist) plus a stablecoin cash bucket, and nudges the basket back toward
target weights infrequently and deliberately. Entries/exits are portfolio
maintenance, not signals.

Design notes / framework divergence
-----------------------------------
This inherits Freqtrade's ``IStrategy`` DIRECTLY, not the repo's ML
``BaseStrategy``. BaseStrategy is built for fast ML trading: trailing stop,
ROI, custom_stoploss and custom_exit are all tuned to *close* positions. A
basket is meant to be *held and maintained*, so we start from a clean
interface and neutralise every auto-exit path instead of fighting inherited
ones. All the exit machinery below is deliberately disabled.

Freqtrade version: targets 2026.4-dev (INTERFACE_VERSION 3). Callback
signatures used: custom_stake_amount, adjust_trade_position (negative stake
== partial trim), bot_loop_start.

No-lookahead discipline
-----------------------
Every indicator here is causal:
  * The Bollinger mid-band is an SMA over a trailing window (qtpylib /
    talib), so row t only uses rows <= t.
  * In callbacks we read ``dp.get_analyzed_dataframe`` and take ``.iloc[-1]``
    — in backtest Freqtrade truncates that frame to the current candle, so
    ``.iloc[-1]`` is the latest CLOSED bar, never a future one.
  * Portfolio value comes from ``self.wallets`` (present state only).
Spots that need care are flagged inline with ``# LOOKAHEAD:``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    BooleanParameter,
    DecimalParameter,
    IntParameter,
)

log = logging.getLogger(__name__)


class BasketStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # 4h suits a coarse-timing, slow rebalancing scheme: the cadence gate
    # throttles actual trades to ~daily, and adjust_trade_position is called
    # once per open trade PER CANDLE, so a finer timeframe (e.g. 15m) just
    # multiplies Freqtrade's per-call Trade deepcopy overhead for no benefit.
    timeframe = "4h"
    can_short = False
    process_only_new_candles = True

    # ---- exits neutralised: the basket is HELD and maintained ----------
    # ROI never triggers (10 000%). Stoploss effectively disabled so a coin
    # we intend to keep rebalancing is never fully closed by a stop. Trailing
    # off. Exit signals off — positions leave the basket only via a rebalance
    # trim in adjust_trade_position(), never via an exit signal / ROI / stop.
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = False
    use_custom_stoploss = False

    # Rebalancing is done by scaling positions, not by opening new trades.
    position_adjustment_enable = True

    # Must exceed bb_period. Raise this if you raise bb_period.
    startup_candle_count = 60

    # =====================================================================
    # TUNABLE PARAMETERS  (hyperopt-optimisable; slow-scheme defaults)
    #
    # All in the 'buy' space, so `hyperopt --spaces buy` optimises the whole
    # scheme in one pass. Optimised values are written to <Strategy>.json and
    # override these defaults. Non-hyperopt backtests use the defaults below.
    #
    # CAVEAT: bb_period / bb_std are used in populate_indicators(), which
    # Freqtrade computes ONCE unless you pass `--analyze-per-epoch`. Without
    # that flag they stay at their defaults during hyperopt; the callback
    # params (band, cadence, weights, gate, CPPI, runaway) vary every epoch.
    # =====================================================================

    # --- Basket weights -------------------------------------------------
    # Fraction of total portfolio held as undeployed stablecoin cash. The
    # cash bucket is NOT a position — it's simply stake we never deploy.
    # NOTE: CPPI overrides weight logic entirely and treats cash as a
    # DYNAMIC residual, so this only governs the constant-mix path.
    cash_target_weight = DecimalParameter(0.0, 0.5, default=0.20, decimals=2, space="buy")

    # Per-coin target weight OVERRIDE. Left as a plain attr (not hyperopt):
    # None => split (1 - cash) equally across the whitelist. Set explicitly to
    # weight coins unequally.
    target_weight_per_coin: float | None = None

    # Hard per-position concentration cap (plain attr, not hyperopt). None =>
    # off. Set e.g. 0.20 so no single coin may exceed 20% of portfolio value:
    # a runaway winner is trimmed back to the cap IMMEDIATELY, bypassing the
    # rebalance cadence and BB timing gate. Bounds single-coin tail risk when a
    # coin runs hard between deliberate rebalances (the normal band-trim can lag).
    max_position_weight: float | None = None

    # --- Bollinger timing gate -----------------------------------------
    # The mid-BB is an EXECUTION-TIMING filter, not a second signal. The
    # drift band is the primary trigger; the BB only decides whether a
    # triggered trade is allowed to fire right now:
    #   * add / initial entry  → only when price is BELOW mid-BB
    #   * trim                 → only when price is ABOVE mid-BB
    # Only the MID band is used (an SMA of typical price), so there is NO
    # std-dev parameter — band width would have no effect on this strategy.
    # (bb_period is an indicator param — see --analyze-per-epoch caveat above)
    bb_period = IntParameter(10, 50, default=20, space="buy")

    # Master switch for the BB timing gate. True = gated (deliberate timing).
    # False = "raw band-rebalance" arm (entries fire ASAP, rebalances ignore BB).
    use_bb_gate = BooleanParameter(default=True, space="buy")

    # How far BELOW mid-BB price must be to count as a "dip" for BUYING (both
    # initial entry and rebalance top-ups). 0.0 (default) = plain `close < mid`,
    # which is loose (true ~half the time). e.g. 0.02 requires price 2% under
    # mid-BB — a more selective entry/add, at the cost of filling more slowly.
    # Sell-side (trim) gating is unaffected.
    entry_band = DecimalParameter(0.0, 0.10, default=0.0, decimals=3, space="buy")

    # --- Band rebalancing ----------------------------------------------
    # Only act when a coin's weight drifts more than this from target.
    # Wide by default so we trade rarely.
    rebalance_band = DecimalParameter(0.01, 0.15, default=0.05, decimals=2, space="buy")

    # --- Cadence --------------------------------------------------------
    # Check rebalances at most this often (per pair), in hours. Not every candle.
    rebalance_interval_hours = IntParameter(6, 168, default=24, space="buy")

    # --- Profit-skim (INCOME overlay) -----------------------------------
    # When ON, a fraction of each new equity high is moved to a RESERVED cash
    # bucket that is NEVER redeployed ("taking chips off the table"). In LIVE
    # you would periodically withdraw that reserved amount as income; in
    # backtest freqtrade can't withdraw mid-run, so it sits as idle cash —
    # which lowers exposure and drawdown but also caps upside (banked money
    # doesn't compound, and earns 0). This is a decumulation/income overlay,
    # NOT alpha. Off by default (existing variants behave exactly as before).
    profit_skim_enable = BooleanParameter(default=False, space="buy")
    skim_fraction = DecimalParameter(0.0, 0.8, default=0.25, decimals=2, space="buy")

    # =====================================================================

    # per-pair timestamp of the last rebalance check that acted
    _last_rebalance: dict[str, datetime]

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # lazily init per-instance state (avoids class-level mutable default)
        if not hasattr(self, "_last_rebalance"):
            self._last_rebalance = {}

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Causal Bollinger MID band = SMA(bb_period) of typical price (only
        # looks backward → no leakage). Only the mid is used as a timing
        # reference, so the std-dev multiplier is irrelevant (it scales only
        # the unused upper/lower bands) — fixed at 2.0.
        bb = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=self.bb_period.value, stds=2.0
        )
        dataframe["bb_mid"] = bb["mid"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initial basket fill. An unheld coin is at 0% weight — i.e. always
        # below its target by more than the band — so the drift trigger is
        # implicitly satisfied and the BB gate alone decides WHEN to start
        # the position (below mid-BB). Freqtrade opens at most one trade per
        # pair, and max_open_trades (= number of coins, set in config) caps
        # the basket. Actual size is set in custom_stake_amount().
        # LOOKAHEAD: close and bb_mid are same-row, current-candle values.
        if self.use_bb_gate.value:
            eb = self.entry_band
            eb = eb.value if hasattr(eb, "value") else eb
            thresh = dataframe["bb_mid"] * (1.0 - eb)
            dataframe.loc[dataframe["close"] < thresh, "enter_long"] = 1
        else:
            # Raw arm: fill the basket ASAP, no BB timing.
            dataframe["enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No signal-driven exits. Positions only shrink via rebalance trims.
        return dataframe

    # ------------------------------------------------------------------
    # Weight target — variant-specific (constant-mix vs CPPI)
    # ------------------------------------------------------------------
    def get_target_weight(self, pair: str, current_time: datetime, dataframe: DataFrame) -> float:
        """Target weight (fraction of total portfolio value) for `pair`.

        Subclasses implement the portfolio philosophy here.
        """
        raise NotImplementedError

    def _portfolio_value(self) -> float:
        """Current MARK-TO-MARKET portfolio value = free cash + Σ position values.

        NOT ``wallets.get_total_stake_amount()`` — that returns the stake
        *budget* (cost basis + free cash), which is ~constant as prices move.
        Constant-mix / CPPI weights need the LIVE total so a coin's share
        reflects current prices. LOOKAHEAD: all prices are current-candle
        closes (.iloc[-1]) / current_rate — present state only.
        """
        from freqtrade.persistence import Trade

        stake_ccy = self.config["stake_currency"]
        pv = self.wallets.get_free(stake_ccy)  # the undeployed cash bucket
        for ot in Trade.get_trades_proxy(is_open=True):
            dframe, _ = self.dp.get_analyzed_dataframe(ot.pair, self.timeframe)
            if dframe is not None and len(dframe):
                price = dframe["close"].iloc[-1]
            else:
                price = ot.open_rate
            pv += ot.amount * price
        return pv

    def _deployable_value(self, current_time: datetime) -> float:
        """Portfolio value the basket is allowed to manage.

        With the profit-skim overlay OFF this is just the full portfolio value.
        With it ON, a fraction of each new equity high is ratcheted into a
        reserved `_banked` bucket that is never redeployed, so the deployable
        total = portfolio value − banked. The banked bucket only grows (one-way
        skim) — that is the "income taken off the table". LOOKAHEAD: only reads
        the current portfolio value and a monotonic high-water mark.
        """
        pv = self._portfolio_value()
        # profit_skim_enable / skim_fraction may be hyperopt Parameters OR plain
        # values — a subclass can fix them (e.g. CppiSkimBasket) to run skim as
        # a fixed income policy rather than a tuned knob.
        skim = self.profit_skim_enable
        skim = skim.value if hasattr(skim, "value") else skim
        if not skim:
            return pv
        frac = self.skim_fraction
        frac = frac.value if hasattr(frac, "value") else float(frac)

        if not hasattr(self, "_banked"):
            self._banked = 0.0
            self._skim_hwm = pv
            self._skim_last = None
            self._skim_log = current_time
        # Ratchet once per candle: bank a slice of any new equity high.
        if current_time != self._skim_last:
            self._skim_last = current_time
            if pv > self._skim_hwm:
                self._banked += frac * (pv - self._skim_hwm)
                self._skim_hwm = pv
            # Monthly income visibility — cumulative amount taken off the table.
            if (current_time - self._skim_log) >= timedelta(days=30):
                self._skim_log = current_time
                start = float(self.config.get("dry_run_wallet") or 0.0)
                pct = (self._banked / start * 100.0) if start else 0.0
                log.info(
                    "INCOME %s  banked=%.0f (%.1f%% of start)  at-risk=%.0f",
                    current_time.date(), self._banked, pct, pv - self._banked,
                )
        return max(pv - self._banked, 0.0)

    def _n_coins(self) -> int:
        return max(1, len(self.dp.current_whitelist()))

    # ------------------------------------------------------------------
    # Cross-sectional helpers — for weight schemes that rank coins against
    # each other (inverse-vol, momentum, min-variance). Cached per candle so
    # every pair's get_target_weight() in one candle shares one computation.
    # LOOKAHEAD: all reads are .iloc[-1] / trailing windows on the analysed
    # frame, which is truncated to the current candle in backtest.
    # ------------------------------------------------------------------
    def _cross_section(self, column: str, current_time: datetime) -> dict[str, float]:
        """Current (causal) value of `column` for every whitelist coin."""
        cache = getattr(self, "_xs_cache", (None, None))
        if cache[0] == (column, current_time):
            return cache[1]
        out: dict[str, float] = {}
        for pair in self.dp.current_whitelist():
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df) and column in df.columns:
                v = df[column].iloc[-1]
                if v == v:  # not NaN
                    out[pair] = float(v)
        self._xs_cache = ((column, current_time), out)
        return out

    def _return_matrix(self, lookback: int, current_time: datetime):
        """Aligned trailing simple returns for whitelist coins (causal).

        DataFrame: columns = pairs, rows = the last `lookback` returns. Used by
        covariance / portfolio-volatility schemes.
        """
        import pandas as pd

        cache = getattr(self, "_rm_cache", (None, None))
        if cache[0] == (lookback, current_time):
            return cache[1]
        cols = {}
        for pair in self.dp.current_whitelist():
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or len(df) < lookback + 1:
                continue
            cols[pair] = df["close"].pct_change().iloc[-lookback:].reset_index(drop=True)
        rm = pd.DataFrame(cols).dropna(how="any")
        self._rm_cache = ((lookback, current_time), rm)
        return rm

    def _periods_per_year(self) -> float:
        """Candles per year for the strategy timeframe (for annualising vol)."""
        from freqtrade.exchange import timeframe_to_minutes

        return (365.0 * 24 * 60) / timeframe_to_minutes(self.timeframe)

    def _equal_coin_weight(self) -> float:
        """Equal split of the deployable (non-cash) fraction across coins."""
        if self.target_weight_per_coin is not None:
            return float(self.target_weight_per_coin)
        return (1.0 - self.cash_target_weight.value) / self._n_coins()

    # ------------------------------------------------------------------
    # Position sizing — initial entry to target weight of total value
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Size the INITIAL entry to the coin's target weight of CURRENT total
        # portfolio value (deployed + free cash), NOT Freqtrade's default.
        total = self._deployable_value(current_time)
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        target_w = self.get_target_weight(pair, current_time, df)
        stake = target_w * total
        stake = min(stake, max_stake)
        if min_stake:
            # Below exchange minimum notional: don't open a dust position.
            if stake < min_stake:
                return 0.0
        return max(stake, 0.0)

    # ------------------------------------------------------------------
    # Rebalancing — the heart of the basket
    # ------------------------------------------------------------------
    def _due_for_rebalance(self, pair: str, current_time: datetime) -> bool:
        if not hasattr(self, "_last_rebalance"):
            self._last_rebalance = {}
        last = self._last_rebalance.get(pair)
        if last is None:
            return True
        return current_time - last >= timedelta(hours=self.rebalance_interval_hours.value)

    def _bb_gate_allows(
        self, adding: bool, current_rate: float, bb_mid: float, pair: str, drift: float
    ) -> bool:
        """Execution-timing gate. Add only below mid-BB (by entry_band); trim
        only above mid-BB.

        Subclasses may widen this (e.g. constant-mix's runaway override).
        """
        if adding:
            eb = self.entry_band
            eb = eb.value if hasattr(eb, "value") else eb
            return current_rate < bb_mid * (1.0 - eb)
        return current_rate > bb_mid

    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        pair = trade.pair

        # Hard per-position cap: if this coin's live weight exceeds the cap,
        # trim it straight back to the cap NOW — bypassing the cadence and BB
        # gates below. This is a safety bound on single-coin concentration, not
        # a normal rebalance, so it runs before the deliberate-cadence path.
        cap = self.max_position_weight
        cap = cap.value if hasattr(cap, "value") else cap
        if cap and cap > 0:
            total_cap = self._deployable_value(current_time)
            if total_cap > 0:
                cur_val = trade.amount * current_rate
                if cur_val / total_cap > cap:
                    delta = cap * total_cap - cur_val  # negative => trim
                    if not min_stake or abs(delta) >= min_stake:
                        self._last_rebalance[pair] = current_time
                        return delta

        # Cadence gate: CHECK at most once per rebalance_interval_hours per
        # coin — deliberate over responsive. We stamp the check time now,
        # before evaluating, so a look that finds nothing to do (or is blocked
        # by the BB gate) still consumes the interval and we wait to the next
        # one, rather than re-checking every candle.
        if not self._due_for_rebalance(pair, current_time):
            return None
        self._last_rebalance[pair] = current_time

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        # LOOKAHEAD: in backtest this frame is truncated to the current
        # candle, so .iloc[-1] is the latest closed bar — never future.
        bb_mid = df["bb_mid"].iloc[-1]
        if bb_mid != bb_mid:  # NaN guard (startup)
            return None

        total = self._deployable_value(current_time)
        if total <= 0:
            return None

        # Current weight of this coin = position value / total portfolio value.
        current_value = trade.amount * current_rate
        current_weight = current_value / total
        target_weight = self.get_target_weight(pair, current_time, df)

        drift = current_weight - target_weight  # +ve == overweight
        if abs(drift) <= self.rebalance_band.value:
            return None  # inside the band — leave it alone

        desired_value = target_weight * total
        delta_value = desired_value - current_value  # +ve add, -ve trim
        adding = delta_value > 0

        # Execution-timing gate (drift already triggered; BB decides WHEN).
        # Skip entirely for the "raw band-rebalance" comparison arm.
        if self.use_bb_gate.value and not self._bb_gate_allows(
            adding, current_rate, bb_mid, pair, drift
        ):
            return None

        # Respect exchange minimum notional so small trims/adds don't fail.
        if min_stake and abs(delta_value) < min_stake:
            return None

        if adding:
            # Never try to add more cash than is available.
            delta_value = min(delta_value, max_stake)
            if delta_value <= 0:
                return None
        else:
            # Trim: keep at least min_stake in the position so a rebalance
            # trim never fully closes a coin we intend to keep maintaining.
            # (CPPI may legitimately want zero exposure — it overrides this.)
            keep_floor = (min_stake or 0.0)
            max_trim = current_value - keep_floor
            if max_trim <= 0:
                return None
            delta_value = -min(abs(delta_value), max_trim)

        return delta_value
