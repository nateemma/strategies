"""OversoldReversion — absolute-oversold mean reversion on liquid alts (1h).

The deliberate counterpart to Basket/MomentumRegimeBasket15m: that book buys what has
gone UP and holds the winners; this one buys deep dislocations and holds a fixed window.
They are run as INDEPENDENT strategies, not blended.

*** SIGNAL (validated, see Reversion/README.md) ***
Enter when RSI(14) crosses BELOW 30 *and* price is more than 30% below its 50-day SMA.
The dislocation term does the work: bucketing by distance-below-SMA50 gives a median
fwd-48h of +136bp in the deepest quintile vs +46/+24/+21/0 for the rest, while RSI depth
barely separates (+44 vs +27). Deep-oversold-and-far-from-trend, not merely oversold.

The -20% threshold used in an earlier draft LOSES MONEY net of costs (-5.9% CAGR): per
signal medians are not portfolio P&L, because you take EVERY signal and pay costs on all
of them. -30% is the viable line.

*** HOLD ***
The signal DECAYS SLOWLY -- mean fwd return rises 4h +64bp -> 48h +105 -> 72h +126 -> 96h
+109. That slow decay is the whole reason this is tradeable where the earlier
CROSS-SECTIONAL reversion study was not: low turnover tolerates costs. A fixed 72h hold
matches the peak. (studies/study1_xsec was rejected -- ZEC was ~92% of its edge and
liquid-only lost -300% at 10bp. THIS signal is liquidity-ROBUST: liquid-only is as good
or better than the full universe, +138.7 vs +125.9bp at 72h.)

*** SIZING -- MANY SMALL BETS ***
MAX_POSITIONS=12, unlike the momentum book's TOP_N=3. This is a real lever, monotonic in
the sweep: Sharpe 0.41/0.46/0.54/0.66 for 3/5/8/12, vol 36%->18%, maxDD -57%->-29%.

*** VECTORISED EXPECTATION (net 40bp, ex-ONE, 14 pairs) ***
  pooled 2021-2026  +11.0% CAGR, 18% vol, Sharpe 0.66, maxDD -29%, 539 trades
  P1 +19.7% (Sh 0.72) | P2 +2.3% (Sh 1.07) | P3 +13.6% (Sh 1.30)
A real backtest should come in BELOW that -- the sim has no slippage and assumes fills at
the modelled prices.

*** NO BTC REGIME GATE -- deliberate. *** btc_corr found a BTC-uptrend entry gate
over-filters mean reversion by ~90% because it fights buy-low. The momentum book's
regime gate is load-bearing; here it would be actively wrong.

Liquidity handling is ported from MomentumRegimeBasket15m, where uncapped exits proved to
carry 75% of reported profit. Entries AND exits are capped to a share of a candle's quote
volume. Duplicated rather than shared because that class's helpers are entangled with its
cross-sectional membership logic; extracting a mixin is future work.

Config: config/config_reversion_1h.json (max_open_trades == MAX_POSITIONS, stake
"unlimited", liquid-only whitelist -- do NOT reuse the momentum whitelist).
"""
from __future__ import annotations

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class OversoldReversion(IStrategy):
    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 1250        # 50d SMA on 1h = 1200 bars, + RSI warmup
    stoploss = -0.99                   # exit is time-based, not stop-based
    minimal_roi = {"0": 100}           # ROI off
    trailing_stop = False
    use_exit_signal = True             # REQUIRED: freqtrade gates custom_exit on this
                                       # (interface.py: `if self.use_exit_signal:` wraps the
                                       # custom_exit call). Setting it False silently disables
                                       # the time-based exit -- trades then ran 184 days.
    use_custom_stoploss = False
    position_adjustment_enable = True  # liquidity-capped accumulation and unwind

    RSI_PERIOD = 14
    RSI_THRESHOLD = 30.0
    SMA_HOURS = 24 * 50                # 50-day SMA on hourly bars
    DISLOCATION = -0.30                # price must be >30% below that SMA
    HOLD_HOURS = 72
    MAX_POSITIONS = 12                 # == config max_open_trades

    # liquidity discipline (same as the momentum book)
    MIN_QUOTE_VOLUME = 1000
    QUOTE_VOLUME_HEADROOM_MULT = 10.0
    FILL_VOLUME_LAG = 0                # iloc[-1] is the last COMPLETED candle in backtest
    EXIT_LIQUIDITY_CAP = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.RSI_PERIOD)
        sma = dataframe["close"].rolling(self.SMA_HOURS).mean()
        dataframe["dislocation"] = dataframe["close"] / sma - 1
        dataframe["rsi_cross_down"] = (
            (dataframe["rsi"] < self.RSI_THRESHOLD)
            & (dataframe["rsi"].shift(1) >= self.RSI_THRESHOLD)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["rsi_cross_down"]
            & (dataframe["dislocation"] < self.DISLOCATION)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe          # time-based exit lives in custom_exit

    def custom_exit(self, pair: str, trade: Trade, current_time, current_rate,
                    current_profit: float, **kwargs):
        held_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if held_h >= self.HOLD_HOURS:
            return "hold_expired"
        return None

    # ---- liquidity helpers (ported from MomentumRegimeBasket15m) ----
    def _quote_volume(self, df) -> float:
        i = -1 - self.FILL_VOLUME_LAG
        if df is None or len(df) < abs(i):
            return 0.0
        bar = df.iloc[i].squeeze()
        return float(bar["volume"]) * float(bar["close"])

    def _portfolio_value(self) -> float:
        pv = self.wallets.get_free(self.config["stake_currency"])
        for ot in Trade.get_trades_proxy(is_open=True):
            df, _ = self.dp.get_analyzed_dataframe(ot.pair, self.timeframe)
            price = df["close"].iloc[-1] if df is not None and len(df) else ot.open_rate
            pv += ot.amount * price
        return pv

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return proposed_stake
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        fillable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
        target = self._portfolio_value() / self.MAX_POSITIONS
        stake = min(proposed_stake, target, fillable, max_stake)
        if min_stake and stake < min_stake:
            return 0.0
        return max(stake, 0.0)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return True
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        return self._quote_volume(df) >= self.MIN_QUOTE_VOLUME

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **kwargs):
        if not self.EXIT_LIQUIDITY_CAP or self.dp.runmode.value in ("plot", "other"):
            return True
        if exit_reason in ("force_exit", "stop_loss", "liquidation"):
            return True
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        sellable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
        return (amount * rate) <= sellable

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return None
        df, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if df is None or len(df) == 0:
            return None
        current_value = trade.amount * current_rate
        held_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        fillable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT

        if held_h >= self.HOLD_HOURS:
            # unwinding: release only what this candle can absorb; when the remainder
            # fits, confirm_trade_exit lets custom_exit finish it
            if not self.EXIT_LIQUIDITY_CAP or fillable <= 0 or fillable >= current_value:
                return None
            reduce = -min(fillable, current_value)
            if min_stake and abs(reduce) < min_stake:
                return None
            return reduce

        target = self._portfolio_value() / self.MAX_POSITIONS
        if current_value >= target * 0.98:
            return None
        add = min(target - current_value, fillable, max_stake)
        if add <= 0 or (min_stake and add < min_stake):
            return None
        return add
