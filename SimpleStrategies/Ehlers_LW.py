from EhlersBase import EhlersBase
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from finta import TA as fta


class Ehlers_LW(EhlersBase):
    """
    Ehlers Strategy using Larry Williams Volatility logic for Cycle and Trend signals.
    """

    # Buy hyperspace params:
    buy_params = {
        **EhlersBase.buy_params,
        "entry_period": 20,
        "entry_wobv_pct": 0.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **EhlersBase.sell_params,
        "exit_guard_metric": 0.1,
    }

    # Strategy parameters
    entry_period = IntParameter(10, 100, default=20, space="buy")

    # Reversion (Cycle) Multipliers
    cycle_entry_mult = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space="buy")
    cycle_exit_mult = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space="sell")

    # Breakout (Trend) Multipliers
    trend_entry_mult = DecimalParameter(0.1, 2.0, default=0.5, decimals=1, space="buy")
    trend_exit_mult = DecimalParameter(0.1, 2.0, default=0.3, decimals=1, space="sell")

    def add_strategy_indicators(self, dataframe):
        if "lw_atr" not in dataframe.columns:
            # We use the same period for ATR for simplicity, or could add separate params
            period = int(self.entry_period.value)
            dataframe["lw_atr"] = fta.ATR(dataframe, period=period)

            # Pre-calculate levels for Cycle (Reversion) - computed on previous close context often,
            # but standard LW is from Open.
            # Reversion: Buy Dip (Open - k*ATR), Sell Rally (Open + k*ATR)
            dataframe["lw_cycle_buy"] = dataframe["open"] - (
                self.cycle_entry_mult.value * dataframe["lw_atr"]
            )
            dataframe["lw_cycle_sell"] = dataframe["open"] + (
                self.cycle_exit_mult.value * dataframe["lw_atr"]
            )

            # Pre-calculate levels for Trend (Breakout)
            # Breakout: Buy Break (Open + k*ATR), Sell (Trailing) Stop (Open - k*ATR) of current bar?
            # Or usually breakouts are from *previous* High/Low, but LW Volatility Breakout often uses Open + Range.
            dataframe["lw_trend_buy"] = dataframe["open"] + (
                self.trend_entry_mult.value * dataframe["lw_atr"]
            )
            dataframe["lw_trend_sell"] = dataframe["open"] - (
                self.trend_exit_mult.value * dataframe["lw_atr"]
            )

        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        # Cycle Entry: Mean Reversion. Buy when price is below Open - k*ATR
        # Note: 'close' here is the current candle's close (or current price in live).
        # We check if we closed below the buy level.

        # Ensure indicators exist (redundancy for safety)
        if "lw_cycle_buy" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        entry_trigger = np.where(
            # (dataframe['close'] < dataframe['lw_cycle_buy']),
            qtpylib.crossed_above(dataframe["close"], dataframe["lw_cycle_buy"]),
            1,
            0,
        )
        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):
        # Trend Entry: Breakout. Buy when price crosses above Open + k*ATR

        if "lw_trend_buy" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        entry_trigger = np.where(
            (dataframe["dsp"] > 0.0)
            & qtpylib.crossed_above(dataframe["close"], dataframe["lw_trend_buy"]),
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):
        # Cycle Exit: Mean Reversion Target. Sell when price is above Open + k*ATR

        if "lw_cycle_sell" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        exit_trigger = np.where(
            (dataframe["close"] > dataframe["lw_cycle_sell"]),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        # Trend Exit: Stop Loss / Trailing. Sell when price crosses below Open - k*ATR

        if "lw_trend_sell" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        exit_trigger = np.where(
            qtpylib.crossed_below(dataframe["close"], dataframe["lw_trend_sell"]),
            1,
            0,
        )
        return exit_trigger
