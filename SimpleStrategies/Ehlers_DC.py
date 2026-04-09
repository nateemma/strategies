from EhlersBase import EhlersBase
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd


class Ehlers_DC(EhlersBase):
    """
    Ehlers Strategy using Donchian Channels for Cycle and Trend signals.
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
    exit_dc_factor = DecimalParameter(0.5, 1.0, default=0.85, decimals=2, space="sell")

    def add_strategy_indicators(self, dataframe):
        if "dc_upper" not in dataframe.columns:
            dataframe["dc_upper"] = (
                dataframe["high"].rolling(int(self.entry_period.value)).max()
            )
            dataframe["dc_lower"] = (
                dataframe["low"].rolling(int(self.entry_period.value)).min()
            )
            dataframe["dc_mid"] = (dataframe["dc_upper"] + dataframe["dc_lower"]) / 2
        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        # Cycle Entry: Mean Reversion. Buy when price crosses above Lower Channel.
        entry_trigger = np.where(
            qtpylib.crossed_above(dataframe["close"], dataframe["dc_lower"]),
            1,
            0,
        )
        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):
        # Trend Entry: Breakout. Buy when price crosses above Upper Channel.
        entry_trigger = np.where(
            (dataframe["dsp"] > 0.0)
            & qtpylib.crossed_above(dataframe["close"], dataframe["dc_upper"]),
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):
        # Cycle Exit: Mean Reversion. Sell when price crosses above Upper Channel.
        exit_trigger = np.where(
            qtpylib.crossed_above(dataframe["close"], dataframe["dc_upper"]),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        # Trend Exit: Trailing stop. Sell when price crosses below Mid + factor * (Upper - Mid).
        factor = self.exit_dc_factor.value
        upper_half_gap = dataframe["dc_upper"] - dataframe["dc_mid"]
        exit_threshold = dataframe["dc_mid"] + (factor * upper_half_gap)

        exit_trigger = np.where(
            qtpylib.crossed_below(dataframe["close"], exit_threshold),
            1,
            0,
        )
        return exit_trigger
