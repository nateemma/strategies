from EhlersBase import EhlersBase
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np


class Ehlers_KC(EhlersBase):
    """
    Ehlers Strategy using Keltner Channels for Cycle and Trend signals.
    """

    # Buy hyperspace params:
    buy_params = {
        **EhlersBase.buy_params,
        "entry_period": 20,
        "entry_atr_period": 10,
        "entry_atr_mult": 2.0,
        "entry_wobv_pct": 0.0,
    }

    # Sell hyperspace params:
    sell_params = {
        **EhlersBase.sell_params,
        "exit_guard_metric": 0.1,
    }

    # Strategy parameters
    entry_period = IntParameter(10, 100, default=20, space="buy")
    entry_atr_period = IntParameter(10, 30, default=10, space="buy")
    entry_atr_mult = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")

    exit_kc_factor = DecimalParameter(0.5, 1.0, default=0.85, decimals=2, space="sell")

    def add_strategy_indicators(self, dataframe):
        if "kc_upper" not in dataframe.columns:
            kc_mid = ta.EMA(dataframe, timeperiod=int(self.entry_period.value))
            atr = ta.ATR(dataframe, timeperiod=int(self.entry_atr_period.value))
            mult = self.entry_atr_mult.value

            dataframe["kc_upper"] = kc_mid + (mult * atr)
            dataframe["kc_lower"] = kc_mid - (mult * atr)
            dataframe["kc_mid"] = kc_mid
        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        # Cycle Entry: Mean Reversion. Buy when price crosses above Lower Channel.

        entry_trigger = np.where(
            qtpylib.crossed_above(dataframe["close"], dataframe["kc_lower"]),
            1,
            0,
        )
        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):
        # Trend Entry: Breakout. Buy when price crosses above Upper Channel.
        entry_trigger = np.where(
            (dataframe["dsp"] > 0.0)
            & qtpylib.crossed_above(dataframe["close"], dataframe["kc_upper"]),
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):
        # Cycle Exit: Mean Reversion. Sell when price crosses above Upper Channel.
        exit_trigger = np.where(
            qtpylib.crossed_above(dataframe["close"], dataframe["kc_upper"]),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        # Trend Exit: Trailing stop. Sell when price crosses below Mid + factor * (Upper - Mid).
        factor = self.exit_kc_factor.value
        upper_half_gap = dataframe["kc_upper"] - dataframe["kc_mid"]
        exit_threshold = dataframe["kc_mid"] + (factor * upper_half_gap)

        exit_trigger = np.where(
            qtpylib.crossed_below(dataframe["close"], exit_threshold),
            1,
            0,
        )
        return exit_trigger
