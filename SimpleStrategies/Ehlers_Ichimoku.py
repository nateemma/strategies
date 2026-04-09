from EhlersBase import EhlersBase
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from finta import TA as fta


class Ehlers_Ichimoku(EhlersBase):
    """
    Ehlers Strategy using Ichimoku logic for Cycle and Trend signals.
    """

    # Buy hyperspace params:
    buy_params = {
        **EhlersBase.buy_params,
        "entry_atr_pct": 0.003,
        "entry_chikou_period": 10,
        "entry_guard_metric": -0.4,
        "entry_kijun_period": 49,
        "entry_senkou_period": 116,
        "entry_tenkan_period": 16,
    }

    # Sell hyperspace params:
    sell_params = {
        **EhlersBase.sell_params,
        "exit_guard_metric": 0.3,
        "exit_ichimoku_factor": 0.54,
        "cexit_enable_profit_checks": True,  # value loaded from strategy
        "cexit_max_days": 21,  # value loaded from strategy
        "cexit_take_profit": 0.008,  # value loaded from strategy
    }

    # Strategy parameters
    entry_tenkan_period = IntParameter(5, 30, default=9, space="buy")
    entry_kijun_period = IntParameter(10, 60, default=26, space="buy")
    entry_senkou_period = IntParameter(20, 120, default=52, space="buy")
    entry_chikou_period = IntParameter(10, 60, default=26, space="buy")

    # Exit parameter (trailing stop factor for trend)
    exit_ichimoku_factor = DecimalParameter(
        0.5, 1.0, default=0.95, decimals=2, space="sell"
    )

    def add_strategy_indicators(self, dataframe):
        if "senkou_span_a" not in dataframe.columns:
            # We use finta for Ichimoku as it handles defaults well, but we need to ensure correct mapping
            # Finta's ICHIMOKU returns TENKAN, KIJUN, SENKOU, CHIKOU, SENKOU_SPAN_A
            # Note: Finta implementation details vary, let's stick to manual or what we know works in Ichimoku.py which used fta.

            ichimoku = fta.ICHIMOKU(
                dataframe,
                tenkan_period=int(self.entry_tenkan_period.value),
                kijun_period=int(self.entry_kijun_period.value),
                senkou_period=int(self.entry_senkou_period.value),
                chikou_period=int(self.entry_chikou_period.value),
            )

            dataframe["senkou_span_a"] = ichimoku["senkou_span_a"]
            dataframe["senkou_span_b"] = ichimoku[
                "SENKOU"
            ]  # Finta typically calls Span B just 'SENKOU'
            dataframe["tenkan_sen"] = ichimoku["TENKAN"]
            dataframe["kijun_sen"] = ichimoku["KIJUN"]
            dataframe["chikou_span"] = ichimoku[
                "CHIKOU"
            ]  # This is usually just Shifted Close.

            # Helper for Cloud Top/Bottom
            dataframe["cloud_top"] = dataframe[["senkou_span_a", "senkou_span_b"]].max(
                axis=1
            )
            dataframe["cloud_bottom"] = dataframe[
                ["senkou_span_a", "senkou_span_b"]
            ].min(axis=1)

            # Chikou check (current close vs close N periods ago)
            # standard Chikou is plotted N periods back.
            # A bullish signal is when current Close > Close[26 periods ago]
            displacement = int(self.entry_chikou_period.value)
            dataframe["chikou_bullish"] = dataframe["close"] > dataframe["close"].shift(
                displacement
            )

        return dataframe

    def get_cycle_entry_trigger(self, dataframe):
        # Cycle Entry: Mean Reversion.
        # Buy when price crosses above Kijun-sen (Base Line) while in a cycle.
        # This is essentially returning to equilibrium.

        if "kijun_sen" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        entry_trigger = np.where(
            (qtpylib.crossed_above(dataframe["close"], dataframe["kijun_sen"]))
            & (dataframe["chikou_bullish"]),  # Confirmation
            1,
            0,
        )
        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):
        # Trend Entry: Breakout.
        # Buy when price closes above the entire Cloud (Kumo Breakout).

        if "cloud_top" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        entry_trigger = np.where(
            (dataframe["close"] > dataframe["cloud_top"])
            & (dataframe["chikou_bullish"]),  # Confirmation
            1,
            0,
        )
        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):
        # Cycle Exit:
        # Sell when price crosses above Cloud? Or just standard indicator exit?
        # Let's use Tenkan-Sen crossover (faster line) as a target exit in cycle.
        # Or even simpler: Close > Cloud Top (target reached)

        if "cloud_top" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        exit_trigger = np.where(
            (dataframe["close"] > dataframe["cloud_top"]),
            1,
            0,
        )
        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        # Trend Exit: Trailing stop below Kijun-Sen or Cloud.
        # Let's use Kijun-Sen (Base Line) as a trailing stop, modified by a factor?
        # Standard Ichimoku trend exit is breaking below Kijun.

        if "kijun_sen" not in dataframe.columns:
            dataframe = self.add_strategy_indicators(dataframe)

        exit_trigger = np.where(
            qtpylib.crossed_below(dataframe["close"], dataframe["kijun_sen"]),
            1,
            0,
        )
        return exit_trigger
