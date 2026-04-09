from freqtrade.strategy import IStrategy
import pandas as pd
import pandas_ta as pta
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_open,
)

# set paths so that we can find imports in parallel directories
import os
import sys
from pathlib import Path
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

from SimpleStrategy import SimpleStrategy

from finta import TA as fta

'''
Ichimoku indicators and cloud
'''
class Ichimoku(SimpleStrategy):

    plot_config = {
        'main_plot': {
            'close': {'color': 'lightsteelblue'},
            'senkou_span_a': {'color': 'lightseagreen'},
            'senkou_span_b': {'color': 'lightsalmon'},
            'chikou_span': {'color': 'lightgrey'},
        },
        'subplots': {
            "Diff": {
                'tenkan_sen': {'color': 'lightseagreen'},
                'kijun_sen': {'color': 'lightsalmon'},
            },
        }
    }

    enable_guards = True # set to True for testing, False for debug

    # Buy hyperspace params:
    buy_params = {
        **SimpleStrategy.buy_params,
        "entry_guard_metric": -0.3,
        "entry_senkou_period": 52,
        "entry_tenkan_period": 12,
        "entry_chikou_period": 26,  # value loaded from strategy
        "entry_kijun_period": 26,  # value loaded from strategy
        "entry_momentum_filter": "roc",
        "entry_tenkan_kijun_filter": "above",
        "entry_price_action_filter": "higher_low",
        "entry_use_cloud_thickness": True,
        "entry_cloud_thickness_max": 0.011,
        "entry_use_htf_filter": True,
        "entry_htf_timeframe": "1h",
    }

    # Sell hyperspace params:
    sell_params = {
        **SimpleStrategy.sell_params,
        "exit_guard_metric": 0.0,
    }

    strategy_type = SimpleStrategy.StrategyType.TREND

    # Strategy parameters

    opt_base_params = True
    opt_filter_params = False

    # entry_tenkan_period = IntParameter(
    #     9, 14, default=9, space="buy", load=True, optimize=True
    # )
    # entry_kijun_period = IntParameter(
    #     26, 40, default=26, space="buy", load=True, optimize=False
    # )
    # entry_senkou_period = IntParameter(
    #     50, 70, default=52, space="buy", load=True, optimize=True
    # )
    # entry_chikou_period = IntParameter(
    #     24, 26, default=26, space="buy", load=True, optimize=False
    # )

    entry_tenkan_period = IntParameter(
        9, 14, default=9, space="buy", load=True, optimize=opt_base_params
    )
    entry_kijun_period = IntParameter(
        26, 40, default=26, space="buy", load=True, optimize=False
    )
    entry_senkou_period = IntParameter(
        50, 70, default=52, space="buy", load=True, optimize=opt_base_params
    )
    entry_chikou_period = IntParameter(
        24, 26, default=26, space="buy", load=True, optimize=False
    )

    entry_momentum_filter = CategoricalParameter(
        ["lagging", "roc", "none"],
        default="roc",
        space="buy",
        load=True,
        optimize=opt_filter_params,
    )
    entry_tenkan_kijun_filter = CategoricalParameter(
        ["above", "cross"],
        default="above",
        space="buy",
        load=True,
        optimize=opt_filter_params,
    )
    entry_price_action_filter = CategoricalParameter(
        ["none", "close_above_prev_high", "higher_low"],
        default="higher_low",
        space="buy",
        load=True,
        optimize=opt_filter_params,
    )
    entry_use_cloud_thickness = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=opt_filter_params
    )
    entry_cloud_thickness_max = DecimalParameter(
        0.005,
        0.05,
        default=0.011,
        decimals=3,
        space="buy",
        load=True,
        optimize=opt_filter_params,
    )
    entry_use_htf_filter = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=opt_filter_params
    )
    entry_htf_timeframe = CategoricalParameter(
        ["1h", "4h"], default="1h", space="buy", load=True, optimize=opt_filter_params
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)

        if self.entry_use_htf_filter.value and self.dp:
            htf_tf = self.entry_htf_timeframe.value
            informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=htf_tf).copy()
            if "date" not in informative.columns:
                informative["date"] = informative.index
            informative["ema200"] = ta.EMA(informative, timeperiod=200)
            informative = informative[["date", "close", "ema200"]]
            dataframe = merge_informative_pair(
                dataframe, informative, self.timeframe, htf_tf, ffill=True
            )
            dataframe["htf_close"] = dataframe[f"close_{htf_tf}"]
            dataframe["htf_ema200"] = dataframe[f"ema200_{htf_tf}"]

        # Recalculate signals after adding HTF columns (if any)
        if self.dp.runmode.value not in ("hyperopt"):
            dataframe["entry_signals"] = self.get_entry_signals(dataframe)
            dataframe["exit_signals"] = self.get_exit_signals(dataframe)

        return dataframe

    def get_entry_signals(self, dataframe):

        ichimoku = fta.ICHIMOKU(dataframe,
                                tenkan_period=self.entry_tenkan_period.value,
                                kijun_period=self.entry_kijun_period.value,
                                senkou_period=self.entry_senkou_period.value,
                                chikou_period=self.entry_chikou_period.value
                                )

        dataframe['senkou_span_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_span_b'] = ichimoku['SENKOU']
        dataframe['tenkan_sen'] = ichimoku['TENKAN']
        dataframe['kijun_sen'] = ichimoku['KIJUN']

        momentum_filter = self.entry_momentum_filter.value
        if momentum_filter == "roc":
            momentum_ok = dataframe["roc"] > 0.0
        elif momentum_filter == "lagging":
            momentum_ok = dataframe["close"] > dataframe["close"].shift(
                int(self.entry_chikou_period.value)
            )
        else:
            momentum_ok = True

        tenkan_kijun_filter = self.entry_tenkan_kijun_filter.value
        if tenkan_kijun_filter == "cross":
            tk_ok = qtpylib.crossed_above(
                dataframe["tenkan_sen"], dataframe["kijun_sen"]
            )
        else:
            tk_ok = dataframe["tenkan_sen"] > dataframe["kijun_sen"]

        price_action_filter = self.entry_price_action_filter.value
        if price_action_filter == "close_above_prev_high":
            price_action_ok = dataframe["close"] > dataframe["high"].shift(1)
        elif price_action_filter == "higher_low":
            price_action_ok = dataframe["low"] > dataframe["low"].shift(1)
        else:
            price_action_ok = True

        if self.entry_use_cloud_thickness.value:
            cloud_thickness = (dataframe["senkou_span_a"] - dataframe["senkou_span_b"]).abs() / dataframe["close"]
            cloud_ok = cloud_thickness < self.entry_cloud_thickness_max.value
        else:
            cloud_ok = True

        if self.entry_use_htf_filter.value and "htf_ema200" in dataframe:
            htf_ok = dataframe["htf_close"] > dataframe["htf_ema200"]
        else:
            htf_ok = True

        series = np.where(
            (
                (tk_ok)
                &
                (dataframe['senkou_span_a'] > dataframe['senkou_span_b'])
                &
                (dataframe['close'] > dataframe['senkou_span_b'])
                &
                (momentum_ok)
                &
                (price_action_ok)
                &
                (cloud_ok)
                &
                (htf_ok)
            ), 1, 0)

        return series

    def get_exit_signals(self, dataframe):

        momentum_filter = self.entry_momentum_filter.value
        if momentum_filter == "roc":
            momentum_exit = dataframe["roc"] < 0.0
        elif momentum_filter == "lagging":
            momentum_exit = dataframe["close"] < dataframe["close"].shift(
                int(self.entry_chikou_period.value)
            )
        else:
            momentum_exit = True

        tenkan_kijun_filter = self.entry_tenkan_kijun_filter.value
        if tenkan_kijun_filter == "cross":
            tk_exit = qtpylib.crossed_below(
                dataframe["tenkan_sen"], dataframe["kijun_sen"]
            )
        else:
            tk_exit = dataframe["tenkan_sen"] < dataframe["kijun_sen"]

        series = np.where(
            (
                (tk_exit)
                &
                (dataframe['senkou_span_a'] < dataframe['senkou_span_b'])
                &
                (dataframe['close'] < dataframe['senkou_span_b'])
                &
                (momentum_exit)
            ), 1, 0)
        return series
