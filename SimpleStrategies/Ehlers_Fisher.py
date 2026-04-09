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

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from EhlersBase import EhlersBase

"""
Fisher Breakout/Reversion
"""


class Ehlers_Fisher(EhlersBase):

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
            # "ss": {"color": "orange"},
        },
        "subplots": {
            "Diff": {
                "trendmode": {"color": "lightskyblue"},
                # "ema_slope": {"color": "orange"},
                "ebsw_slope": {"color": "lightcoral"},
                "cg_norm": {"color": "gray"},
                "cg_slope": {"color": "silver"},
                "dsp": {"color": "lightsalmon"},
            },
            "Triggers": {
                "cycle_buy": {"color": "lightseagreen"},
                "trend_buy": {"color": "lightskyblue"},
                "cycle_sell": {"color": "orange"},
                "trend_sell": {"color": "lightsalmon"},
            },
            "Indicators": {
                "fisher_dsp": {"color": "green"},
                "fisher_roc": {"color": "red"},
                "fisher_rsi": {"color": "blue"},
                "fisher_stoch": {"color": "orange"},
                "fisher_mfi": {"color": "purple"},
                "fisher_cci": {"color": "yellow"},
                "fisher_macd": {"color": "cyan"},
                "fisher_composite": {"color": "black"},
            },
            "Filters": {
                "atr_ok": {"color": "gray"},
            },
        },
    }

    # Buy hyperspace params:

    # Strategy parameters

    # Cycle regime: oversold/overbought extremes
    cycle_entry_fisher = DecimalParameter(
        -2.0, -0.5, default=-1.5, decimals=2, space="buy", load=True, optimize=True
    )
    cycle_exit_fisher = DecimalParameter(
        0.5, 2.0, default=1.5, decimals=2, space="sell", load=True, optimize=True
    )

    # Trend regime: momentum crossover levels (centered around 0)
    trend_entry_fisher = DecimalParameter(
        -1.0, 1.0, default=0.0, decimals=2, space="buy", load=True, optimize=True
    )
    trend_exit_fisher = DecimalParameter(
        -1.0, 1.0, default=0.0, decimals=2, space="sell", load=True, optimize=True
    )

    # Choose which fisher transform to use as the primary signal
    fisher_indicator = CategoricalParameter(
        ["dsp", "roc", "composite", "stoch", "mfi", "cci", "macd"],
        default="composite",
        space="buy",
        load=True,
        optimize=True,
    )

    # Trigger mode
    trigger_type = CategoricalParameter(
        ["level", "diff", "dual"],
        default="diff",
        space="buy",
        load=True,
        optimize=True,
    )

    def add_strategy_indicators(self, dataframe):

        if "fisher_dsp" not in dataframe.columns:
            # Fisher Transform of DSP (Detrended Price)
            # We apply the same 20-period detrending and 5-period smoothing approach as CNorm

            # Map of internal names to their raw inputs
            inputs = {}

            # 1. Prepare raw inputs
            inputs["dsp"] = dataframe["dsp"].ffill().fillna(0.0)

            rsi_clean = dataframe["rsi"].ffill().fillna(50.0)
            # RSI needs to be centered/normalized before Fisher if we don't detrend,
            # but we will detrend everything for consistency.
            inputs["rsi"] = rsi_clean

            inputs["stoch"] = ta.STOCH(dataframe)["slowk"].ffill().fillna(50.0)
            inputs["mfi"] = ta.MFI(dataframe).ffill().fillna(50.0)
            inputs["cci"] = ta.CCI(dataframe).ffill().fillna(0.0)

            macd = ta.MACD(dataframe)
            inputs["macd"] = macd["macdhist"].ffill().fillna(0.0)

            # 10-period ROC
            inputs["roc"] = (
                (dataframe["close"] - dataframe["close"].shift(10))
                / dataframe["close"].shift(10).replace(0, np.nan).fillna(1e-12)
            ).fillna(0.0)

            # 2. Process each indicator with the smoothing approach
            # (Detrend 20 -> Fisher 9 -> Smooth 5 -> Slope Rolling 3)
            for name, raw_data in inputs.items():
                # Detrend
                smooth_baseline = self.super_smoother(raw_data, period=20)
                detrended = raw_data - smooth_baseline

                # Fisher + Smoothing
                raw_fisher = self.safe_fisher(detrended, length=9).fillna(0.0)
                fisher_final = self.super_smoother(raw_fisher, period=5)

                dataframe[f"fisher_{name}"] = fisher_final

                # Smoothed Slope (Diff)
                dataframe[f"fisher_{name}_diff"] = (
                    fisher_final.diff().rolling(window=3).mean().fillna(0.0)
                )

            # 3. Special Case: Composite
            # Average the normalized versions of components before Fisher-ing
            stoch_norm = ((inputs["stoch"] / 50.0) - 1.0).clip(-0.99, 0.99)
            mfi_norm = ((inputs["mfi"] / 50.0) - 1.0).clip(-0.99, 0.99)
            cci_norm = self.rolling_normalize(inputs["cci"], length=20)
            macd_norm = self.rolling_normalize(inputs["macd"], length=20)
            roc_norm = self.rolling_normalize(inputs["roc"], length=20)

            composite_raw = (
                stoch_norm + mfi_norm + cci_norm + macd_norm + roc_norm
            ) / 5.0

            # Apply the pattern to composite
            comp_smooth_baseline = self.super_smoother(composite_raw, period=20)
            comp_detrended = composite_raw - comp_smooth_baseline
            comp_raw_fisher = self.safe_fisher(comp_detrended, length=9).fillna(0.0)
            comp_fisher_final = self.super_smoother(comp_raw_fisher, period=5)

            dataframe["fisher_composite"] = comp_fisher_final
            dataframe["fisher_composite_diff"] = (
                comp_fisher_final.diff().rolling(window=3).mean().fillna(0.0)
            )

        return dataframe

    # cycle = mean reversion, trend = breakout

    def get_cycle_entry_trigger(self, dataframe):
        entry_thresh = self.cycle_entry_fisher.value
        trigger = self.trigger_type.value
        ind = self.fisher_indicator.value
        fisher_col = f"fisher_{ind}"
        fisher_diff_col = f"fisher_{ind}_diff"
        # Only buy a bounce if the short-term trend is not already heading down
        trend_up = dataframe["dsp"] > 0.0
        fisher = dataframe[fisher_col]
        fisher_diff = dataframe[fisher_diff_col]

        if trigger == "level":
            # 2-candle confirmation: fisher must be above threshold for 2 consecutive candles
            # after being below it (one-shot, not perpetual)
            level_confirmed = (
                (fisher > entry_thresh)
                & (fisher.shift(1) > entry_thresh)
                & (fisher.shift(2) <= entry_thresh)
            )
            entry_trigger = trend_up & level_confirmed

        elif trigger == "dual":
            second_ind = "fisher_composite" if ind != "composite" else "fisher_dsp"
            second = dataframe[second_ind]
            both_oversold = (fisher < 0.0) & (second < 0.0)
            # 2-candle confirmation on the trigger indicator
            primary_confirmed = (
                (fisher > entry_thresh)
                & (fisher.shift(1) > entry_thresh)
                & (fisher.shift(2) <= entry_thresh)
            )
            secondary_confirmed = (
                (second > entry_thresh)
                & (second.shift(1) > entry_thresh)
                & (second.shift(2) <= entry_thresh)
            )
            entry_trigger = (
                trend_up & both_oversold & (primary_confirmed | secondary_confirmed)
            )

        elif trigger == "diff":
            # 2-candle confirmation: diff/slope must be positive for 2 consecutive candles
            diff_confirmed = (
                (fisher_diff > 0.0)
                & (fisher_diff.shift(1) > 0.0)
                & (fisher_diff.shift(2) <= 0.0)
            )
            entry_trigger = trend_up & diff_confirmed & (fisher < entry_thresh)

        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):

        trigger = self.trigger_type.value
        ind = self.fisher_indicator.value
        fisher_col = f"fisher_{ind}"
        fisher_diff_col = f"fisher_{ind}_diff"
        trend_thresh = self.trend_entry_fisher.value
        # Require dsp > 0 to confirm the trend is actually upward
        trend_up = dataframe["dsp"] > 0.0

        if trigger == "level":
            entry_trigger = trend_up & qtpylib.crossed_above(
                dataframe[fisher_col], trend_thresh
            )

        elif trigger == "dual":
            second_ind = "fisher_composite" if ind != "composite" else "fisher_dsp"
            both_bullish = (dataframe[fisher_col] > trend_thresh) & (
                dataframe[second_ind] > trend_thresh
            )
            trigger_event = qtpylib.crossed_above(
                dataframe[fisher_col], trend_thresh
            ) | qtpylib.crossed_above(dataframe[second_ind], trend_thresh)
            entry_trigger = trend_up & both_bullish & trigger_event

        elif trigger == "diff":
            # Momentum turns UP while below the trend threshold
            entry_trigger = (
                trend_up
                & qtpylib.crossed_above(dataframe[fisher_diff_col], 0.0)
                & (dataframe[fisher_col] < trend_thresh)
            )

        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):
        exit_thresh = self.cycle_exit_fisher.value
        trigger = self.trigger_type.value
        ind = self.fisher_indicator.value
        fisher_col = f"fisher_{ind}"
        fisher_diff_col = f"fisher_{ind}_diff"

        if trigger == "level":
            # Leaving overbought zone (turning down)
            exit_trigger = qtpylib.crossed_below(dataframe[fisher_col], exit_thresh)

        elif trigger == "dual":
            # Both are broadly overbought, and at least one is now crossing below the threshold
            second_ind = "fisher_composite" if ind != "composite" else "fisher_dsp"
            both_overbought = (dataframe[fisher_col] > 0.0) & (
                dataframe[second_ind] > 0.0
            )
            trigger_event = qtpylib.crossed_below(
                dataframe[fisher_col], exit_thresh
            ) | qtpylib.crossed_below(dataframe[second_ind], exit_thresh)
            exit_trigger = both_overbought & trigger_event

        elif trigger == "diff":
            # Momentum turned DOWN (slope < 0) while deep in overbought territory
            exit_trigger = qtpylib.crossed_below(dataframe[fisher_diff_col], 0.0) & (
                dataframe[fisher_col] > exit_thresh
            )

        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):
        trigger = self.trigger_type.value
        ind = self.fisher_indicator.value
        fisher_col = f"fisher_{ind}"
        fisher_diff_col = f"fisher_{ind}_diff"
        trend_thresh = self.trend_exit_fisher.value

        # Trend exit: dsp breaking down OR momentum fading below threshold
        if trigger == "level":
            exit_trigger = (dataframe["dsp"] < 0.0) | qtpylib.crossed_below(
                dataframe[fisher_col], trend_thresh
            )

        elif trigger == "dual":
            second_ind = "fisher_composite" if ind != "composite" else "fisher_dsp"
            both_bearish = (dataframe[fisher_col] < trend_thresh) & (
                dataframe[second_ind] < trend_thresh
            )
            trigger_event = qtpylib.crossed_below(
                dataframe[fisher_col], trend_thresh
            ) | qtpylib.crossed_below(dataframe[second_ind], trend_thresh)
            exit_trigger = (dataframe["dsp"] < 0.0) | (both_bearish & trigger_event)

        elif trigger == "diff":
            # Momentum turns DOWN (early trend exit, profit before the full cross)
            exit_trigger = (dataframe["dsp"] < 0.0) | (
                qtpylib.crossed_below(dataframe[fisher_diff_col], 0.0)
                & (dataframe[fisher_col] > trend_thresh)
            )

        return exit_trigger
