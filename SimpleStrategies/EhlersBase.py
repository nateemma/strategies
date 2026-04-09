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

from finta import TA as fta

import warnings

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from SimpleStrategy import SimpleStrategy

# from finta import TA as fta

"""
EhlersBase - investigate use of various Ehlers' filters/indicators. Intended as a base class for other strategies.
The intent is to provide a framework for a combination of mean reversion and breakout approaches.
Use mean reversion when in cycle mode, breakout when trending

This version uses hand-coded indicators to enforce no lookahead (unlike the ta-lib and pandas-ta versions)
"""


class EhlersBase(SimpleStrategy):

    strategy_type = SimpleStrategy.StrategyType.OTHER

    enable_guards = True

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
            # "ss": {"color": "orange"},
        },
        "subplots": {
            "Diff": {
                "trendmode": {"color": "lightskyblue"},
                "regime_score": {"color": "white"},
                "ema_slope": {"color": "orange"},
                "ebsw_slope": {"color": "lightcoral"},
                "cg_norm": {"color": "gray"},
                "cg_slope": {"color": "silver"},
                "dpo": {"color": "lightsalmon"},
            },
            "Triggers": {
                # "ht_sine": {"color": "green"},
                # "ht_leadsine": {"color": "blue"},
                "cycle_buy": {"color": "yellow"},
                "trend_buy": {"color": "orange"},
                "cycle_sell": {"color": "lightgreen"},
                "trend_sell": {"color": "lightcoral"},
            },
            "Indicators": {
                "roc": {"color": "green"},
                "rsi": {"color": "red"},
            },
            "Filters": {
                "atr_ok": {"color": "gray"},
            },
        },
    }

    # ------------- HYPEROPTABLE PARAMETERS -------------

    # Buy hyperspace params:

    use_strict_regions = CategoricalParameter(
        [True, False], default=False, space="buy", load=True, optimize=True
    )

    # When True, regime is decided by a 5-indicator vote instead of HT_TRENDMODE alone.
    # The vote threshold controls how many indicators must agree on 'trend' (out of 5).
    # Default False = original HT_TRENDMODE-only behaviour (safe fallback).
    use_composite_regime = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
    )
    regime_vote_threshold = IntParameter(
        2, 6, default=3, space="buy", load=True, optimize=True
    )

    # ------------- REGIME HELPER -------------

    def _get_regime_masks(self, dataframe):
        """
        Returns (cycle_mode, trend_mode) boolean Series.

        Simple mode  (use_composite_regime=False):
            Directly uses HT_TRENDMODE == 0 / 1.

        Composite mode (use_composite_regime=True):
            Builds a 0-5 vote score from independent indicators.
            Each cast a trend-vote (1) when it agrees the market is trending.
            trend_mode  = score >= regime_vote_threshold
            cycle_mode  = score <  regime_vote_threshold

            Votes:
              1. HT_TRENDMODE == 1
              2. ema_slope > 0          (slow EMA still rising)
              3. dsp rolling 5-bar mean > 0  (price sustained above super-smoother)
              4. |ebsw_slope| < 0.05    (EBSW not actively oscillating)
              5. cg_slope > 0           (Center of Gravity still climbing)
              6. dymi_period < 12       (Volatility expansion / Trend speed)
        """
        if self.use_composite_regime.value:
            # Need dymi_period for the 6th vote
            dymi_period = self.safe_dymi_period(dataframe)

            score = (
                pd.Series(dataframe["trendmode"] == 1, dtype=int)
                + pd.Series(dataframe["ema_slope"] > 0, dtype=int)
                + pd.Series(
                    dataframe["dsp"].rolling(5, min_periods=1).mean() > 0, dtype=int
                )
                + pd.Series(dataframe["ebsw_slope"].abs() < 0.05, dtype=int)
                + pd.Series(dataframe["cg_slope"] > 0, dtype=int)
                + pd.Series(dymi_period < 12, dtype=int)
            )
            dataframe["regime_score"] = score
            trend_mode = score >= self.regime_vote_threshold.value
            cycle_mode = ~trend_mode
        else:
            dataframe["regime_score"] = dataframe["trendmode"]  # keep chart consistent
            cycle_mode = dataframe["trendmode"] == 0
            trend_mode = dataframe["trendmode"] == 1

        return cycle_mode, trend_mode

    def safe_dymi_period(self, dataframe: DataFrame) -> Series:
        """
        Calculates the adaptive DYMI period based on volatility.
        Low value = fast/trend, High value = slow/cycle.
        """
        close = dataframe["close"]
        sd = close.rolling(5).std()
        asd = sd.rolling(10).mean()
        v = (sd / asd).replace(0, 1e-12)
        t = (14 / v).fillna(14).clip(5, 30)
        return t

    # ------------- HELPER FUNCTIONS -------------

    def safe_dpo(self, close, length=20):
        """Standard DPO shifted to be causal (no lookahead)."""
        shift = int(length / 2 + 1)
        sma = close.rolling(length).mean()
        return close - sma.shift(shift)

    def safe_cg(self, close, length=10) -> Series:
        """Vectorized Center of Gravity."""
        # Weighted sum: (1*Price[oldest] + 2*Price[next] ... + length*Price[newest])
        num = sum((i + 1) * close.shift(length - 1 - i) for i in range(length))
        den = close.rolling(length).sum()
        return Series(num / den, index=close.index)

    def safe_fisher(self, data: Series, length: int = 9) -> Series:
        """Fisher transform with recursive smoothing to filter noise."""
        low: Series = data.rolling(length).min()
        high: Series = data.rolling(length).max()
        spread: Series = (high - low).replace(0, 1e-12)

        # Normalized value (-1 to 1)
        # Wrap in Series to ensure type inference doesn't incorrectly flag fillna on a primitive
        value = Series(2.0 * ((data - low) / spread - 0.5))
        value = value.fillna(0.0).clip(-0.999, 0.999)

        # Recursive smoothing
        vals = value.values
        smoothed_val = np.zeros_like(vals)
        fisher = np.zeros_like(vals)

        for i in range(1, len(vals)):
            # Ehlers filters use specific smoothing factors (0.33 and 0.5)
            # This recursion is what makes the Fisher transform stable
            smoothed_val[i] = 0.33 * vals[i] + 0.67 * smoothed_val[i - 1]
            # Clip to avoid log(0) or log(inf)
            sv = max(min(smoothed_val[i], 0.999), -0.999)
            fisher_raw = 0.5 * np.log((1 + sv) / (1 - sv))
            fisher[i] = 0.5 * fisher_raw + 0.5 * fisher[i - 1]

        return Series(fisher, index=data.index)

    def rolling_normalize(self, series: Series, length: int = 20) -> Series:
        """Normalizes a series to a strict [-1.0, 1.0] range using a rolling window."""
        roll = series.rolling(length, min_periods=min(5, length))
        s_min = roll.min()
        s_max = roll.max()
        s_range = (s_max - s_min).replace(0, 1e-12)
        # Wrap returned calculation in Series for type safety
        result = Series(2.0 * ((series - s_min) / s_range) - 1.0)
        return result.fillna(0.0).clip(-0.999, 0.999)

    def super_smoother(self, series, period=10):
        """Ehlers Super Smoother filter. Causal: uses only current and past bars (i, i-1, i-2)."""
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3

        n = len(series)
        filt = np.zeros(n)
        s = series.values if hasattr(series, "values") else np.asarray(series)

        for i in range(2, n):
            filt[i] = c1 * (s[i] + s[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

        return pd.Series(filt, index=series.index)

    def adaptive_super_smoother(self, series, periods):
        """Ehlers Super Smoother filter with adaptive period."""
        n = len(series)
        filt = np.zeros(n)
        s = series.values if hasattr(series, "values") else np.asarray(series)
        p = periods.values if hasattr(periods, "values") else np.asarray(periods)

        for i in range(2, n):
            period = max(2.5, p[i])  # enforce min period for stability
            a1 = np.exp(-1.414 * np.pi / period)
            b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            filt[i] = c1 * (s[i] + s[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

        return pd.Series(filt, index=series.index)

    def dsp(self, close, period=10):
        smooth = self.super_smoother(close, period)
        return close - smooth

    # ------------- STRATEGY LOGIC -------------

    # override the following functions to implement specific triggers for cycle/trend regimes

    def add_strategy_indicators(self, dataframe):

        if "ht_sine" not in dataframe.columns:
            sine, leadsine = ta.HT_SINE(dataframe["close"])
            dataframe["ht_sine"] = sine
            dataframe["ht_leadsine"] = leadsine

        return dataframe

    def get_cycle_entry_trigger(self, dataframe):

        entry_trigger = qtpylib.crossed_above(
            dataframe["ht_sine"], dataframe["ht_leadsine"]
        )

        return entry_trigger

    def get_trend_entry_trigger(self, dataframe):

        entry_trigger = qtpylib.crossed_below(
            dataframe["ht_sine"], dataframe["ht_leadsine"]
        )

        return entry_trigger

    def get_cycle_exit_trigger(self, dataframe):

        exit_trigger = qtpylib.crossed_above(
            dataframe["ht_sine"], dataframe["ht_leadsine"]
        )

        return exit_trigger

    def get_trend_exit_trigger(self, dataframe):

        exit_trigger = qtpylib.crossed_below(
            dataframe["ht_sine"], dataframe["ht_leadsine"]
        )

        return exit_trigger

    def get_entry_signals(self, dataframe):

        # --------------------------------------------------
        # 1) EHLERS INDICATORS
        # --------------------------------------------------

        # Even Better Sinewave (cycle phase proxy)
        dataframe["ebsw"] = pta.ebsw(dataframe["close"])

        # Center of Gravity (raw); normalise for display (no bfill: avoid lookahead in warmup)
        cg_raw = self.safe_cg(dataframe["close"], length=10)
        dataframe["cg"] = cg_raw
        cg_roll = cg_raw.rolling(window=min(50, len(cg_raw)), min_periods=10)
        cg_mean = cg_roll.mean()
        cg_std = cg_roll.std().replace(0, np.nan).fillna(1.0)
        dataframe["cg_norm"] = (cg_raw - cg_mean) / cg_std

        # Hilbert Transform (TA-Lib) and EBSW: may use symmetric/FFT-style filters (lookahead).
        # If lookahead analysis flags these, consider causal alternatives or shift by period.
        dataframe["trendmode"] = ta.HT_TRENDMODE(
            dataframe["close"]
        )  # 0 = cycle, 1 = trend
        dataframe["dcperiod"] = ta.HT_DCPERIOD(dataframe["close"])

        # dataframe["dsp"] = self.dsp(dataframe["close"])
        dataframe["dpo"] = self.safe_dpo(dataframe["close"], length=20)

        dataframe["ss"] = self.super_smoother(dataframe["close"], 10)
        dataframe["dsp"] = dataframe["close"] - dataframe["ss"]

        # --------------------------------------------------
        # 2) HELPER FEATURES
        # --------------------------------------------------

        # EBSW slope (turn detection)
        dataframe["ebsw_slope"] = dataframe["ebsw"].diff()

        # CG slope
        dataframe["cg_slope"] = dataframe["cg"].diff()
        dataframe["cg_trend_alive"] = np.where((dataframe["cg_slope"] > 0), 1, 0)

        dataframe["ema_fast"] = ta.EMA(dataframe, length=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, length=50)

        dataframe["cg_turn_up"] = qtpylib.crossed_above(dataframe["cg_slope"], 0)
        dataframe["cg_turn_down"] = qtpylib.crossed_below(dataframe["cg_slope"], 0)

        dataframe["ebsw_turn_up"] = qtpylib.crossed_above(dataframe["ebsw_slope"], 0)
        dataframe["ebsw_turn_down"] = qtpylib.crossed_below(dataframe["ebsw_slope"], 0)

        dataframe["ema_slope"] = dataframe["ema_slow"].diff()

        dataframe["trend_alive"] = dataframe["ema_slope"] > 0

        dataframe["ht_trend_start"] = qtpylib.crossed_above(dataframe["trendmode"], 0)

        # add strategy-specific indicators (override in subclass)
        dataframe = self.add_strategy_indicators(dataframe)

        # --------------------------------------------------
        # 3) SIGNAL LOGIC (cycle vs trend regimes)
        # --------------------------------------------------

        cycle_mode, trend_mode = self._get_regime_masks(dataframe)

        cycle_entry_trigger = self.get_cycle_entry_trigger(dataframe)
        trend_entry_trigger = self.get_trend_entry_trigger(dataframe)

        if self.use_strict_regions.value:
            ehlers_cycle = (
                cycle_mode
                & ((dataframe["cg_slope"] > 0) | (dataframe["ebsw_slope"] > 0))
                & (dataframe["dsp"] < 0.0)  # Buy the dip (below SuperSmoother)
            )
            ehlers_trend = (
                trend_mode & (dataframe["ema_slope"] > 0) & (dataframe["dsp"] > 0.0)
            )
        else:
            ehlers_cycle = cycle_mode
            ehlers_trend = trend_mode

        cycle_entry = ehlers_cycle & cycle_entry_trigger

        trend_entry = ehlers_trend & trend_entry_trigger

        # In trend mode only trend_entry can be True (ehlers_cycle is False).
        # In cycle mode only cycle_entry can be True (ehlers_trend is False).
        entry_mask = (cycle_entry | trend_entry) & dataframe["atr_ok"]

        entry_series = np.where(entry_mask, 1, 0)

        # save signals for displaying/debug
        dataframe["ehler_buy"] = entry_series
        dataframe["cycle_buy"] = np.where(cycle_entry, 1, 0)
        dataframe["trend_buy"] = np.where(trend_entry, 1, 0)
        dataframe["ehler_cycle"] = np.where(ehlers_cycle, 1, 0)
        dataframe["ehler_trend"] = np.where(ehlers_trend, 1, 0)
        # Raw triggers (for verification: when ehler_trend=1, entries come from trend_entry_trigger only)
        dataframe["cycle_entry_trigger"] = np.where(cycle_entry_trigger, 1, 0)
        dataframe["trend_entry_trigger"] = np.where(trend_entry_trigger, 1, 0)

        return pd.Series(entry_series, index=dataframe.index)

    def get_exit_signals(self, dataframe):

        cycle_mode, trend_mode = self._get_regime_masks(dataframe)

        cycle_exit_trigger = self.get_cycle_exit_trigger(dataframe)
        trend_exit_trigger = self.get_trend_exit_trigger(dataframe)

        """
        # Cycle exit (sell near cycle highs): CG / EBSW turning down, trigger fires, DPO < 0
        cycle_exit = (
            cycle_mode
            # & (dataframe["cg_turn_down"] | dataframe["ebsw_turn_down"])
            & ((dataframe["cg_slope"] < 0) | (dataframe["ebsw_slope"] < 0))
            # & (dataframe["dsp"] < 0.0)
            & cycle_exit_trigger
        )

        trend_exit = (
            trend_mode 
            & trend_exit_trigger 
            & (dataframe["dsp"] < 0.0)
        )
        """

        if self.use_strict_regions.value:
            ehlers_cycle = (
                cycle_mode
                & ((dataframe["cg_slope"] < 0) | (dataframe["ebsw_slope"] < 0))
                & (dataframe["dsp"] > 0.0)  # Sell the peak (above SuperSmoother)
            )
            ehlers_trend = trend_mode & (dataframe["dsp"] < 0.0)  # Trend breaking down
        else:
            ehlers_cycle = cycle_mode & cycle_exit_trigger
            ehlers_trend = trend_mode & trend_exit_trigger

        cycle_exit = ehlers_cycle & cycle_exit_trigger
        trend_exit = ehlers_trend & trend_exit_trigger

        exit_mask = (cycle_exit | trend_exit) & dataframe["atr_ok"]
        exit_series = np.where(exit_mask, 1, 0)

        # save signals for displaying/debug
        dataframe["ehler_sell"] = exit_series
        dataframe["cycle_sell"] = np.where(cycle_exit, 1, 0)
        dataframe["trend_sell"] = np.where(trend_exit, 1, 0)

        return pd.Series(exit_series, index=dataframe.index)
