#
# This is a set of functions used to populate the indicators in a dataframe
# They are in a seperate file because I use the same set of indicators across several types of strategies, so it's
# just convenient to do it this way
#
import math

import numpy as np
import pandas as pd

import pywt
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d

from typing import Any, List
from pandas import DataFrame, Series
import sys
from pathlib import Path
from functools import reduce
from enum import Enum

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import freqtrade.vendor.qtpylib.indicators as qtpylib

import custom_indicators as cta
from finta import TA as fta

import legendary_ta as lta

from DataframeUtils import DataframeUtils
from scipy.stats import linregress


#################


# the type of dataset used for input
class DatasetType(Enum):
    DEFAULT = 0
    MINIMAL = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    CUSTOM1 = 5
    CUSTOM2 = 6
    CUSTOM3 = 7
    GAIN_ONLY = 8


class DataframePopulator:
    # global vars that control data generation. Ok to set these from a strategy

    startup_win = 128  # should be a power of 2
    win_size = 14
    runmode = ""  # set this to self.dp.runmode.value

    lookahead = 6

    n_profit_stddevs = 0.0
    n_loss_stddevs = 0.0

    dataframeUtils: DataframeUtils | None = None

    def __init__(self):
        super().__init__()
        self.dataframeUtils = DataframeUtils()

    #################

    def set_lookahead(self, lookahead):
        self.lookahead = lookahead

    def drop_index(self, dataframe: DataFrame) -> DataFrame:

        if "date" in dataframe.columns:
            # drop the date column and re-index
            dataframe = dataframe.drop("date", axis=1)
            dataframe.reset_index()
        return dataframe

    # --------------------------------

    def encode_cyclical_feature(self, df, column_name, cycle_length) -> DataFrame:
        """
        Encodes a cyclical feature using sine and cosine transformation.
            pd.DataFrame: The DataFrame with the new sin/cos columns added.
        """

        # Calculate the sin component (captures the primary cycle)
        df[f"{column_name}_sin"] = np.sin(2 * np.pi * df[column_name] / cycle_length)

        # Calculate the cos component (captures the phase shift/secondary cycle)
        df[f"{column_name}_cos"] = np.cos(2 * np.pi * df[column_name] / cycle_length)

        return df

    def process_date_features(self, dataframe: DataFrame) -> DataFrame:

        if "date" in dataframe.columns:
            # dataframe["month"] = dataframe["date"].dt.month
            dataframe["dow"] = dataframe["date"].dt.dayofweek
            dataframe["doy"] = dataframe["date"].dt.dayofyear
            # dataframe["hour"] = dataframe["date"].dt.hour
            # Minute of day: total minutes within the day (0-1439)
            dataframe["mod"] = (
                dataframe["date"].dt.hour * 60 + dataframe["date"].dt.minute
            )

            # dataframe = self.encode_cyclical_feature(dataframe, 'month', 12)
            dataframe = self.encode_cyclical_feature(dataframe, "dow", 7)
            # Use 365.25 to account for leap years (average year length)
            dataframe = self.encode_cyclical_feature(dataframe, "doy", 365.25)
            # dataframe = self.encode_cyclical_feature(dataframe, 'hour', 24)
            dataframe = self.encode_cyclical_feature(dataframe, "mod", 1440)

            # dataframe = dataframe.drop(columns=["dow", "hour", "mod"])
            dataframe = dataframe.drop(columns=["doy", "mod", "dow"])
        return dataframe

    # -------------------------
    def calc_dymi_fast(self, dataframe: DataFrame) -> tuple[Series, Series]:
        """
        Vectorized DYMI approximation using precomputed RSI(5..30).
        Matches the finta.DYMI period selection logic without per-row RSI calls.
        """
        close = dataframe["close"]
        sd = Series(close.rolling(5).std())
        asd = Series(sd.rolling(10).mean())
        # Use a small epsilon to avoid division by zero during volatility calc
        v = Series(sd / asd.replace(0, 1.0))
        v_round = Series(v.round().replace(0, 1.0))
        t_raw = Series(14.0 / v_round)
        t = Series(t_raw.fillna(14.0))
        t = Series(t.clip(5, 30).astype(int))

        rsi_by_period = {}
        for period in range(5, 31):
            rsi_by_period[period] = ta.RSI(dataframe, timeperiod=period).fillna(50.0)

        dymi = Series(np.nan, index=dataframe.index, dtype=float)
        for period, rsi in rsi_by_period.items():
            mask = Series(t == period)
            if mask.any():
                dymi.loc[mask] = rsi.loc[mask]

        return dymi.fillna(50.0), t

    def safe_cg(self, close, length=10) -> Series:
        """Vectorized Center of Gravity."""
        # Weighted sum: (1*Price[oldest] + 2*Price[next] ... + length*Price[newest])
        num = sum((i + 1) * close.shift(length - 1 - i) for i in range(length))
        den = close.rolling(length).sum()
        return Series(num / den, index=close.index)

    def safe_fisher(self, data: Series, length: int = 9) -> Series:
        """Fisher transform with recursive smoothing to filter noise."""
        # Normalized value (-1 to 1)
        value = self.rolling_normalize(data, length=length)

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

    def rolling_normalize(self, series: Series, length: int = 20) -> Series:
        """Normalizes a series to a strict [-1.0, 1.0] range using a rolling window."""
        roll = series.rolling(length, min_periods=min(5, length))
        s_min = roll.min()
        s_max = roll.max()
        s_range = (s_max - s_min).replace(0, 1e-12)
        # Wrap returned calculation in Series for type safety
        result = Series(2.0 * ((series - s_min) / s_range) - 1.0)
        return result.fillna(0.0).clip(-0.999, 0.999)

    # -------------------------

    # populate dataframe with desired technical indicators
    # Warning: do not use indicators that might produce 'inf' results, it messes up the scaling
    # Be careful about including anything that varies a lot across pairs (e.g. price/volume-related numbers)

    # use the minimal set of indicators needed to satisfy the trading signals logic and base class checks
    def add_minimal_indicators(self, dataframe: DataFrame) -> DataFrame:

        # Note that we manually scale anything is naturally bounded to [-1,1] range
        # This helps with normalisation later in the pipeline
        epsilon = 1e-6

        dataframe = self.process_date_features(dataframe)

        close_safe = np.maximum(dataframe["close"].values, 1e-6)
        dataframe["close_safe"] = close_safe
        wlen = 48  # rolling window for normalization

        dataframe["log_volume"] = np.log1p(dataframe["volume"])

        # Relative Volume
        dataframe["rvol"] = (
            dataframe["volume"] / dataframe["volume"].ewm(span=10, adjust=False).mean()
        )

        # dataframe["close_smoothed"] = (
        #     dataframe["close"].ewm(span=5, adjust=False).mean()
        # )

        # rolling MinMax normed close (to [-1, 1])
        dataframe["close_norm"] = self.rolling_normalize(
            dataframe["close_safe"], length=wlen
        )

        price_at_N = dataframe["close"].shift(self.win_size)

        # Calculate the fractional profit (Change in price / Starting price)
        profit_fraction = (price_at_N / (dataframe["close"] + epsilon)) - 1.0
        gain = np.nan_to_num(np.array(profit_fraction))

        # 'gain' is typically left unscaled or pre-scaled; we'll assume pre-normalized (PRE-NORMALIZED)
        dataframe["gain"] = gain

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / (dataframe["bb_middleband"] + epsilon)
        # bb_width stays at its natural ratio scale (typically [0, 0.10]).
        # The fixed-constant normalization that used to live here divided
        # by 0.028 and clipped to [-1, 1], which made the label-side
        # thresholds (e.g. labels_gbb's `bb_width_threshold = 0.03`)
        # mean "1.44% raw" instead of the intended "3% raw". main_scaler /
        # main_tensor_scaler now z-scores bb_width like other ratio
        # features (see BaseNNStrategy.pre_normalized_columns).

        # ATR (normalised to recent prices)
        atr_raw = ta.ATR(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=30,
        )

        # ATR: normalize by price, then use rolling MinMax (PRE-NORMALIZED, to [-1, 1])
        atr_raw_non_negative = np.maximum(0, atr_raw)
        safe_close = np.maximum(dataframe["close"], 1e-8)
        atr_normalized = atr_raw_non_negative / safe_close
        atr_normalized = atr_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Rolling MinMax normalization (lookahead-free, wlen already defined above)
        dataframe["atr_norm"] = self.rolling_normalize(atr_normalized, length=wlen)

        dataframe["atr_pct"] = atr_raw / safe_close
        dataframe["atr_pct_roll"] = (
            dataframe["atr_pct"].rolling(window=20, min_periods=1).mean()
        )

        # RSI
        rsi = ta.RSI(dataframe, timeperiod=self.win_size)

        dataframe["rsi_scaled"] = ((rsi / 50.0) - 1.0).fillna(0.0)

        # Williams %R
        wr = 0.02 * (self.williams_r(dataframe, period=self.win_size) + 50.0)

        # Fisher RSI (scaling RSI from 0-100 to approx -1 to 1)
        rsi = 0.1 * (rsi - 50)
        fisher_rsi = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R (PRE-NORMALIZED, approx -1 to 1)
        dataframe["fisher_wr"] = (wr + fisher_rsi) / 2.0

        # MFI (0-100 range)
        mfi = ta.MFI(dataframe, timeperiod=14)
        dataframe["mfi_scaled"] = (mfi / 50.0) - 1.0

        # RMI (0-100 range)
        rmi = cta.RMI(dataframe, length=self.win_size, mom=5)
        # srmi scaled to -1 to 1 (PRE-NORMALIZED)
        srmi = 2.0 * (rmi - 50.0) / 100.0
        dataframe["guard_metric"] = srmi
        # guard_metric smooth bimodal split (was hard clip; produced a
        # Dirac at 0 in the opposite-class samples that no continuous-
        # output GAN could match — real class-2 σ(guard_metric_pos) was
        # ~0.026, GAN synth was ~0.14, σ_ratio ~5x, persistent across
        # all filter sweeps). Softplus preserves the same sign-magnitude
        # signal for the classifier (high for positive guard_metric, low
        # for negative) but every row has a small non-zero value, so the
        # σ never collapses and the GAN can model the distribution
        # without a discontinuity at 0.
        _gm_scale = 0.2
        _gm = dataframe["guard_metric"]
        dataframe["guard_metric_pos"] = _gm_scale * np.log1p(np.exp(_gm / _gm_scale))
        dataframe["guard_metric_neg"] = _gm_scale * np.log1p(np.exp(-_gm / _gm_scale))

        # MACD (Absolute values - need scaling)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Fast EMA for direction
        dataframe["ema_fast"] = ta.EMA(dataframe["close"], timeperiod=20)

        # use MinMax normalised versions of macd indicators (PRE-NORMALIZED, to [-1, 1])
        # wlen already defined above
        for col in ["gain", "log_volume", "macd", "macdhist", "macdsignal", "ema_fast"]:
            dataframe[f"{col}_norm"] = self.rolling_normalize(
                dataframe[col], length=wlen
            )
            dataframe[col] = dataframe[col].fillna(0.0)

        # macd_norm bimodal split: pos = max(x, 0), neg = -min(x, 0).
        # Two unimodal features (each in [0, 1]) replace one bimodal
        # feature in the NN include_list — the GAN's MLP regression
        # can model unimodal distributions cleanly, and the classifier
        # gets explicit sign-magnitude features instead of having to
        # derive the sign internally.
        dataframe["macd_pos"] = dataframe["macd_norm"].clip(lower=0)
        dataframe["macd_neg"] = (-dataframe["macd_norm"]).clip(lower=0)

        # Stoch fast (0-100 range)
        stoch_fast = ta.STOCHF(dataframe, fastk_period=15)
        fastd = stoch_fast["fastd"]
        dataframe["fastd_scaled"] = (fastd / 50.0) - 1.0
        fastk = stoch_fast["fastk"]
        dataframe["fastk_scaled"] = (fastk / 50.0) - 1.0
        dataframe["fast_diff"] = dataframe["fastd_scaled"] - dataframe["fastk_scaled"]

        # ADX (0-100 range)
        adx = ta.ADX(dataframe)
        dataframe["adx"] = adx

        # ADX scaling (adjust constant if implemented manually):
        ADX_FIXED_MAX = 55.0  # Empirical

        dataframe["adx_scaled"] = (adx / ADX_FIXED_MAX).clip(0, 1) * 2.0 - 1.0

        # Aroon Oscillator (-100 to 100 range)
        aroonosc = ta.AROONOSC(dataframe, timeperiod=30)
        dataframe["aroonosc"] = aroonosc
        dataframe["aroonosc_scaled"] = aroonosc / 100.0

        # Price-based features that generalize across pairs
        # BB position (PRE-NORMALIZED, 0-1 scale)
        bb_range = dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        dataframe["bb_position"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) / (bb_range + epsilon)
        ).clip(0, 1)

        # OBV (On-Balance Volume) - use percentage change, then rolling MinMax (PRE-NORMALIZED, to [-1, 1])
        # Cumulative indicators grow unbounded, so use percentage change instead of raw values
        obv = ta.OBV(dataframe)
        obv_pct_change = obv.pct_change(periods=1)
        # Replace inf and -inf values (can occur when previous value is 0)
        obv_pct_change = obv_pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Rolling MinMax normalization (lookahead-free)
        dataframe["obv_scaled"] = self.rolling_normalize(obv_pct_change, length=wlen)

        # AD (Accumulation/Distribution) - use percentage change, then rolling MinMax (PRE-NORMALIZED, to [-1, 1])
        ad = ta.AD(dataframe)
        ad_pct_change = ad.pct_change(periods=1)
        # Replace inf and -inf values (can occur when previous value is 0)
        ad_pct_change = ad_pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Rolling MinMax normalization (lookahead-free)
        dataframe["ad_scaled"] = self.rolling_normalize(ad_pct_change, length=wlen)

        # VWAP - normalize by close price (ratio) to make pair-independent
        # Clip using rolling percentiles (lookahead-free)
        vwap = qtpylib.rolling_vwap(dataframe)
        vwap_ratio = (vwap / close_safe) - 1.0  # deviation from current price
        dataframe["vwap_ratio"] = self.rolling_clip(
            vwap_ratio, window=wlen, lower_pct=5, upper_pct=95
        )
        # Bimodal split — same rationale as macd_pos / macd_neg above.
        dataframe["vwap_pos"] = dataframe["vwap_ratio"].clip(lower=0)
        dataframe["vwap_neg"] = (-dataframe["vwap_ratio"]).clip(lower=0).fillna(0.0)

        # Directional Indicators - scale from 0-100 to -1 to 1 (PRE-NORMALIZED)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["plus_di_scaled"] = (plus_di / 50.0) - 1.0
        dataframe["minus_di_scaled"] = (minus_di / 50.0) - 1.0
        # DI difference: clip using rolling percentiles (lookahead-free)
        di_diff = dataframe["plus_di_scaled"] - dataframe["minus_di_scaled"]
        dataframe["di_diff_scaled"] = self.rolling_clip(
            di_diff, window=wlen, lower_pct=5, upper_pct=95
        ).fillna(0.0)
        dataframe["plus_di_scaled"] = dataframe["plus_di_scaled"].fillna(0.0)
        dataframe["minus_di_scaled"] = dataframe["minus_di_scaled"].fillna(0.0)

        # SAR - normalize by close price (ratio) to make pair-independent
        # Clip using rolling percentiles (lookahead-free)
        sar = ta.SAR(dataframe, acceleration=0.02, maximum=0.20)
        sar_ratio = (sar / close_safe) - 1.0  # deviation from current price
        dataframe["sar_ratio"] = self.rolling_clip(
            sar_ratio, window=wlen, lower_pct=5, upper_pct=95
        ).fillna(0.0)

        # SAR trend - already binary (0/1), pair-independent
        sar_trend = (dataframe["close"] > sar).astype(int)
        dataframe["sar_trend"] = sar_trend

        # RSI SMA - already using rsi_scaled, so it's normalized
        rsi_sma = dataframe["rsi_scaled"].rolling(5).mean()
        dataframe["rsi_sma"] = rsi_sma.fillna(0.0)

        # Spread analysis (using high-low as proxy for bid-ask spread)
        # Already pair-independent as percentage, but needs normalization to [-1, 1]
        spread = dataframe["high"] - dataframe["low"]
        spread_pct_raw = spread / close_safe
        # Spread MA - normalize the raw rolling mean using rolling percentiles
        spread_ma_raw = spread_pct_raw.rolling(20).mean()
        dataframe["spread_ma"] = self.rolling_clip(
            spread_ma_raw, window=wlen, lower_pct=5, upper_pct=95
        ).fillna(0.0)

        # # Price position in range (PRE-NORMALIZED, 0-1 scale)
        # lookback = 48
        # price_min = dataframe['low'].rolling(lookback).min()
        # price_max = dataframe['high'].rolling(lookback).max()
        # price_range = price_max - price_min
        # dataframe['price_position'] = (
        #     (dataframe['close'] - price_min) /
        #     (price_range + epsilon)
        # ).clip(0, 1)

        # convert to [-1,1] range
        dataframe["bb_position"] = 2.0 * (dataframe["bb_position"] - 0.5)
        # dataframe['price_position'] = 2.0 * (dataframe['price_position'] - 0.5)

        # Fill any remaining NaN values in all features
        dataframe["bb_position"] = dataframe["bb_position"].fillna(0.0)
        # dataframe['price_position'].fillna(0.0, inplace=True)
        # price_ma*_log already initialized to 0.0, so no NaN to fill

        # CCI (common period is 20)
        cci_period = 20
        dataframe["cci"] = ta.CCI(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=cci_period,
        )

        # 2. Scale the CCI to [-1, 1] using clipping and then a simple division
        # We clip it at a common extreme (e.g., 200) and then divide by that clip value.
        # This assumes that any value beyond 200 is treated as max extremity (1 or -1).
        clip_val = 200

        # Clip the values
        cci_clipped = dataframe["cci"].clip(lower=-clip_val, upper=clip_val)

        # Scale the clipped values from [-200, 200] to [-1, 1]
        dataframe["cci_scaled"] = (cci_clipped / clip_val).fillna(0.0)

        # Hilbert Transform Trend Mode (1 or 0)
        dataframe["trend_mode"] = ta.HT_TRENDMODE(dataframe["close"])

        # DYMI (Lag-less adaptive RSI)
        dymi, dymi_period = self.calc_dymi_fast(dataframe)
        dataframe["dymi"] = dymi
        dataframe["dymi_period"] = dymi_period
        dataframe["dymi_scaled"] = (dymi / 50.0) - 1.0

        # Ehlers Filters & Transforms
        # 1. Adaptive Super Smoother (reduces noise while keeping lag low)
        ss_close = self.adaptive_super_smoother(dataframe["close"], dymi_period)
        dataframe["ss_close"] = ss_close

        # 2. Difference between Price and SS (extracts the cycle/noise component)
        ss_diff = dataframe["close"] - ss_close

        # 3. Fisher Transform of the Price-SS difference (identifies cyclic extremes)
        # Normalized using rolling min/max to stay in [-1, 1]
        dataframe["fisher_ss"] = self.rolling_normalize(self.safe_fisher(ss_diff))

        # 4. Center of Gravity of the Price-SS difference (finds balance point of the cycle)
        # Normalized using rolling min/max to stay in [-1, 1]
        dataframe["cg_ss"] = self.rolling_normalize(self.safe_cg(ss_diff))

        return dataframe

    # ------------------------------
    # 'small' set of indicators - basically, the best-known ones
    def add_small_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)

        # DWT model
        # if in backtest or hyperopt, then we have to do rolling calculations
        if self.runmode in ("hyperopt", "backtest", "plot"):
            # dataframe['dwt'] = dataframe['close'].rolling(window=self.startup_win).apply(self.roll_get_dwt)
            dataframe["dwt"] = (
                dataframe["mid"]
                .rolling(window=self.startup_win)
                .apply(self.roll_get_dwt)
            )
        else:
            # dataframe['dwt'] = self.get_dwt(dataframe['close'])
            dataframe["dwt"] = self.get_dwt(dataframe["mid"])

        dataframe["dwt_gain"] = (
            100.0
            * (dataframe["dwt"] - dataframe["dwt"].shift())
            / dataframe["dwt"].shift()
        )
        dataframe["dwt_profit"] = dataframe["dwt_gain"].clip(lower=0.0)
        dataframe["dwt_loss"] = dataframe["dwt_gain"].clip(upper=0.0)

        dataframe["dwt_profit_mean"] = (
            dataframe["dwt_profit"].rolling(self.win_size, center=False).mean()
        )
        dataframe["dwt_profit_std"] = (
            dataframe["dwt_profit"].rolling(self.win_size, center=False).std()
        )
        dataframe["dwt_loss_mean"] = (
            dataframe["dwt_loss"].rolling(self.win_size, center=False).mean()
        )
        dataframe["dwt_loss_std"] = (
            dataframe["dwt_loss"].rolling(self.win_size, center=False).std()
        )

        # (Local) Profit & Loss thresholds are used extensively, do not remove!
        dataframe["profit_threshold"] = dataframe[
            "dwt_profit_mean"
        ] + self.n_profit_stddevs * abs(dataframe["dwt_profit_std"])

        dataframe["loss_threshold"] = dataframe[
            "dwt_loss_mean"
        ] - self.n_loss_stddevs * abs(dataframe["dwt_loss_std"])

        # Sequences of consecutive up/downs
        dataframe["dwt_dir"] = 0.0
        dataframe["dwt_dir"] = np.where(dataframe["dwt"].diff() > 0, 1.0, -1.0)

        dataframe["dwt_dir_up"] = dataframe["dwt_dir"].clip(lower=0.0)
        dataframe["dwt_nseq_up"] = dataframe["dwt_dir_up"] * (
            dataframe["dwt_dir_up"]
            .groupby(
                (dataframe["dwt_dir_up"] != dataframe["dwt_dir_up"].shift()).cumsum()
            )
            .cumcount()
            + 1
        )
        dataframe["dwt_nseq_up"] = dataframe["dwt_nseq_up"].clip(
            lower=0.0, upper=20.0
        )  # removes startup artifacts

        dataframe["dwt_dir_dn"] = abs(dataframe["dwt_dir"].clip(upper=0.0))
        dataframe["dwt_nseq_dn"] = dataframe["dwt_dir_dn"] * (
            dataframe["dwt_dir_dn"]
            .groupby(
                (dataframe["dwt_dir_dn"] != dataframe["dwt_dir_dn"].shift()).cumsum()
            )
            .cumcount()
            + 1
        )
        dataframe["dwt_nseq_dn"] = dataframe["dwt_nseq_dn"].clip(lower=0.0, upper=20.0)

        # rolling linear slope of the DWT (i.e. average trend) of near-past
        dataframe["dwt_slope"] = (
            dataframe["dwt"].rolling(window=6).apply(self.roll_get_slope)
        )

        # moving averages
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=self.win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(self.win_size).std()

        # Donchian Channels
        dataframe["dc_upper"] = ta.MAX(dataframe["high"], timeperiod=self.win_size)
        dataframe["dc_lower"] = ta.MIN(dataframe["low"], timeperiod=self.win_size)
        dataframe["dc_mid"] = ta.TEMA(
            ((dataframe["dc_upper"] + dataframe["dc_lower"]) / 2),
            timeperiod=self.win_size,
        )

        dataframe["dcbb_dist_upper"] = dataframe["dc_upper"] - dataframe["bb_upperband"]
        dataframe["dcbb_dist_lower"] = dataframe["dc_lower"] - dataframe["bb_lowerband"]

        # Fibonacci Levels (of Donchian Channel)
        dataframe["dc_dist"] = dataframe["dc_upper"] - dataframe["dc_lower"]
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        return dataframe

    # ------------------------------

    def add_default_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_small_indicators(dataframe)

        # Recent min/max (using only past data to avoid lookahead bias)
        dataframe["recent_min"] = (
            dataframe["close"].rolling(window=self.win_size, center=False).min()
        )
        dataframe["recent_max"] = (
            dataframe["close"].rolling(window=self.win_size, center=False).max()
        )

        # Bollinger Bands (must include these)
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["bb_gain"] = (
            dataframe["bb_upperband"] - dataframe["close"]
        ) / dataframe["close"]
        dataframe["bb_loss"] = (
            dataframe["bb_lowerband"] - dataframe["close"]
        ) / dataframe["close"]

        # moving averages
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=self.win_size)
        # dataframe['tema_stddev'] = dataframe['tema'].rolling(self.win_size).std()

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        # Normalize to [-1, 1] then rescale to [0, 1] for standard Stochastic formula
        stochrsi = (self.rolling_normalize(dataframe["rsi"], length=period) + 1.0) / 2.0
        dataframe["srsi_k"] = stochrsi.rolling(SmoothK, center=False).mean() * 100
        dataframe["srsi_d"] = dataframe["srsi_k"].rolling(smoothD, center=False).mean()

        # Donchian Channels
        dataframe["dc_upper"] = ta.MAX(dataframe["high"], timeperiod=self.win_size)
        dataframe["dc_lower"] = ta.MIN(dataframe["low"], timeperiod=self.win_size)
        dataframe["dc_mid"] = ta.TEMA(
            ((dataframe["dc_upper"] + dataframe["dc_lower"]) / 2),
            timeperiod=self.win_size,
        )

        dataframe["dcbb_dist_upper"] = dataframe["dc_upper"] - dataframe["bb_upperband"]
        dataframe["dcbb_dist_lower"] = dataframe["dc_lower"] - dataframe["bb_lowerband"]

        # Fibonacci Levels (of Donchian Channel)
        dataframe["dc_dist"] = dataframe["dc_upper"] - dataframe["dc_lower"]
        # dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236  # Highest Fib
        # dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382  # Centre High Fib
        # dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618  # Centre Low Fib
        # dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764  # Low Fib

        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # SMA
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        # Plus Directional Indicator / Movement
        dataframe["dm_plus"] = ta.PLUS_DM(dataframe)
        dataframe["di_plus"] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe["dm_minus"] = ta.MINUS_DM(dataframe)
        dataframe["di_minus"] = ta.MINUS_DI(dataframe)
        dataframe["dm_delta"] = dataframe["dm_plus"] - dataframe["dm_minus"]
        dataframe["di_delta"] = dataframe["di_plus"] - dataframe["di_minus"]

        # Volume Flow Indicator (VFI) for volume based on the direction of price movement
        dataframe["vfi"] = fta.VFI(dataframe, period=14)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

        # Oscillators

        # EWO
        dataframe["ewo"] = self.ewo(dataframe, 50, 200)

        # Ultimate Oscillator
        dataframe["uo"] = ta.ULTOSC(dataframe)

        # Awesome Oscillator
        dataframe["ao"] = qtpylib.awesome_oscillator(dataframe)

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe["cci"] = ta.CCI(dataframe)

        # Legenadry TA indicators
        dataframe = lta.fisher_cg(dataframe)
        dataframe = lta.exhaustion_bars(dataframe)
        dataframe = lta.smi_momentum(dataframe)
        dataframe = lta.pinbar(dataframe, dataframe["smi"])
        dataframe = lta.breakouts(dataframe)

        return dataframe

    # ------------------------------

    def add_medium_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_small_indicators(dataframe)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        # Plus Directional Indicator / Movement
        dataframe["dm_plus"] = ta.PLUS_DM(dataframe)
        dataframe["di_plus"] = ta.PLUS_DI(dataframe)

        # Minus Directional Indicator / Movement
        dataframe["dm_minus"] = ta.MINUS_DM(dataframe)
        dataframe["di_minus"] = ta.MINUS_DI(dataframe)
        dataframe["dm_delta"] = dataframe["dm_plus"] - dataframe["dm_minus"]
        dataframe["di_delta"] = dataframe["di_plus"] - dataframe["di_minus"]

        # Volume Flow Indicator (VFI) for volume based on the direction of price movement
        dataframe["vfi"] = fta.VFI(dataframe, period=14)

        # Oscillators

        # EWO
        dataframe["ewo"] = self.ewo(dataframe, 50, 200)

        # Ultimate Oscillator
        dataframe["uo"] = ta.ULTOSC(dataframe)

        # Aroon, Aroon Oscillator
        aroon = ta.AROON(dataframe)
        dataframe["aroonup"] = aroon["aroonup"]
        dataframe["aroondown"] = aroon["aroondown"]
        dataframe["aroonosc"] = ta.AROONOSC(dataframe)

        # Awesome Oscillator
        dataframe["ao"] = qtpylib.awesome_oscillator(dataframe)

        return dataframe

    # ------------------------------

    def add_large_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_medium_indicators(dataframe)

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        # Normalize to [-1, 1] then rescale to [0, 1] for standard Stochastic formula
        stochrsi = (self.rolling_normalize(dataframe["rsi"], length=period) + 1.0) / 2.0
        dataframe["srsi_k"] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe["srsi_d"] = dataframe["srsi_k"].rolling(smoothD).mean()

        # RSI
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # SMA
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

        # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe["cci"] = ta.CCI(dataframe)

        # Legendary TA indicators
        dataframe = lta.fisher_cg(dataframe)
        dataframe = lta.exhaustion_bars(dataframe)
        dataframe = lta.smi_momentum(dataframe)
        dataframe = lta.pinbar(dataframe, dataframe["smi"])
        dataframe = lta.breakouts(dataframe)

        return dataframe

    # ------------------------------

    # for experimentation
    def add_custom1_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["bb_gain"] = (
            dataframe["bb_upperband"] - dataframe["close"]
        ) / dataframe["close"]
        dataframe["bb_loss"] = (
            dataframe["bb_lowerband"] - dataframe["close"]
        ) / dataframe["close"]

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        """"""
        # Keltner Channels (Warning: these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # Donchian Channels
        dataframe["dc_upper"] = ta.MAX(dataframe["high"], timeperiod=self.win_size)
        dataframe["dc_lower"] = ta.MIN(dataframe["low"], timeperiod=self.win_size)
        dataframe["dc_mid"] = ta.TEMA(
            ((dataframe["dc_upper"] + dataframe["dc_lower"]) / 2),
            timeperiod=self.win_size,
        )

        """"""

        return dataframe

    # ------------------------------

    def add_custom2_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Add comprehensive technical indicators for enhanced anomaly detection
        """
        dataframe = self.add_minimal_indicators(dataframe)

        # Add all enhanced indicators
        self._add_volatility_indicators(dataframe)
        self._add_momentum_indicators(dataframe)
        self._add_trend_indicators(dataframe)
        self._add_volume_indicators(dataframe)
        self._add_price_action_features(dataframe)
        self._add_time_based_features(dataframe)
        self._add_cross_timeframe_features(dataframe)
        self._add_market_microstructure_features(dataframe)

        return dataframe
        # ------------------------------

    # Helper function for rolling percentile-based clipping (lookahead-free)
    def rolling_clip(self, series, window=48, lower_pct=5, upper_pct=95):
        """Clip values using rolling percentiles (only uses past data)"""
        lower_bound = series.rolling(window=window, min_periods=1).quantile(
            lower_pct / 100.0
        )
        upper_bound = series.rolling(window=window, min_periods=1).quantile(
            upper_pct / 100.0
        )
        clipped = series.clip(lower=lower_bound, upper=upper_bound)
        # Normalize to [-1, 1] range using rolling bounds
        range_val = upper_bound - lower_bound
        range_val = range_val.replace(0, 1.0)  # Avoid division by zero
        normalized = 2.0 * (clipped - lower_bound) / range_val - 1.0
        return normalized.fillna(0.0)

    def add_custom3_indicators(self, dataframe: DataFrame) -> DataFrame:

        dataframe = self.add_minimal_indicators(dataframe)

        close_safe = np.maximum(dataframe["close"].values, 1e-6)
        wlen = 48  # rolling window for percentile-based clipping

        # add some extras for experimentation (not in include_list)
        # Williams %R - already in -100 to 0 range, scale to -1 to 1 (PRE-NORMALIZED)
        willr = ta.WILLR(dataframe, timeperiod=14)
        dataframe["willr_scaled"] = (willr / 100.0).fillna(0.0)  # -1 to 0 range

        # Price-Volume trend - already binary (0/1), pair-independent
        # Ensure volume_sma exists (create if not present)
        if "volume_sma" not in dataframe.columns:
            dataframe["volume_sma"] = dataframe["volume"].rolling(20).mean()

        # Normalize volume_sma using log + rolling MinMax (lookahead-free)
        volume_sma_log = np.log1p(dataframe["volume_sma"].fillna(0))
        dataframe["volume_sma_norm"] = self.rolling_normalize(
            volume_sma_log, length=wlen
        )

        # Use raw volume_sma for pv_trend calculation (comparison needs raw values)
        pv_trend = (
            (dataframe["close"].pct_change() > 0)
            == (dataframe["volume"] > dataframe["volume_sma"])
        ).astype(int)
        dataframe["pv_trend"] = pv_trend

        # Spread analysis (using high-low as proxy for bid-ask spread)
        # Already pair-independent as percentage, but needs normalization to [-1, 1]
        spread = dataframe["high"] - dataframe["low"]
        spread_pct_raw = spread / close_safe
        # Normalize using rolling percentiles (lookahead-free)
        dataframe["spread_pct"] = self.rolling_clip(
            spread_pct_raw, window=wlen, lower_pct=5, upper_pct=95
        ).fillna(0.0)

        return dataframe

    def _add_volatility_indicators(self, dataframe: DataFrame):
        """
        Volatility indicators help detect market stress and regime changes
        - ATR: Measures volatility magnitude
        - Bollinger Bands: Dynamic support/resistance levels
        - BB Width: Volatility expansion/contraction
        """
        import talib.abstract as ta
        import numpy as np

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        bb_bands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb_bands["upperband"]
        dataframe["bb_lower"] = bb_bands["lowerband"]
        dataframe["bb_middle"] = bb_bands["middleband"]

        # Handle division by zero for Bollinger Band calculations
        bb_diff = dataframe["bb_upper"] - dataframe["bb_lower"]
        dataframe["bb_width"] = bb_diff / dataframe["bb_middle"].replace(0, np.nan)
        dataframe["bb_position"] = (
            dataframe["close"] - dataframe["bb_lower"]
        ) / bb_diff.replace(0, np.nan)

    def _add_momentum_indicators(self, dataframe: DataFrame):
        """
        Momentum indicators detect overbought/oversold conditions
        - RSI: Most popular momentum oscillator
        - MFI: Volume-weighted RSI
        - CCI: Commodity Channel Index for cyclical movements
        - Williams %R: Another momentum oscillator
        """
        import talib.abstract as ta

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=14)
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=14)

        # Momentum divergence indicators
        dataframe["rsi_sma"] = dataframe["rsi"].rolling(5).mean()
        dataframe["price_rsi_div"] = (
            (dataframe["close"].pct_change(5) > 0) != (dataframe["rsi"].diff(5) > 0)
        ).astype(int)

    def _add_trend_indicators(self, dataframe: DataFrame):
        """
        Trend indicators identify market direction and strength
        - ADX: Trend strength (not direction)
        - DI+/DI-: Directional indicators
        - MACD: Moving average convergence divergence
        - Parabolic SAR: Trend reversal indicator
        """
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["di_diff"] = dataframe["plus_di"] - dataframe["minus_di"]

        # macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        # dataframe['macd'] = macd['macd']
        # dataframe['macd_signal'] = macd['macdsignal']
        # dataframe['macd_hist'] = macd['macdhist']

        dataframe["sar"] = ta.SAR(dataframe, acceleration=0.02, maximum=0.20)
        dataframe["sar_trend"] = (dataframe["close"] > dataframe["sar"]).astype(int)

    def _add_volume_indicators(self, dataframe: DataFrame):
        """
        Volume indicators show institutional activity and confirm trends
        - OBV: On-Balance Volume (cumulative volume flow)
        - VWAP: Volume Weighted Average Price
        - Volume Rate: Current vs average volume
        - A/D Line: Accumulation/Distribution Line
        """
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["ad"] = ta.AD(dataframe)

        # Calculate VWAP manually (more control)
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        volume_cumsum = dataframe["volume"].cumsum()
        dataframe["vwap"] = (
            typical_price * dataframe["volume"]
        ).cumsum() / volume_cumsum.replace(0, np.nan)

        # Volume analysis
        dataframe["volume_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe[
            "volume_sma"
        ].replace(0, np.nan)
        # Calculate volume momentum, handling division by zero
        volume_pct_change = dataframe["volume"].pct_change(5)
        # Replace infinities with 0 (when previous volume was 0)
        volume_pct_change = volume_pct_change.replace([np.inf, -np.inf], 0)
        dataframe["volume_momentum"] = volume_pct_change.fillna(0)

        # Price-Volume relationship
        dataframe["pv_trend"] = (
            (dataframe["close"].pct_change() > 0)
            == (dataframe["volume"] > dataframe["volume_sma"])
        ).astype(int)

        # Volume Flow Indicator (VFI) for volume based on the direction of price movement
        dataframe["vfi"] = fta.VFI(dataframe, period=14)

    def _add_price_action_features(self, dataframe: DataFrame):
        """
        Price action features capture candlestick patterns and market sentiment
        - Body size: Strength of directional movement
        - Shadows: Rejection levels and market indecision
        - Price position: Where close is relative to high/low
        """
        dataframe["body_size"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["body_size_pct"] = dataframe["body_size"] / dataframe["close"]

        dataframe["upper_shadow"] = dataframe["high"] - dataframe[
            ["open", "close"]
        ].max(axis=1)
        dataframe["lower_shadow"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        )

        dataframe["upper_shadow_pct"] = dataframe["upper_shadow"] / dataframe["close"]
        dataframe["lower_shadow_pct"] = dataframe["lower_shadow"] / dataframe["close"]

        # Price position within the range
        range_diff = dataframe["high"] - dataframe["low"]
        dataframe["price_position"] = (
            dataframe["close"] - dataframe["low"]
        ) / range_diff.replace(0, np.nan)

        # Candle patterns
        dataframe["is_green"] = (dataframe["close"] > dataframe["open"]).astype(int)
        dataframe["is_doji"] = (dataframe["body_size_pct"] < 0.005).astype(int)
        dataframe["is_hammer"] = (
            (dataframe["lower_shadow"] > 2 * dataframe["body_size"])
            & (dataframe["upper_shadow"] < dataframe["body_size"])
        ).astype(int)

    def _add_time_based_features(self, dataframe: DataFrame):
        """
        Time-based features capture market cycles and trading sessions
        - Hour: Intraday patterns (market open, close, lunch)
        - Day of week: Weekly patterns (Monday blues, Friday effect)
        - Month: Seasonal patterns
        """
        import pandas as pd

        # Ensure we have a datetime index or date column
        if "date" not in dataframe.columns:
            dataframe = dataframe.reset_index()

        if pd.api.types.is_datetime64_any_dtype(dataframe["date"]):
            dataframe["hour"] = dataframe["date"].dt.hour
            dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
            dataframe["month"] = dataframe["date"].dt.month

            # Trading session indicators (assuming UTC time)
            dataframe["is_asian_session"] = (
                (dataframe["hour"] >= 0) & (dataframe["hour"] < 8)
            ).astype(int)
            dataframe["is_london_session"] = (
                (dataframe["hour"] >= 8) & (dataframe["hour"] < 16)
            ).astype(int)
            dataframe["is_ny_session"] = (
                (dataframe["hour"] >= 13) & (dataframe["hour"] < 21)
            ).astype(int)
        else:
            # Fallback if date is not datetime
            dataframe["hour"] = 12  # Default to midday
            dataframe["day_of_week"] = 2  # Default to Wednesday
            dataframe["month"] = 6  # Default to June
            dataframe["is_asian_session"] = 0
            dataframe["is_london_session"] = 1
            dataframe["is_ny_session"] = 0

    def _add_cross_timeframe_features(self, dataframe: DataFrame):
        """
        Cross-timeframe features provide context from higher timeframes
        - 1h average: Short-term trend
        - 4h average: Medium-term trend
        - Daily average: Long-term trend
        """
        # 1-hour averages (12 x 5min candles)
        dataframe["close_1h"] = dataframe["close"].rolling(12).mean()
        dataframe["volume_1h"] = dataframe["volume"].rolling(12).mean()
        dataframe["high_1h"] = dataframe["high"].rolling(12).max()
        dataframe["low_1h"] = dataframe["low"].rolling(12).min()

        # 4-hour averages (48 x 5min candles)
        dataframe["close_4h"] = dataframe["close"].rolling(48).mean()
        dataframe["volume_4h"] = dataframe["volume"].rolling(48).mean()
        dataframe["high_4h"] = dataframe["high"].rolling(48).max()
        dataframe["low_4h"] = dataframe["low"].rolling(48).min()

        # Daily averages (288 x 5min candles)
        dataframe["close_1d"] = dataframe["close"].rolling(288).mean()
        dataframe["volume_1d"] = dataframe["volume"].rolling(288).mean()

        # Cross-timeframe momentum
        dataframe["momentum_1h"] = (
            dataframe["close"] / dataframe["close_1h"] - 1
        ) * 100
        dataframe["momentum_4h"] = (
            dataframe["close"] / dataframe["close_4h"] - 1
        ) * 100
        dataframe["momentum_1d"] = (
            dataframe["close"] / dataframe["close_1d"] - 1
        ) * 100

    def _add_market_microstructure_features(self, dataframe: DataFrame):
        """
        Market microstructure features capture liquidity and market depth
        - Spread: Bid-ask spread proxy (high-low)
        - Price impact: How price moves with volume
        - Liquidity: Market depth indicators
        """
        import numpy as np

        # Spread analysis (using high-low as proxy for bid-ask spread)
        dataframe["spread"] = dataframe["high"] - dataframe["low"]
        dataframe["spread_pct"] = dataframe["spread"] / dataframe["close"].replace(
            0, np.nan
        )
        dataframe["spread_ma"] = dataframe["spread_pct"].rolling(20).mean()
        dataframe["spread_relative"] = dataframe["spread_pct"] / dataframe[
            "spread_ma"
        ].replace(0, np.nan)

        # Price momentum
        # Calculate price momentum, handling division by zero
        price_momentum_1 = dataframe["close"].pct_change(1)
        price_momentum_1 = price_momentum_1.replace([np.inf, -np.inf], 0)
        dataframe["price_momentum_1"] = price_momentum_1.fillna(0)

        price_momentum_5 = dataframe["close"].pct_change(5)
        price_momentum_5 = price_momentum_5.replace([np.inf, -np.inf], 0)
        dataframe["price_momentum_5"] = price_momentum_5.fillna(0)

        price_momentum_20 = dataframe["close"].pct_change(20)
        price_momentum_20 = price_momentum_20.replace([np.inf, -np.inf], 0)
        dataframe["price_momentum_20"] = price_momentum_20.fillna(0)

        # Volatility clustering (GARCH-like effects)
        dataframe["returns_squared"] = dataframe["price_momentum_1"] ** 2
        dataframe["volatility_cluster"] = (
            dataframe["returns_squared"].rolling(20).mean()
        )

        # Liquidity proxies
        volume_value = dataframe["volume"] * dataframe["close"]
        dataframe["amihud_illiq"] = abs(
            dataframe["price_momentum_1"]
        ) / volume_value.replace(0, np.nan)
        volume_log = np.log(1 + dataframe["volume"].replace(0, 1))  # Avoid log(0)
        dataframe["price_impact"] = abs(dataframe["price_momentum_1"]) / volume_log

        # Replace any infinite values
        dataframe = dataframe.replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------------
    def add_gain_only_indicators(self, dataframe: DataFrame) -> DataFrame:

        if "gain" not in dataframe.columns:
            bgain = (
                100.0
                * (dataframe["close"] - dataframe["close"].shift(self.lookahead))
                / dataframe["close"].shift(self.lookahead)
            )

            gain = np.nan_to_num(np.array(bgain))
            dataframe["gain"] = self.smooth(gain, 4)
            dataframe["gain"] = dataframe["gain"].round(2)

        dataframe = dataframe[["gain"]].copy()
        return dataframe

    # ------------------------------

    def add_indicators(
        self, dataframe: DataFrame, dataset_type=DatasetType.DEFAULT
    ) -> DataFrame:

        # print(f"dataset_type: {dataset_type}")

        if dataset_type == DatasetType.DEFAULT:
            dataframe = self.add_default_indicators(dataframe)
        elif dataset_type == DatasetType.MINIMAL:
            dataframe = self.add_minimal_indicators(dataframe)
        elif dataset_type == DatasetType.SMALL:
            dataframe = self.add_small_indicators(dataframe)
        elif dataset_type == DatasetType.MEDIUM:
            dataframe = self.add_medium_indicators(dataframe)
        elif dataset_type == DatasetType.LARGE:
            dataframe = self.add_large_indicators(dataframe)
        elif dataset_type == DatasetType.CUSTOM1:
            dataframe = self.add_custom1_indicators(dataframe)
        elif dataset_type == DatasetType.CUSTOM2:
            dataframe = self.add_custom2_indicators(dataframe)
        elif dataset_type == DatasetType.CUSTOM3:
            dataframe = self.add_custom3_indicators(dataframe)
        # elif dataset_type == DatasetType.GAIN_ONLY:
        #     dataframe = self.add_gain_only_indicators(dataframe)
        else:
            print(f"    ERROR: Unknown dataset type: {dataset_type}")

        # TODO: remove/fix any columns that contain 'inf'
        self.dataframeUtils.check_inf(dataframe)

        # fix NaNs
        dataframe = dataframe.fillna(0.0)

        return dataframe

    ################################

    # 'hidden' indicators. These are ostensibly backward looking, but may inadvertently use means, smoothing etc.
    def add_hidden_indicators(self, dataframe: DataFrame) -> DataFrame:

        if "mid" not in dataframe:
            print("")
            print("*** ERR: missing columns (mid)")
            print(f"    cols: {dataframe.columns.values}")
            print("")

        dataframe["fwd_dwt"] = self.get_dwt(dataframe["mid"])

        dataframe["dwt_deriv"] = np.gradient(dataframe["dwt"])
        # dataframe['dwt_deriv'] = np.gradient(dataframe['dwt'])
        dataframe["dwt_top"] = np.where(
            qtpylib.crossed_below(dataframe["dwt_deriv"], 0.0), 1, 0
        )
        dataframe["dwt_bottom"] = np.where(
            qtpylib.crossed_above(dataframe["dwt_deriv"], 0.0), 1, 0
        )

        # dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['mid']) / dataframe['mid']
        dataframe["dwt_diff"] = (
            100.0 * (dataframe["fwd_dwt"] - dataframe["mid"]) / dataframe["mid"]
        )
        # dataframe['dwt_diff'] = 100.0 * (dataframe['dwt'] - dataframe['dwt']) / dataframe['dwt']

        dataframe["dwt_trend"] = np.where(
            dataframe["dwt_dir"].rolling(5, center=False).sum() > 3.0, 1.0, -1.0
        )

        # get rolling mean & stddev so that we have a localised estimate of (recent) activity
        dataframe["dwt_mean"] = (
            dataframe["dwt"].rolling(self.win_size, center=False).mean()
        )
        dataframe["dwt_std"] = (
            dataframe["dwt"].rolling(self.win_size, center=False).std()
        )

        # Recent min/max
        dataframe["dwt_recent_min"] = (
            dataframe["dwt"].rolling(window=self.win_size, center=False).min()
        )
        dataframe["dwt_recent_max"] = (
            dataframe["dwt"].rolling(window=self.win_size, center=False).max()
        )
        dataframe["dwt_maxmin"] = (
            100.0
            * (dataframe["dwt_recent_max"] - dataframe["dwt_recent_min"])
            / dataframe["dwt_recent_max"]
        )
        dataframe["dwt_delta_min"] = Series(
            100.0
            * (dataframe["dwt_recent_min"] - dataframe["close"])
            / dataframe["close"]
        ).clip(lower=-5.0)
        dataframe["dwt_delta_max"] = Series(
            100.0
            * (dataframe["dwt_recent_max"] - dataframe["close"])
            / dataframe["close"]
        ).clip(upper=5.0)
        # longer term high/low
        dataframe["dwt_low"] = (
            dataframe["dwt"].rolling(window=self.startup_win, center=False).min()
        )
        dataframe["dwt_high"] = (
            dataframe["dwt"].rolling(window=self.startup_win, center=False).max()
        )

        # # these are (primarily) clues for the ML algorithm:
        # dataframe['dwt_at_min'] = np.where(dataframe['dwt'] <= dataframe['dwt_recent_min'], 1.0, 0.0)
        # dataframe['dwt_at_max'] = np.where(dataframe['dwt'] >= dataframe['dwt_recent_max'], 1.0, 0.0)
        dataframe["dwt_at_low"] = np.where(
            dataframe["dwt"] <= dataframe["dwt_low"], 1.0, 0.0
        )
        dataframe["dwt_at_high"] = np.where(
            dataframe["dwt"] >= dataframe["dwt_high"], 1.0, 0.0
        )

        # dataframe['loss_threshold'] = dataframe['dwt_loss_mean'] - self.n_loss_stddevs * abs(dataframe['dwt_loss_std'])

        return dataframe

    # calculate future gains. Used for setting targets. Yes, we lookahead in the data!
    def add_future_data(self, dataframe: DataFrame, lookahead: int) -> DataFrame:

        lookahead_win = max(lookahead, 14)

        # make a copy of the dataframe so that we do not put any forward looking data into the main dataframe
        # Also, use a different name to avoid cut & paste errors
        future_df = dataframe.copy()

        # we can either use the actual closing price, or the DWT model (smoother)

        use_dwt = True
        if use_dwt:
            price_col = "full_dwt"

            # get the 'full' DWT transform. This models the entire dataframe, so cannot be used in the 'main' dataframe
            future_df["full_dwt"] = self.get_dwt(dataframe["close"])

        else:
            price_col = "close"
            future_df["full_dwt"] = 0.0

        # calculate future gains
        future_df["future_close"] = future_df[price_col].shift(-lookahead_win)

        future_df["future_gain"] = (
            100.0
            * (future_df["future_close"] - future_df[price_col])
            / future_df[price_col]
        ).clip(lower=-5.0, upper=5.0)

        future_df["future_profit"] = future_df["future_gain"].clip(lower=0.0)
        future_df["future_loss"] = future_df["future_gain"].clip(upper=0.0)

        # get rolling mean & stddev so that we have a localised estimate of (recent) future activity
        # Note: window in past because we already looked forward (with 'future_close')
        future_df["future_gain_mean"] = (
            future_df["future_gain"].rolling(lookahead_win).mean()
        )
        future_df["future_gain_std"] = (
            future_df["future_gain"].rolling(lookahead_win).std()
        )
        future_df["future_gain_sum"] = (
            future_df["future_gain"].rolling(lookahead_win).sum()
        )

        future_df["future_profit_mean"] = (
            future_df["future_profit"].rolling(lookahead_win).mean()
        )
        future_df["future_profit_std"] = (
            future_df["future_profit"].rolling(lookahead_win).std()
        )
        future_df["future_loss_mean"] = (
            future_df["future_loss"].rolling(lookahead_win).mean()
        )
        future_df["future_loss_std"] = (
            future_df["future_loss"].rolling(lookahead_win).std()
        )

        future_df["future_profit_max"] = (
            future_df["future_profit"].rolling(lookahead_win).max()
        )
        future_df["future_profit_min"] = (
            future_df["future_profit"].rolling(lookahead_win).min()
        )
        future_df["future_loss_max"] = (
            future_df["future_loss"].rolling(lookahead_win).max()
        )
        future_df["future_loss_min"] = (
            future_df["future_loss"].rolling(lookahead_win).min()
        )

        # future_df['profit_threshold'] = future_df['profit_mean'] + self.n_profit_stddevs * abs(future_df['profit_std'])
        # future_df['loss_threshold'] = future_df['loss_mean'] - self.n_loss_stddevs * abs(future_df['loss_std'])

        future_df["future_profit_threshold"] = future_df[
            "dwt_profit_mean"
        ] + self.n_profit_stddevs * abs(future_df["dwt_profit_std"])
        future_df["future_loss_threshold"] = future_df[
            "dwt_loss_mean"
        ] - self.n_loss_stddevs * abs(future_df["dwt_loss_std"])

        future_df["future_profit_diff"] = (
            future_df["future_profit"] - future_df["future_profit_threshold"]
        ) * 10.0
        future_df["future_loss_diff"] = (
            future_df["future_loss"] - future_df["future_loss_threshold"]
        ) * 10.0

        # future_df['buy_signal'] = np.where(future_df['profit_diff'] > 0.0, 1.0, 0.0)
        # future_df['sell_signal'] = np.where(future_df['loss_diff'] < 0.0, -1.0, 0.0)

        # these explicitly uses dwt
        future_df["future_dwt"] = future_df["full_dwt"].shift(-lookahead_win)
        # future_df['curr_trend'] = np.where(future_df['full_dwt'].shift(-1) > future_df['full_dwt'], 1.0, -1.0)
        # future_df['future_trend'] = np.where(future_df['future_dwt'].shift(-1) > future_df['future_dwt'], 1.0, -1.0)

        future_df["trend"] = np.where(
            future_df[price_col] >= future_df[price_col].shift(), 1.0, -1.0
        )
        future_df["ftrend"] = np.where(
            future_df["future_close"] >= future_df["future_close"].shift(), 1.0, -1.0
        )

        future_df["curr_trend"] = np.where(
            future_df["trend"].rolling(3).sum() > 0.0, 1.0, -1.0
        )
        future_df["future_trend"] = np.where(
            future_df["ftrend"].rolling(3).sum() > 0.0, 1.0, -1.0
        )

        # Sequences of consecutive up/downs (using full_dwt)
        future_df["full_dwt_dir"] = 0.0
        future_df["full_dwt_dir"] = np.where(
            future_df["full_dwt"].diff() > 0, 1.0, -1.0
        )

        future_df["full_dwt_dir_up"] = future_df["full_dwt_dir"].clip(lower=0.0)
        future_df["full_dwt_nseq_up"] = future_df["full_dwt_dir_up"] * (
            future_df["full_dwt_dir_up"]
            .groupby(
                (
                    future_df["full_dwt_dir_up"] != future_df["full_dwt_dir_up"].shift()
                ).cumsum()
            )
            .cumcount()
            + 1
        )
        future_df["full_dwt_nseq_up"] = future_df["full_dwt_nseq_up"].clip(
            lower=0.0, upper=20.0
        )  # removes startup artifacts

        future_df["full_dwt_dir_dn"] = abs(future_df["full_dwt_dir"].clip(upper=0.0))
        future_df["full_dwt_nseq_dn"] = future_df["full_dwt_dir_dn"] * (
            future_df["full_dwt_dir_dn"]
            .groupby(
                (
                    future_df["full_dwt_dir_dn"] != future_df["full_dwt_dir_dn"].shift()
                ).cumsum()
            )
            .cumcount()
            + 1
        )
        future_df["full_dwt_nseq_dn"] = future_df["full_dwt_nseq_dn"].clip(
            lower=0.0, upper=20.0
        )

        # build forward-looking sum of up/down trends
        future_win = pd.api.indexers.FixedForwardWindowIndexer(
            window_size=int(self.win_size)
        )  # don't use a big window

        future_df["future_nseq_up"] = future_df["full_dwt_nseq_up"].shift(
            -self.win_size
        )

        future_df["future_nseq_up_mean"] = (
            future_df["future_nseq_up"].rolling(window=future_win).mean()
        )
        future_df["future_nseq_up_std"] = (
            future_df["future_nseq_up"].rolling(window=future_win).std()
        )
        future_df["future_nseq_up_thresh"] = (
            future_df["future_nseq_up_mean"]
            + self.n_profit_stddevs * future_df["future_nseq_up_std"]
        )

        future_df["future_nseq_dn"] = future_df["full_dwt_nseq_dn"].shift(
            -self.win_size
        )

        future_df["future_nseq_dn_mean"] = (
            future_df["future_nseq_dn"].rolling(future_win).mean()
        )
        future_df["future_nseq_dn_std"] = (
            future_df["future_nseq_dn"].rolling(future_win).std()
        )
        future_df["future_nseq_dn_thresh"] = (
            future_df["future_nseq_dn_mean"]
            - self.n_loss_stddevs * future_df["future_nseq_dn_std"]
        )

        # Recent min/max
        # future_df['future_min'] = future_df[price_col].rolling(window=future_win).min()
        # future_df['future_max'] = future_df[price_col].rolling(window=future_win).max()
        future_df["future_min"] = future_df["dwt"].rolling(window=future_win).min()
        future_df["future_max"] = future_df["dwt"].rolling(window=future_win).max()

        future_df["future_maxmin"] = (
            100.0
            * (future_df["future_max"] - future_df["future_min"])
            / future_df["future_max"]
        )
        future_df["future_delta_min"] = (
            100.0 * (future_df["future_min"] - future_df["close"]) / future_df["close"]
        )
        future_df["future_delta_max"] = (
            100.0 * (future_df["future_max"] - future_df["close"]) / future_df["close"]
        )

        future_df["future_maxmin"] = future_df["future_maxmin"].clip(
            lower=0.0, upper=10.0
        )

        # rolling linear slope of the DWT (i.e. average trend) of near-past (shifted forward)
        future_df["future_slope"] = (
            future_df["future_dwt"].rolling(window=6).apply(self.roll_get_slope)
        )

        # get average gain & stddev
        profit_mean = future_df["future_profit"].mean()
        profit_std = future_df["future_profit"].std()
        loss_mean = future_df["future_loss"].mean()
        loss_std = future_df["future_loss"].std()

        return future_df

    ###################################

    # add indicators used by stoploss/custom sell logic
    def add_stoploss_indicators(self, dataframe) -> DataFrame:

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe["rmi"] = cta.RMI(dataframe, length=24, mom=5)

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        dataframe["mastreak"] = cta.mastreak(dataframe, period=4)

        # Trends
        dataframe["candle_up"] = np.where(
            dataframe["close"] >= dataframe["close"].shift(), 1.0, -1.0
        )
        dataframe["candle_up_trend"] = np.where(
            dataframe["candle_up"].rolling(5).sum() > 0.0, 1.0, -1.0
        )
        dataframe["candle_up_seq"] = dataframe["candle_up"].rolling(5).sum()

        dataframe["candle_dn"] = np.where(
            dataframe["close"] < dataframe["close"].shift(), 1.0, -1.0
        )
        dataframe["candle_dn_trend"] = np.where(
            dataframe["candle_up"].rolling(5).sum() > 0.0, 1.0, -1.0
        )
        dataframe["candle_dn_seq"] = dataframe["candle_up"].rolling(5).sum()

        dataframe["rmi_up"] = np.where(
            dataframe["rmi"] >= dataframe["rmi"].shift(), 1.0, -1.0
        )
        dataframe["rmi_up_trend"] = np.where(
            dataframe["rmi_up"].rolling(5).sum() > 0.0, 1.0, -1.0
        )

        dataframe["rmi_dn"] = np.where(
            dataframe["rmi"] <= dataframe["rmi"].shift(), 1.0, -1.0
        )
        dataframe["rmi_dn_count"] = dataframe["rmi_dn"].rolling(8).sum()

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe["sroc"] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe["ssl_dir"] = 0
        dataframe["ssl_dir"] = np.where(sslup > ssldown, 1.0, -1.0)

        return dataframe

    ##################

    # smooth a series
    def smooth(self, y, window):
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode="same")
        # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        y_smooth = np.round(y_smooth, decimals=3)
        return np.nan_to_num(y_smooth)

    # returns (rolling) smoothed version of input column
    def roll_smooth(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        smooth = self.smooth(col, 4)
        # smooth = gaussian_filter1d(col, 4)
        # smooth = gaussian_filter1d(col, 2)

        length = len(smooth)
        if length > 0:
            return smooth[length - 1]
        else:
            print("model:", smooth)
            return 0.0

    def get_dwt(self, col):

        a = np.array(col)

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        a_notrend = (a - w_mean) / w_std
        # a_notrend = a_notrend.clip(min=-3.0, max=3.0)

        # get DWT model of data
        restored_sig = self.dwtModel(a_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        return model

    def roll_get_dwt(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        model = self.get_dwt(col)

        length = len(model)
        if length > 0:
            return model[length - 1]
        else:
            # cannot calculate DWT (e.g. at startup), just return original value
            return col[len(col) - 1]

    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        wavelet = "db8"
        # wavelet = 'bior1.1'
        # wavelet = 'haar'  # deals well with harsh transitions
        level = 1
        wmode = "smooth"
        tmode = "hard"
        length = len(data)

        # Apply DWT transform
        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        std = np.std(coeff[level])
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        # sigma = madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))

        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=tmode) for i in coeff[1:])

        # inverse DWT transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        # there is a known bug in waverec where odd numbered lengths result in an extra item at the end
        diff = len(model) - len(data)
        return model[0 : len(model) - diff]
        # return model[diff:]

    def madev(self, d, axis=None):
        """Mean absolute deviation of a signal"""
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def roll_get_slope(self, col) -> float:
        # must return scalar, so just calculate prediction and take last value

        slope = np.polyfit(col.index, col, 1)[0]

        if np.isnan(slope) or np.isinf(slope):
            slope = 10.0

        if (slope < 0) and math.isinf(slope):
            slope = -10.0

        return slope

    #######################

    # Utility functions

    # Elliot Wave Oscillator
    def ewo(self, dataframe, sma1_length=5, sma2_length=35):
        sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
        sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
        smadif = (sma1 - sma2) / dataframe["close"] * 100
        return smadif

    # Chaikin Money Flow
    def chaikin_money_flow(self, dataframe, n=20, fillna=False) -> Series:
        """Chaikin Money Flow (CMF)
        It measures the amount of Money Flow Volume over a specific period.
        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
        Args:
            dataframe(pandas.Dataframe): dataframe containing ohlcv
            n(int): n period.
            fillna(bool): if True, fill nan values.
        Returns:
            pandas.Series: New feature generated.
        """
        mfv = (
            (dataframe["close"] - dataframe["low"])
            - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"])
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= dataframe["l1" "volume"]
        cmf = (
            mfv.rolling(n, min_periods=0).sum()
            / dataframe["volume"].rolling(n, min_periods=0).sum()
        )
        if fillna:
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Series(cmf, name="cmf")

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100

    # Volume Weighted Moving Average
    def vwma(self, dataframe: DataFrame, length: int = 10):
        """Indicator: Volume Weighted Moving Average (VWMA)"""
        # Calculate Result
        pv = dataframe["close"] * dataframe["volume"]
        vwma = Series(
            ta.SMA(pv, timeperiod=length)
            / ta.SMA(dataframe["volume"], timeperiod=length)
        ).fillna(0)
        return vwma

    # Exponential moving average of a volume weighted simple moving average
    def ema_vwma_osc(self, dataframe, len_slow_ma):
        slow_ema = Series(ta.EMA(self.vwma(dataframe, len_slow_ma), len_slow_ma))
        return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

    def t3_average(self, dataframe, length=5):
        """
        T3 Average by HPotter on Tradingview
        https://www.tradingview.com/script/qzoC9H1I-T3-Average/
        """
        df = dataframe.copy()

        df["xe1"] = ta.EMA(df["close"], timeperiod=length).fillna(0)
        df["xe2"] = ta.EMA(df["xe1"], timeperiod=length).fillna(0)
        df["xe3"] = ta.EMA(df["xe2"], timeperiod=length).fillna(0)
        df["xe4"] = ta.EMA(df["xe3"], timeperiod=length).fillna(0)
        df["xe5"] = ta.EMA(df["xe4"], timeperiod=length).fillna(0)
        df["xe6"] = ta.EMA(df["xe5"], timeperiod=length).fillna(0)
        b = 0.7
        c1 = -b * b * b
        c2 = 3 * b * b + 3 * b * b * b
        c3 = -6 * b * b - 3 * b - 3 * b * b * b
        c4 = 1 + 3 * b + b * b * b + 3 * b * b
        df["T3Average"] = (
            c1 * df["xe6"] + c2 * df["xe5"] + c3 * df["xe4"] + c4 * df["xe3"]
        )

        return df["T3Average"]

    def pivot_points(self, dataframe: DataFrame, mode="fibonacci") -> Series:
        if mode == "simple":
            hlc3_pivot = (
                dataframe["high"] + dataframe["low"] + dataframe["close"]
            ).shift(1) / 3
            res1 = hlc3_pivot * 2 - dataframe["low"].shift(1)
            sup1 = hlc3_pivot * 2 - dataframe["high"].shift(1)
            res2 = hlc3_pivot + (dataframe["high"] - dataframe["low"]).shift()
            sup2 = hlc3_pivot - (dataframe["high"] - dataframe["low"]).shift()
            res3 = hlc3_pivot * 2 + (dataframe["high"] - 2 * dataframe["low"]).shift()
            sup3 = hlc3_pivot * 2 - (2 * dataframe["high"] - dataframe["low"]).shift()
            return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
        elif mode == "fibonacci":
            hlc3_pivot = (
                dataframe["high"] + dataframe["low"] + dataframe["close"]
            ).shift(1) / 3
            hl_range = (dataframe["high"] - dataframe["low"]).shift(1)
            res1 = hlc3_pivot + 0.382 * hl_range
            sup1 = hlc3_pivot - 0.382 * hl_range
            res2 = hlc3_pivot + 0.618 * hl_range
            sup2 = hlc3_pivot - 0.618 * hl_range
            res3 = hlc3_pivot + 1 * hl_range
            sup3 = hlc3_pivot - 1 * hl_range
            return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
        elif mode == "DeMark":
            demark_pivot_lt = (
                dataframe["low"] * 2 + dataframe["high"] + dataframe["close"]
            )
            demark_pivot_eq = (
                dataframe["close"] * 2 + dataframe["low"] + dataframe["high"]
            )
            demark_pivot_gt = (
                dataframe["high"] * 2 + dataframe["low"] + dataframe["close"]
            )
            demark_pivot = np.where(
                (dataframe["close"] < dataframe["open"]),
                demark_pivot_lt,
                np.where(
                    (dataframe["close"] > dataframe["open"]),
                    demark_pivot_gt,
                    demark_pivot_eq,
                ),
            )
            dm_pivot = demark_pivot / 4
            dm_res = demark_pivot / 2 - dataframe["low"]
            dm_sup = demark_pivot / 2 - dataframe["high"]
            return dm_pivot, dm_res, dm_sup

    def heikin_ashi(
        self, dataframe, smooth_inputs=False, smooth_outputs=False, length=10
    ):
        df = dataframe[["open", "close", "high", "low"]].copy().fillna(0)
        if smooth_inputs:
            df["open_s"] = ta.EMA(df["open"], timeframe=length)
            df["high_s"] = ta.EMA(df["high"], timeframe=length)
            df["low_s"] = ta.EMA(df["low"], timeframe=length)
            df["close_s"] = ta.EMA(df["close"], timeframe=length)

            open_ha = (df["open_s"].shift(1) + df["close_s"].shift(1)) / 2
            high_ha = df.loc[:, ["high_s", "open_s", "close_s"]].max(axis=1)
            low_ha = df.loc[:, ["low_s", "open_s", "close_s"]].min(axis=1)
            close_ha = (df["open_s"] + df["high_s"] + df["low_s"] + df["close_s"]) / 4
        else:
            open_ha = (df["open"].shift(1) + df["close"].shift(1)) / 2
            high_ha = df.loc[:, ["high", "open", "close"]].max(axis=1)
            low_ha = df.loc[:, ["low", "open", "close"]].min(axis=1)
            close_ha = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

        open_ha = open_ha.fillna(0)
        high_ha = high_ha.fillna(0)
        low_ha = low_ha.fillna(0)
        close_ha = close_ha.fillna(0)

        if smooth_outputs:
            open_sha = ta.EMA(open_ha, timeframe=length)
            high_sha = ta.EMA(high_ha, timeframe=length)
            low_sha = ta.EMA(low_ha, timeframe=length)
            close_sha = ta.EMA(close_ha, timeframe=length)

            return open_sha, close_sha, low_sha
        else:
            return open_ha, close_ha, low_ha

    # Range midpoint acts as Support
    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] > row_data[row + 1])
            else:
                conditions.append(row_data[row] < row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

    # Range midpoint acts as Resistance
    def is_resistance(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] < row_data[row + 1])
            else:
                conditions.append(row_data[row] > row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == "HL":
            return (
                dataframe["high"].rolling(length).max()
                - dataframe["low"].rolling(length).min()
            ) / dataframe["low"].rolling(length).min()
        elif method == "OC":
            return (
                dataframe["open"].rolling(length).max()
                - dataframe["close"].rolling(length).min()
            ) / dataframe["close"].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

    # Unbound (Dynamic Range) Indicators from DataframePopulator
    unbound_indicators = [
        # Price-based indicators (unbounded)
        "atr",  # Average True Range - 0 to ∞
        "bb_upper",  # Bollinger Bands Upper - price-based
        "bb_lower",  # Bollinger Bands Lower - price-based
        "bb_middle",  # Bollinger Bands Middle - price-based
        "bb_width",  # Bollinger Bands Width - 0 to ∞
        "bb_position",  # Bollinger Bands Position - 0 to 1 (but can be >1 or <0)
        "bb_gain",  # Bollinger Bands Gain - -∞ to +∞
        "bb_loss",  # Bollinger Bands Loss - -∞ to +∞
        "bb_upperband",  # Bollinger Bands Upper (legacy) - price-based
        "bb_lowerband",  # Bollinger Bands Lower (legacy) - price-based
        "bb_middleband",  # Bollinger Bands Middle (legacy) - price-based
        "close_smoothed",
        "close_norm",
        # Moving averages (price-based, unbounded)
        "sma",  # Simple Moving Average - price-based
        "ema",  # Exponential Moving Average - price-based
        "tema",  # Triple Exponential Moving Average - price-based
        "sma_200",  # 200-period SMA - price-based
        "sma_14",  # 14-period SMA - price-based
        # Price levels and ranges
        "dc_upper",  # Donchian Channel Upper - price-based
        "dc_lower",  # Donchian Channel Lower - price-based
        "dc_mid",  # Donchian Channel Middle - price-based
        "dc_dist",  # Donchian Channel Distance - 0 to ∞
        "dcbb_dist_upper",  # DC-BB Distance Upper - -∞ to +∞
        "dcbb_dist_lower",  # DC-BB Distance Lower - -∞ to +∞
        # Volume-based indicators (unbounded)
        "obv",  # On-Balance Volume - -∞ to +∞
        "ad",  # Accumulation/Distribution - -∞ to +∞
        "volume_sma",  # Volume SMA - 0 to ∞
        "volume_ratio",  # Volume Ratio - 0 to ∞
        "volume_momentum",  # Volume Momentum - -∞ to +∞
        "volume_1h",  # 1h Volume Average - 0 to ∞
        "volume_4h",  # 4h Volume Average - 0 to ∞
        "volume_1d",  # 1d Volume Average - 0 to ∞
        # VWAP and price-volume
        "vwap",  # Volume Weighted Average Price - price-based
        "typical_price",  # Typical Price - price-based
        # MACD components (unbounded)
        "macd",  # MACD Line - -∞ to +∞
        "macd_signal",  # MACD Signal - -∞ to +∞
        "macd_hist",  # MACD Histogram - -∞ to +∞
        "macdsignal",  # MACD Signal (legacy) - -∞ to +∞
        "macdhist",  # MACD Histogram (legacy) - -∞ to +∞
        # Price action features (unbounded)
        "body_size",  # Candle Body Size - 0 to ∞
        "body_size_pct",  # Body Size Percentage - 0 to ∞
        "upper_shadow",  # Upper Shadow - 0 to ∞
        "lower_shadow",  # Lower Shadow - 0 to ∞
        "upper_shadow_pct",  # Upper Shadow Percentage - 0 to ∞
        "lower_shadow_pct",  # Lower Shadow Percentage - 0 to ∞
        "price_position",  # Price Position in Range - 0 to 1 (but can be >1 or <0)
        # Spread and volatility
        "spread",  # High-Low Spread - 0 to ∞
        "spread_pct",  # Spread Percentage - 0 to ∞
        "spread_ma",  # Spread Moving Average - 0 to ∞
        "spread_relative",  # Relative Spread - -∞ to +∞
        # Momentum and returns (unbounded)
        "price_momentum_1",  # 1-period Price Momentum - -∞ to +∞
        "price_momentum_5",  # 5-period Price Momentum - -∞ to +∞
        "price_momentum_20",  # 20-period Price Momentum - -∞ to +∞
        "momentum_1h",  # 1h Momentum - -∞ to +∞
        "momentum_4h",  # 4h Momentum - -∞ to +∞
        "momentum_1d",  # 1d Momentum - -∞ to +∞
        "returns_squared",  # Squared Returns - 0 to ∞
        "volatility_cluster",  # Volatility Clustering - 0 to ∞
        # Cross-timeframe averages (price-based)
        "close_1h",  # 1h Close Average - price-based
        "close_4h",  # 4h Close Average - price-based
        "close_1d",  # 1d Close Average - price-based
        "high_1h",  # 1h High Average - price-based
        "high_4h",  # 4h High Average - price-based
        "low_1h",  # 1h Low Average - price-based
        "low_4h",  # 4h Low Average - price-based
        # Liquidity and market microstructure
        "amihud_illiq",  # Amihud Illiquidity - 0 to ∞
        "price_impact",  # Price Impact - 0 to ∞
        "volume_value",  # Volume * Price - 0 to ∞
        "volume_log",  # Log Volume - -∞ to +∞
        # DWT (Discrete Wavelet Transform) features
        "dwt",  # DWT - -∞ to +∞
        "dwt_gain",  # DWT Gain - -∞ to +∞
        "dwt_profit_mean",  # DWT Profit Mean - -∞ to +∞
        "dwt_loss_mean",  # DWT Loss Mean - -∞ to +∞
        "dwt_profit_std",  # DWT Profit Std - 0 to ∞
        "dwt_loss_std",  # DWT Loss Std - 0 to ∞
        "dwt_diff",  # DWT Difference - -∞ to +∞
        "dwt_maxmin",  # DWT Max-Min - 0 to ∞
        "dwt_delta_min",  # DWT Delta Min - -∞ to +∞
        "dwt_delta_max",  # DWT Delta Max - -∞ to +∞
        "dwt_recent_max",  # DWT Recent Max - price-based
        "dwt_recent_min",  # DWT Recent Min - price-based
        # Thresholds (unbounded)
        "profit_threshold",  # Profit Threshold - -∞ to +∞
        "loss_threshold",  # Loss Threshold - -∞ to +∞
        # Mid price
        "mid",  # Mid Price - price-based
        # Stochastic and oscillator components
        "fastk",  # Fast %K - 0 to 100 (but can be >100 or <0)
        "fastd",  # Fast %D - 0 to 100 (but can be >100 or <0)
        "fast_diff",  # Fast K-D Difference - -∞ to +∞
        "stochrsi",  # Stochastic RSI - 0 to 1 (but can be >1 or <0)
        "srsi_k",  # Stochastic RSI %K - 0 to 100 (but can be >100 or <0)
        "srsi_d",  # Stochastic RSI %D - 0 to 100 (but can be >100 or <0)
        # Fisher Transform
        "fisher_wr",  # Fisher Williams %R - -∞ to +∞
        # VFI (Volume Flow Indicator)
        "vfi",  # Volume Flow Indicator - -∞ to +∞
        # SAR (Parabolic SAR)
        "sar",  # Parabolic SAR - price-based
        # RMI (Relative Momentum Index)
        "rmi",  # Relative Momentum Index - 0 to 100 (but can be >100 or <0)
        # SROC (Stochastic Rate of Change)
        "sroc",  # Stochastic Rate of Change - -∞ to +∞
        # Mastreak
        "mastreak",  # Mastreak - 0 to ∞
        # SSL Channels
        "sslup",  # SSL Up - price-based
        "ssldown",  # SSL Down - price-based
        # Directional indicators (unbounded)
        "dm_delta",  # DM Plus - DM Minus - -∞ to +∞
        "di_delta",  # DI Plus - DI Minus - -∞ to +∞
        "di_diff",  # Plus DI - Minus DI - -∞ to +∞
        # Candle direction indicators (binary but unbounded in calculations)
        "candle_up",  # Candle Up - -1 or 1
        "candle_dn",  # Candle Down - -1 or 1
        "candle_up_trend",  # Candle Up Trend - -1 or 1
        "candle_dn_trend",  # Candle Down Trend - -1 or 1
        "rmi_up",  # RMI Up - -1 or 1
        "rmi_up_trend",  # RMI Up Trend - -1 or 1
        "rmi_dn",  # RMI Down - -1 or 1
        "ssl_dir",  # SSL Direction - -1 or 1
        "dwt_dir",  # DWT Direction - -1 or 1
        "dwt_trend",  # DWT Trend - -1 or 1
        # Sequence counters (unbounded)
        "dwt_nseq_up",  # DWT Up Sequence - 0 to ∞
        "dwt_nseq_dn",  # DWT Down Sequence - 0 to ∞
        # Future DWT (for training labels)
        "fwd_dwt",  # Forward DWT - -∞ to +∞
        # Raw price data (always unbound)
        "open",  # Open Price - price-based
        "high",  # High Price - price-based
        "low",  # Low Price - price-based
        "close",  # Close Price - price-based
        "volume",  # Volume - 0 to ∞
    ]

    # Note: This list includes indicators that are technically bounded (like RSI 0-100)
    # but can exceed their theoretical bounds due to calculation artifacts or extreme market conditions.
    # For normalization purposes, these are treated as unbound since their actual ranges can vary significantly.

    # return the list of unbound indicators that are in the dataframe
    def get_unbound_indicators(self, df: DataFrame) -> list:
        return [col for col in self.unbound_indicators if col in df.columns]
