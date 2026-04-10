# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413,  W1203, W291

"""
####################################################################################
TS_Predict - base class for 'simple' time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             override the model-related functions

             Note that I use gain rather than price because it is a normalised value, and works better with prediction
             algorithms. I use the actual (future) gain to train a base model, which is then further refined for each
             individual pair.
             The model is created if it does not exist, and is trained on all available data before being saved.
             Models are saved in user_data/strategies/saved_data/<class>/<class>.sav, where <class> is the name
             of the current class (TS_Predict if running this directly, or the name of the subclass).
             If the model already exits, then it is just loaded and used.
             So, it makes sense to do initial training over a long period of time to create the base model.
             If training, then no backtesting or tuning for individual pairs is performed (way faster).
             If you want to retrain (e.g. you changed indicators), then delete the model and run the strategy over a
             long time period

####################################################################################
"""


import copy
import cProfile
import os
import pstats

import sys
import traceback
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Optional

import logging
import warnings

import joblib
import numpy as np


import pandas as pd
import pywt
from pandas import DataFrame, Series

import talib.abstract as ta
import finta

import technical.indicators as ftt

# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from freqtrade import leverage

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IStrategy

from Framework.BaseStrategy import (
    BaseStrategy,
    StrategyConfig,
    NormalizationType,
    ModelType,
)


# from lightgbm import LGBMRegressor
# from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from xgboost import XGBRegressor
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)


import utils.custom_indicators as cta
from utils.Scalers import load_scaler, scaler_exists

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from utils.DataframeUtils import DataframeUtils, ScalerType  # pylint: disable=E0401

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

pd.options.mode.chained_assignment = None  # default='warn'


# -----------------------------------------------------------------------
# MLX Model for vectorized prediction
class VectorizedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def train_mlx_global(X, y, epochs=100, seed=42):
    mx.random.seed(seed)
    X_mx = mx.array(X)
    y_mx = mx.array(y)

    # ensure y is 2D if single output
    if y_mx.ndim == 1:
        y_mx = y_mx.reshape(-1, 1)

    model = VectorizedMLP(X_mx.shape[1], y_mx.shape[1])
    mx.eval(model.parameters())

    def loss_fn(model, X, y):
        return mx.mean(mx.square(model(X) - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-3)

    for _ in range(epochs):
        loss, grads = loss_and_grad(model, X_mx, y_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return model


class TSPredict(BaseStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # Strategy configuration
    strategy_config = StrategyConfig(
        normalization=NormalizationType.ROLLING_ROBUST,
        model_type=ModelType.CUSTOM,
        needs_training=True,
    )

    # indicators to include in normalisation. Anything not in the list will be dropped.
    include_list = [
        "gain_norm",
        "rvol",
        "close_norm",
        "bb_width",
        "atr_norm",
        "rsi_scaled",
        "fisher_wr",
        "mfi_scaled",
        "guard_metric",
        "macd_norm",
        "macdhist_norm",
        "macdsignal_norm",
        "ema_fast_norm",
    ]

    # columns that are already normalized and should not be scaled
    pre_normalized_columns = [
        "ad_scaled",
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        "bb_width",
        "close_norm",
        "di_diff_scaled",
        "ema_fast_norm",
        "fast_diff",
        "fastk_scaled",
        "fisher_ss",
        "cg_ss",
        "gain_norm",
        "log_volume_norm",
        "macd_norm",
        "macdhist_norm",
        "macdsignal_norm",
        "rsi_scaled",
        "sar_ratio",
        "vwap_ratio",
        "rvol",
        "guard_metric",
        "fisher_wr",
        "mfi_scaled",
    ]

    aggregate_pairs = True  # use all pairs for training (in backtest)
    norm_data = True  # Now enabled by default: normalization applied before decomposition/analysis
    scale_results = True

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "predicted_gain": {"color": "rebeccapurple"},
                "shifted_pred": {"color": "skyblue"},
                "gain": {"color": "green"},
                "target_profit": {"color": "lightgreen"},
                "target_loss": {"color": "lightsalmon"},
            },
        },
    }

    # Required
    startup_candle_count: int = 128  # must be power of 2

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    lookahead = 6

    df_coeffs: DataFrame = None
    coeff_table = None
    coeff_array = None
    gain_data = None
    merge_indicators = (
        False  # set to False to not merge indicators into prediction data
    )

    use_rolling = False  # True = rolling (slow but realistic), False = Jumping (much faster, less realistic)
    single_col_prediction = (
        False  # True = use only gain. False = use all columns (better, but much slower)
    )

    wavelet_type: Wavelets.WaveletType = Wavelets.WaveletType.DWT
    wavelet = None

    forecaster_type: Forecasters.ForecasterType = Forecasters.ForecasterType.PA
    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.SGD
    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.SVR
    forecaster = None

    use_mlx = False
    try:
        import mlx.core as mx

        use_mlx = hasattr(mx, "metal") and mx.metal.is_available()
    except ImportError:
        pass

    data = None

    wavelet_size = 64  # needed for consistently-sized transforms
    win_size = wavelet_size  # this can vary

    train_min_len = wavelet_size  # longer = slower
    train_len = min(128, wavelet_size * 4)  # longer = slower
    # scale_len = wavelet_size // 2 # no. recent candles to use when scaling
    scale_len = min(8, wavelet_size // 2)  # no. recent candles to use when scaling
    win_size = min(32, wavelet_size)
    model_window = wavelet_size  # longer = slower

    profit_nstd = 2.6
    loss_nstd = 2.6

    training_data = None
    training_labels = None
    training_mode = False  # do not set manually
    supports_incremental_training = True
    model_per_pair = False
    combine_models = True
    model_trained = False
    new_model = False
    detrend_data = False
    scale_results = False

    # norm_data = False  # REMOVED: using global setting above

    dataframeUtils = None
    scaler = RobustScaler()
    model = None
    base_forecaster = None

    curr_dataframe: DataFrame = None

    target_profit = 0.0
    target_loss = 0.0

    # hyperparams

    # Buy hyperspace params:
    buy_params = {
        **BaseStrategy.buy_params,
        "entry_enable_guards": True,
        "entry_guard_threshold": -0.0,
        "cexit_min_profit_th": 0.5,
        "cexit_profit_nstd": 1.9,
    }

    # Sell hyperspace params:
    sell_params = {
        **BaseStrategy.sell_params,
        "exit_guard_threshold": 0.0,
        "cexit_loss_nstd": 1.8,
        "cexit_min_loss_th": -0.5,
    }

    # Custom Exit

    # No. Standard Deviations of profit/loss for target, and lower limit
    cexit_min_profit_th = DecimalParameter(
        0.0, 1.5, default=0.7, decimals=1, space="buy", load=True, optimize=True
    )
    cexit_profit_nstd = DecimalParameter(
        0.0, 3.0, default=0.9, decimals=1, space="buy", load=True, optimize=True
    )

    cexit_min_loss_th = DecimalParameter(
        -1.5, -0.0, default=-0.4, decimals=1, space="sell", load=True, optimize=True
    )
    cexit_loss_nstd = DecimalParameter(
        0.0, 3.0, default=0.7, decimals=1, space="sell", load=True, optimize=True
    )

    ###################################

    def bot_start(self, **kwargs) -> None:
        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)

        if self.wavelet is None:
            self.wavelet = Wavelets.make_wavelet(self.wavelet_type)

        if self.forecaster is None:
            self.forecaster = Forecasters.make_forecaster(self.forecaster_type)
            self.forecaster.set_detrend(self.detrend_data)

        if (not self.forecaster.supports_multiple_columns()) and (
            not self.single_col_prediction
        ):
            print("    ****")
            print(
                f"    **** ERROR: forecaster ({self.forecaster_type.name}) does not support multiple indicators"
            )
            print("    ****")

        if not self.forecaster.supports_retrain():
            print("    ****")
            print(
                f"    **** WARNING: forecaster ({self.forecaster_type.name}) does not support retrainings"
            )
            print("    ****")

        # reset global vars based on wavelet_size, which can be changed by subclasses
        self.win_size = self.wavelet_size  # this can vary
        self.train_min_len = self.wavelet_size  # longer = slower
        self.train_len = min(128, self.wavelet_size * 4)  # longer = slower
        # scale_len = wavelet_size // 2 # no. recent candles to use when scaling
        self.scale_len = min(
            8, self.wavelet_size // 2
        )  # no. recent candles to use when scaling
        self.win_size = min(32, self.wavelet_size)
        self.model_window = self.wavelet_size  # longer = slower

        print("")
        print(f"    wavelet_type:    {self.wavelet_type.name} ({self.wavelet_size})")
        print(f"    win_size:        {self.win_size}")
        print(f"    forecaster_type: {self.forecaster.get_name()}")
        print(f"    detrend_data:    {self.forecaster.detrend_data}")
        print("")
        return

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    ###################################

    # update saved data based on current pairlist
    def update_pairlist_data(self):
        # this only makes sense in 'live' modes
        if self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            return

        # current pairlist
        curr_pairlist = np.array(self.dp.current_whitelist())

        # pairlist from previous calls
        saved_pairlist = np.array(list(self.custom_trade_info.keys()))

        # get the pairs that are no longer in the list
        removed_pairs = np.setdiff1d(saved_pairlist, curr_pairlist)
        added_pairs = np.setdiff1d(curr_pairlist, saved_pairlist)

        if len(removed_pairs) > 0:
            print("    Pairlist changed:")
            print(f"    old pairs: {saved_pairlist}")
            print(f"    new pairs: {curr_pairlist}")
            print(f"    pairs removed: {removed_pairs}")
            print(f"    pairs added: {added_pairs}")

            for pair in removed_pairs:
                print(f"    Removing historical data for: {pair}")
                del self.custom_trade_info[pair]

    ###################################
    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # NOTE: if you change the indicators, you need to regenerate the model

        dataframe = super().populate_indicators(dataframe, metadata)

        window_size = min(32, self.win_size)

        # Base pair dataframe timeframe indicators
        curr_pair = metadata["pair"]

        self.curr_dataframe = dataframe
        self.curr_pair = curr_pair

        self.update_pairlist_data()

        # The following are needed for base functions, so do not remove.
        # Add custom indicators to add_strategy_indicators()

        # backward looking gain
        gain = (
            100.0
            * (dataframe["close"] - dataframe["close"].shift(self.lookahead))
            / dataframe["close"].shift(self.lookahead)
        ).fillna(0.0)
        dataframe["gain"] = self.super_smoother(gain, 10).fillna(0.0)
        dataframe["gain"] = dataframe["gain"].round(4)

        # need to save the gain data for later scaling
        self.gain_data = dataframe["gain"].to_numpy().copy()

        # target profit/loss thresholds
        dataframe["profit"] = dataframe["gain"].clip(lower=0.0)
        dataframe["loss"] = dataframe["gain"].clip(upper=0.0)

        dataframe = self.update_gain_targets(dataframe)

        # Add strategy-specific indicators
        dataframe = self.add_strategy_indicators(dataframe)

        # create and init the model, if first time (dataframe has to be populated first)
        if self.model is None:
            # print("    Loading model")
            self.load_model(np.shape(dataframe))

        # init prediction column
        dataframe["predicted_gain"] = 0.0

        # add the predictions
        # print("    Making predictions...")
        dataframe = self.add_predictions(dataframe)

        dataframe["target_profit"] = 0.0
        dataframe["target_loss"] = 0.0
        dataframe["buy_region"] = 0
        dataframe["sell_region"] = 0

        return dataframe

    def update_gain_targets(self, dataframe):
        # use a fixed window for thresholds to improve stability
        win_size = 40
        self.profit_nstd = float(self.cexit_profit_nstd.value)
        self.loss_nstd = float(self.cexit_loss_nstd.value)

        dataframe["target_profit"] = (
            dataframe["profit"].rolling(window=win_size).mean()
            + self.profit_nstd * dataframe["profit"].rolling(window=win_size).std()
        )

        dataframe["target_loss"] = dataframe["loss"].rolling(
            window=win_size
        ).mean() - self.loss_nstd * abs(
            dataframe["loss"].rolling(window=win_size).std()
        )

        dataframe["target_profit"] = dataframe["target_profit"].clip(
            lower=float(self.cexit_min_profit_th.value)
        )
        dataframe["target_loss"] = dataframe["target_loss"].clip(
            upper=float(self.cexit_min_loss_th.value)
        )

        dataframe["target_profit"] = np.nan_to_num(dataframe["target_profit"])
        dataframe["target_loss"] = np.nan_to_num(dataframe["target_loss"])

        dataframe["local_mean"] = dataframe["close"].rolling(window=win_size).mean()
        dataframe["local_min"] = dataframe["close"].rolling(window=win_size).min()
        dataframe["local_max"] = dataframe["close"].rolling(window=win_size).max()

        return dataframe

    ###################################

    def add_strategy_indicators(self, dataframe):
        # Override this function in subclasses and add extra indicators here

        return dataframe

    ###################################

    def super_smoother(self, series, period):
        """Ehlers Super Smoother filter."""
        n = len(series)
        filt = np.zeros(n)
        s = series.values if hasattr(series, "values") else np.asarray(series)

        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3

        for i in range(2, n):
            filt[i] = c1 * (s[i] + s[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

        if hasattr(series, "index"):
            return pd.Series(filt, index=series.index)
        else:
            return filt

    def smooth(self, y, window):
        return self.super_smoother(y, window)

    # -----------------------

    # look ahead to get future gain. Do *not* put this into the main dataframe!
    def get_future_gain(self, dataframe):
        df = self.convert_dataframe(dataframe)
        future_gain = df["gain"].shift(-self.lookahead).fillna(0.0)
        future_gain = self.super_smoother(future_gain, 10).fillna(0.0).to_numpy().copy()
        future_gain[-self.lookahead :] = 0.0
        future_gain = np.round(future_gain, decimals=3)
        future_gain = np.nan_to_num(future_gain)

        return future_gain

    # -------------
    # Normalisation

    array_scaler = RobustScaler()

    def update_scaler(self, data):
        if not self.array_scaler:
            self.array_scaler = RobustScaler()

        self.array_scaler.fit(data.reshape(-1, 1))

    def norm_array(self, a):
        return self.array_scaler.transform(a.reshape(-1, 1))

    def denorm_array(self, a):
        return self.array_scaler.inverse_transform(a.reshape(-1, 1)).squeeze()

    # scales array data, based on array target
    def scale_array(self, target, data):
        # detrend the input arrays
        t = np.arange(0, len(target))
        t_poly = np.polyfit(t, target, 1)
        t_line = np.polyval(t_poly, target)
        x = target - t_line

        t = np.arange(0, len(data))
        d_poly = np.polyfit(t, data, 1)
        d_line = np.polyval(d_poly, data)
        y = data - d_line

        # scale untrended data
        self.update_scaler(x)
        y_scaled = self.denorm_array(y)

        # retrend
        y_scaled = y_scaled + d_line

        return y_scaled

    # -------------

    ###################################

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
        """
        Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the
        high and low of the past N days (for a given N). It was developed by a publisher and promoter of trading
        materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in
        between,  of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100

    ###################################

    # -------------

    def convert_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        # filter down to selected columns
        cols = [col for col in self.include_list if col in df.columns]

        # also include 'gain' if present, as it is used as a target
        if ("gain" in df.columns) and ("gain" not in cols):
            cols.append("gain")

        df = df[cols]

        if self.norm_data:
            # only scale columns that are not already normalized
            cols_to_scale = [
                col for col in cols if col not in self.pre_normalized_columns
            ]
            if len(cols_to_scale) > 0:
                df = self.dataframeUtils.normalize_selected_columns(
                    df, cols_to_scale, window=128
                )

        df = df.fillna(0.0)
        return df

    ###################################

    def get_model_path(self, pair):
        category = self.__class__.__name__
        root_dir = self.get_storage_location() + category
        model_name = category
        if self.model_per_pair and (len(pair) > 0):
            model_name = model_name + "_" + pair.split("/")[0]
        path = root_dir + "/" + model_name + ".sav"
        return path

    def load_model(self, df_shape):
        model_path = self.get_model_path("")

        # load from file or create new model
        if os.path.exists(model_path):
            # use joblib to reload model state
            print("    loading from: ", model_path)
            self.model = joblib.load(model_path)
            self.model_trained = True
            self.new_model = False
            self.training_mode = False
            # set the model in the forecaster
            self.forecaster.set_model(self.model)
        else:
            self.model = self.forecaster.get_model()
            self.model_trained = False
            self.new_model = True
            self.training_mode = True

        # sklearn family of regressors sometimes support starting with an existing model (warm_start),
        # or incremental training (partial_fit())
        if hasattr(self.model, "warm_start"):
            self.model.warm_start = True
            self.supports_incremental_training = True  # override default

        if hasattr(self.model, "partial_fit"):
            self.supports_incremental_training = True  # override default

        # if self.model is None:
        #     print("***    ERR: model was not created properly ***")

        return

    # -------------

    def save_model(self):
        # save trained model

        model_path = self.get_model_path("")

        # create directory if it doesn't already exist
        save_dir = os.path.dirname(model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # extract underlying model from forecaster
        model = self.forecaster.get_model()

        # use joblib to save model state
        print("    saving to: ", model_path)
        joblib.dump(model, model_path)

        return

    # -------------

    # train the model. Override if not an sklearn-compatible algorithm
    # set save_model=False if you don't want to save the model (needed for ML algorithms)
    def train_model(
        self,
        forecaster: Forecasters.base_forecaster,
        data: np.array,
        results: np.array,
        save_model,
    ):
        if forecaster is None:
            print("***    ERR: no forecaster ***")
            return

        x = np.nan_to_num(data)
        y = np.nan_to_num(results)

        forecaster.train(x, y, incremental=True)

        # print(f'   train_model() data:{np.shape(data)} results:{np.shape(results)}')

        return

    # -------------

    # initial training of the model
    def init_model(self, dataframe: DataFrame):
        # if model is not yet trained, or this is a new model and we want to combine across pairs, then train
        if (not self.model_trained) or (self.new_model and self.combine_models):
            df = dataframe

            future_gain_data = self.get_future_gain(df)
            data, _ = self.get_data(df)

            if self.single_col_prediction:
                training_data = dataframe["gain"].to_numpy()
                # training_data = self.smooth(training_data, 2)
                training_data = training_data.reshape(-1, 1)
            else:
                training_data = data.copy()
            training_data = training_data[: -self.lookahead - 1]
            training_labels = future_gain_data[: -self.lookahead - 1].copy()

            if not self.model_trained:
                print(f"    initial training ({self.curr_pair})")
            else:
                print(f"    incremental training ({self.curr_pair})")

            if self.forecaster.supports_retrain:
                # loop through data and train on self.wavelet_length amounts of data
                start = 0
                end = self.train_len - 1
                num_buffs = int((np.shape(training_data)[0]) / self.train_len)
                for i in range(num_buffs):
                    # print(f'   start:{start} end:{end} self.train_len:{self.train_len}')
                    self.train_model(
                        self.forecaster,
                        training_data[start:end],
                        training_labels[start:end],
                        True,
                    )

                    start = start + self.train_len
                    end = end + self.train_len

            else:
                self.train_model(self.forecaster, training_data, training_labels, True)

            self.model_trained = True

            if self.new_model:
                self.save_model()

        # print(f'    model_trained:{self.model_trained} new_model:{self.new_model}  combine_models:{self.combine_models}')

        return

    # -------------

    # set the data for this straegy. Override if necessary
    def get_data(self, dataframe):
        # build feature set from converted dataframe
        df = self.convert_dataframe(dataframe)

        # features are everything EXCEPT 'gain' (which is the target)
        features = [col for col in df.columns if col != "gain"]
        self.data = df[features].to_numpy()

        return self.data, features

    # -------------

    # generate predictions for an np array (intended to be overriden if needed)
    def predict_data(self, forecaster: Forecasters.base_forecaster, data):
        x = np.nan_to_num(data)

        preds = forecaster.forecast(x, self.lookahead)

        # print(f'    data:{np.shape(data)} preds:{np.shape(preds)}')

        # # smooth predictions to try and avoid drastic changes
        # preds = self.smooth(preds, 2)

        # scale the results to generally match the input characteristics
        if self.scale_results:
            preds = self.scale_array(data[-8:], preds)

        preds = np.clip(preds, -3.0, 3.0)
        return preds

    # -------------

    # single prediction (for use in rolling calculation)
    def predict(self, gain, dataframe) -> float:
        # Get the start and end index labels of the series
        start = gain.index[0]
        end = gain.index[-1]

        # Get the integer positions of the labels in the dataframe index
        start_row = dataframe.index.get_loc(start)
        end_row = dataframe.index.get_loc(end) + 1  # need to add the 1, don't know why!

        # if end_row < (self.wavelet_size + self.lookahead):
        if start_row < (self.wavelet_size + self.lookahead):  # need buffer for training
            return 0.0

        # train on previous data
        train_end = start_row - self.lookahead - 1
        train_start = max(0, train_end - self.train_len)
        scale_start = max(0, end - self.scale_len)

        if (not self.training_mode) and (self.supports_incremental_training):
            train_data = self.training_data[train_start : start - 1].copy()
            train_results = self.training_labels[train_start : start - 1].copy()
            # pair_forecaster = copy.deepcopy(self.forecaster)  # reset to avoid over-training
            pair_forecaster = self.forecaster
            self.train_model(pair_forecaster, train_data, train_results, False)
        else:
            pair_forecaster = self.forecaster

        # predict for current window
        dslice = self.training_data[start:end].copy()
        self.gain_data = np.array(
            dataframe["gain"].iloc[scale_start:end]
        )  # needed for scaling
        y_pred = self.predict_data(pair_forecaster, dslice)

        return y_pred[-1]

    # -------------

    # alternate rolling prediction approach. The pandas rolling mechanism seems to have issues for some reason
    def rolling_predict(self, gain, window_size):
        win_size = window_size

        x = np.nan_to_num(np.array(gain))
        preds = np.zeros(len(x), dtype=float)
        nrows = np.shape(self.training_data)[0]

        start = 0
        end = start + win_size
        scale_start = max(0, end - self.scale_len)

        # train_end = max(0, start  - self.lookahead - 1)
        # train_end = max(0, start  - 1)
        # train_end = max(0, start - self.lookahead - 1)
        train_end = min(
            end - self.lookahead - 1, nrows - self.lookahead - 2
        )  # potential lookahead problem
        train_start = max(0, train_end - self.train_len)

        # get the forecaster for this pair
        if self.custom_trade_info[self.curr_pair]["forecaster"] is None:
            # make a deep copy so that we don't override the baseline model
            pair_forecaster = copy.deepcopy(self.forecaster)
            self.custom_trade_info[self.curr_pair]["forecaster"] = pair_forecaster
        else:
            pair_forecaster = self.custom_trade_info[self.curr_pair]["forecaster"]

        # loop through each row
        while end <= len(x):
            if start < (self.wavelet_size + self.lookahead):  # need buffer for training
                preds[end - 1] = 0.0
            else:
                # (re-)train the model on prior data and get predictions

                if (not self.training_mode) and (self.supports_incremental_training):
                    train_data = self.training_data[train_start:train_end].copy()
                    train_results = self.training_labels[train_start:train_end].copy()
                    # pair_forecaster = copy.deepcopy(self.forecaster)  # reset to avoid over-training
                    self.train_model(pair_forecaster, train_data, train_results, False)
                    # print(f'    start:{start} end:{end} train_start:{train_start} train_end:{train_end}')

                # rebuild data up to end of current window
                dslice = self.training_data[start:end].copy()
                self.gain_data = x[scale_start:end]  # needed for scaling
                forecast = self.predict_data(pair_forecaster, dslice)

                # print(f'    forecast:{forecast}')
                preds[end - 1] = forecast[-1]

            # move the window to the next segment
            end = end + 1
            start = start + 1
            # train_end = start - self.lookahead - 1
            # train_end = start - 1
            train_end = min(
                end - self.lookahead - 1, nrows - self.lookahead - 2
            )  # potential lookahead problem
            train_start = max(0, train_end - self.train_len)

        # save the updated/trained forecaster
        self.custom_trade_info[self.curr_pair]["forecaster"] = pair_forecaster

        return preds

    # ----------
    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]

        # set up training data
        future_gain_data = self.get_future_gain(df)
        data, _ = self.get_data(dataframe)

        self.training_data = data.copy()
        self.training_labels = np.zeros(np.shape(future_gain_data), dtype=float)
        self.training_labels = future_gain_data.copy()

        # initialise the prediction array, using the close data
        pred_array = np.zeros(np.shape(future_gain_data), dtype=float)

        win_size = self.model_window

        # loop until we get to/past the end of the buffer
        # start = win_size
        start = self.lookahead + self.train_len
        end = start + win_size - 1
        # train_end = max(0, start - self.lookahead - 1)
        # train_end = max(0, end - self.lookahead - 1)
        train_size = self.train_len
        # train_start = max(0, train_end - train_size)
        scale_start = max(0, end - self.scale_len)

        train_end = min(
            end - self.lookahead - 1, nrows - self.lookahead - 2
        )  # potential lookahead problem
        train_start = max(0, train_end - self.train_len)

        # get the forecaster for this pair
        if self.custom_trade_info[self.curr_pair]["forecaster"] is None:
            # make a deep copy so that we don't override the baseline model
            pair_forecaster = copy.deepcopy(self.forecaster)
        else:
            pair_forecaster = self.custom_trade_info[self.curr_pair]["forecaster"]

        # loop through the rows
        while end < nrows:
            # extract the data and coefficients from the current window

            # (re-)train the model on prior data and get predictions

            if (not self.training_mode) and (self.supports_incremental_training):
                train_data = self.training_data[train_start:train_end].copy()
                train_results = self.training_labels[train_start:train_end].copy()
                pair_forecaster = copy.deepcopy(
                    self.forecaster
                )  # reset to avoid over-training
                self.train_model(pair_forecaster, train_data, train_results, False)
                # print(f'train_data: {np.shape(train_data)}')
                # print(f'train_results: {np.shape(train_results)}')

            # rebuild data up to end of current window
            dslice = self.training_data[start:end].copy()
            self.gain_data = np.array(
                dataframe["gain"].iloc[scale_start:end]
            )  # needed for scaling
            preds = self.predict_data(pair_forecaster, dslice)

            # print(f'dslice: {np.shape(dslice)}')
            # print(f'preds: {np.shape(preds)}')

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds.copy()

            # move the window to the next segment
            end = end + win_size
            start = start + win_size
            train_end = end - self.lookahead - 1
            train_start = max(0, train_end - train_size)

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data

        # predict for last window
        dslice = self.training_data[-win_size:]
        # preds = self.forecaster.predict(dslice)
        slen = min(win_size, 32)
        self.gain_data = np.array(dataframe["gain"].iloc[-slen:])  # needed for scaling
        preds = self.predict_data(pair_forecaster, dslice)
        pred_array[-len(preds) :] = preds.copy()

        dataframe["predicted_gain"] = pred_array.copy()

        # save the updated/trained forecaster
        self.custom_trade_info[self.curr_pair]["forecaster"] = pair_forecaster

        return dataframe

    # -------------

    # -------------
    def add_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:
        try:
            nrows = dataframe.shape[0]
            future_gain_data = self.get_future_gain(dataframe)
            data, column_names = self.get_data(dataframe)

            # Initialise the prediction array
            results_all = np.zeros(nrows)

            # Walk-forward training to avoid lookahead bias and speed up processing
            chunk_size = 2500  # Re-train every 2500 candles for speed
            initial_train_size = max(self.train_min_len, 2000)

            print(
                f"    Training TSPredict Causal Model ({'MLX' if self.use_mlx else 'LightGBM'})..."
            )

            for win_end in range(initial_train_size, nrows, chunk_size):
                # train_end must be behind win_end by at least lookahead
                # however, get_future_gain already shifts the labels.
                # So to be safe, we only train up to win_end - lookahead
                train_end = win_end - self.lookahead
                predict_end = min(win_end + chunk_size, nrows)

                if train_end <= 0:
                    continue

                X_train = data[:train_end]
                y_train = future_gain_data[:train_end]
                X_predict = data[win_end:predict_end]

                if len(X_predict) == 0:
                    continue

                # Selective scaling: only scale columns not in pre_normalized_columns
                needs_scale_indices = [
                    i
                    for i, col in enumerate(column_names)
                    if col not in self.pre_normalized_columns
                ]
                X_train_processed = X_train.copy()
                X_predict_processed = X_predict.copy()
                
                if len(needs_scale_indices) > 0:
                    scaler = None
                    scaler_dir = self.get_storage_location()
                    scaler_name = "main_scaler"
                    
                    # Try to load global scaler
                    if scaler_exists(scaler_dir, scaler_name):
                        try:
                            global_scaler = load_scaler(scaler_dir, scaler_name)
                            # Check compatibility (feature count must match)
                            if hasattr(global_scaler, "n_features_in_") and global_scaler.n_features_in_ == len(needs_scale_indices):
                                scaler = global_scaler
                        except Exception as e:
                            print(f"    WARN: could not load global scaler: {e}")

                    if scaler is None:
                        # Fallback to local RobustScaler
                        scaler = RobustScaler()
                        X_train_processed[:, needs_scale_indices] = scaler.fit_transform(X_train[:, needs_scale_indices])
                    else:
                        # Use global scaler (no fit_transform, just transform)
                        X_train_processed[:, needs_scale_indices] = scaler.transform(X_train[:, needs_scale_indices])

                    X_predict_processed[:, needs_scale_indices] = scaler.transform(X_predict[:, needs_scale_indices])

                if self.use_mlx:
                    # MLX Path
                    model = train_mlx_global(X_train_processed, y_train, epochs=60)
                    X_mx = mx.array(X_predict_processed)
                    preds_mx = model(X_mx)
                    mx.eval(preds_mx)
                    chunk_preds = np.array(preds_mx).squeeze()
                else:
                    # LightGBM Path
                    model = LGBMRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train_processed, y_train)
                    chunk_preds = model.predict(X_predict_processed)

                # Ensure chunk_preds is 1D if single output
                if chunk_preds.ndim > 1:
                    chunk_preds = chunk_preds.ravel()

                results_all[win_end:predict_end] = chunk_preds

            dataframe.iloc[:, dataframe.columns.get_loc("predicted_gain")] = results_all

        except Exception as e:
            print("*** Exception in add_rolling_predictions()")
            print(e)
            print(traceback.format_exc())

        return dataframe

    # -------------

    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe

        try:
            # set up training data
            # TODO: see if we can do this incrementally instead of rebuilding every time, or just use portion of data
            future_gain_data = self.get_future_gain(df)
            data, _ = self.get_data(dataframe)

            plen = len(self.custom_trade_info[self.curr_pair]["predictions"])
            dlen = len(dataframe["gain"])
            clen = min(plen, dlen)

            # self.training_data = data[-clen:].copy()
            # self.training_labels = future_gain_data[-clen:].copy()
            self.training_data = data
            self.training_labels = future_gain_data

            pred_array = np.zeros(clen, dtype=float)

            # print(f"[predictions]:{np.shape(self.custom_trade_info[self.curr_pair]['predictions'])}  pred_array:{np.shape(pred_array)}")

            # copy previous predictions and shift down by 1
            pred_array[-clen:] = self.custom_trade_info[self.curr_pair]["predictions"][
                -clen:
            ].copy()
            pred_array = np.roll(pred_array, -1)
            pred_array[-1] = 0.0

            # train on previous data
            # train_end = clen - self.model_window - self.lookahead
            train_end = np.shape(self.training_data)[0] - self.lookahead - 2
            train_start = max(0, train_end - self.train_len)

            # cannot use last portion because we are looking ahead
            tslice = self.training_data[train_start:train_end]
            lslice = self.training_labels[train_start:train_end]

            # get the forecaster for this pair
            if self.custom_trade_info[self.curr_pair]["forecaster"] is None:
                # make a deep copy so that we don't override the baseline model
                pair_forecaster = copy.deepcopy(self.forecaster)
                # forecaster should already be there, so print warning
                print(
                    f"    *** WARNING: No pre-existing forecaster. Creating from model"
                )
            else:
                pair_forecaster = self.custom_trade_info[self.curr_pair]["forecaster"]

            # update forecaster and get predictions

            self.train_model(pair_forecaster, tslice, lslice, False)

            slen = min(clen, self.scale_len)
            self.gain_data = np.array(
                dataframe["gain"].iloc[-slen:]
            )  # needed for scaling
            preds = self.predict_data(
                pair_forecaster, self.training_data[-self.model_window :]
            )

            # self.forecaster = copy.deepcopy(base_forecaster) # restore original model

            # only replace last prediction (i.e. don't overwrite the historical predictions)
            pred_array[-1] = preds[-1]

            dataframe["predicted_gain"] = 0.0
            dataframe["predicted_gain"][-clen:] = pred_array[-clen:].copy()
            self.custom_trade_info[self.curr_pair]["predictions"][-clen:] = pred_array[
                -clen:
            ].copy()

            # save the updated/trained forecaster
            self.custom_trade_info[self.curr_pair]["forecaster"] = pair_forecaster

            """"""
            # Debug: print info if in buy or sell region (nothing otherwise)
            pg = preds[-1]
            if pg <= dataframe["target_loss"].iloc[-1]:
                print(f"    (v) predict {pg:6.2f}% loss for:   {self.curr_pair}")
            elif pg >= dataframe["target_profit"].iloc[-1]:
                print(f"     ^  predict {pg:6.2f}% profit for: {self.curr_pair}")

            """"""

        except Exception as e:
            print("*** Exception in add_latest_prediction()")
            print(e)  # prints the error message
            print(traceback.format_exc())  # prints the full traceback

        return dataframe

    # -------------

    # add predictions to dataframe['predicted_gain']
    def add_predictions(self, dataframe: DataFrame) -> DataFrame:
        # print(f"    {self.curr_pair} adding predictions")

        run_profiler = False

        if run_profiler:
            prof = cProfile.Profile()
            prof.enable()

        self.scaler = RobustScaler()  # reset scaler each time

        self.init_model(dataframe)

        if self.curr_pair not in self.custom_trade_info:
            self.custom_trade_info[self.curr_pair] = {
                "forecaster": None,
                "initialised": False,
                "predictions": None,
                "curr_prediction": 0.0,
                "curr_target": 0.0,
            }

        if self.training_mode:
            print(f"    Training mode. Skipping backtest for {self.curr_pair}")
            dataframe["predicted_gain"] = 0.0
        else:
            """
            if not self.custom_trade_info[self.curr_pair]["initialised"]:
                print(f"    backtesting {self.curr_pair}")
                if self.use_rolling:
                    dataframe = self.add_rolling_predictions(dataframe)
                else:
                    dataframe = self.add_jumping_predictions(dataframe)

                self.custom_trade_info[self.curr_pair]["initialised"] = True
                self.custom_trade_info[self.curr_pair]["predictions"] = dataframe["predicted_gain"].copy()
            else:
                # print(f'    updating latest prediction for: {self.curr_pair}')
                dataframe = self.add_latest_prediction(dataframe)

                # save latest prediction and threshold for later use (where dataframe is not available)
                self.custom_trade_info[self.curr_pair]["curr_prediction"] = dataframe["predicted_gain"].iloc[-1]
                self.custom_trade_info[self.curr_pair]["curr_target"] = dataframe["target_profit"].iloc[-1]

            """

            print(f"    backtesting {self.curr_pair}")
            if self.use_rolling:
                dataframe = self.add_rolling_predictions(dataframe)
            else:
                dataframe = self.add_jumping_predictions(dataframe)

            # predictions can spike, so constrain range and smooth slightly
            dataframe["predicted_gain"] = (
                dataframe["predicted_gain"].fillna(0.0).clip(lower=-3.0, upper=3.0)
            )
            dataframe["predicted_gain"] = self.super_smoother(
                dataframe["predicted_gain"], 4
            ).fillna(0.0)

            # save target rate for later use
            dataframe["curr_target"] = dataframe["close"] * (
                1.0 + dataframe["predicted_gain"] / 100.0
            )
            # TODO: really should set target to value predicted at previous buy signal

            # save latest prediction and threshold for later use (where dataframe is not available)
            curr_prediction = dataframe["predicted_gain"].iloc[-1]
            curr_target = dataframe["close"].iloc[-1] * (1.0 + curr_prediction / 100.0)
            self.custom_trade_info[self.curr_pair]["curr_prediction"] = curr_prediction
            self.custom_trade_info[self.curr_pair]["curr_target"] = curr_target

        # add shifted version, for debug only
        dataframe["shifted_pred"] = dataframe["predicted_gain"].shift(self.lookahead)

        if run_profiler:
            prof.disable()
            # print profiling output
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(20)  # top 20 rows

        return dataframe

    ###################################

    def get_entry_conditions(self, dataframe: DataFrame) -> Series:
        if self.training_mode:
            return pd.Series([False] * len(dataframe))

        # update gain targets here so that we can use hyperopt parameters
        dataframe = self.update_gain_targets(dataframe)

        # model triggers
        # (choose one)
        # threshold = dataframe["target_profit"]  # breakout
        threshold = dataframe["target_loss"]  # mean reversion
        model_cond = (
            # prediction crossed target
            qtpylib.crossed_above(dataframe["predicted_gain"], threshold)
            # | (
            #     # add this version if volume checks are enabled, because we might miss the crossing otherwise
            #     (dataframe["predicted_gain"] > dataframe["target_profit"])
            #     & (
            #         dataframe["predicted_gain"].shift()
            #         > dataframe["target_profit"].shift()
            #     )
            # )
        )

        return model_cond

    """
    exit Signal
    """

    def get_exit_conditions(self, dataframe: DataFrame) -> Series:
        if self.training_mode:
            return pd.Series([False] * len(dataframe))

        # model triggers
        model_cond = (
            # prediction crossed target
            qtpylib.crossed_below(
                dataframe["predicted_gain"], dataframe["target_profit"]
            )
        )

        return model_cond
