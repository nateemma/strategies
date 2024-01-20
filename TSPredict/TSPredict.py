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
             Models are saved in user_data/strategies/TSPredict/models/<class>/<class>.sav, where <class> is the name
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


# from lightgbm import LGBMRegressor
# from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from xgboost import XGBRegressor

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)


import utils.custom_indicators as cta

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from utils.DataframeUtils import DataframeUtils, ScalerType  # pylint: disable=E0401

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

pd.options.mode.chained_assignment = None  # default='warn'


class TSPredict(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    plot_config = {
        "main_plot": {
            "close": {"color": "cornflowerblue"},
            },
        "subplots": {
            "Diff": {
                "predicted_gain": {"color": "rebeccapurple"},
                "shifted_pred": {"color": "skyblue"},
                # "squeeze": {"color": "red"},
                "gain": {"color": "green"},
                "target_profit": {"color": "lightgreen"},
                "target_loss": {"color": "lightsalmon"},
                "buy_region": {"color": "darkseagreen"},
                "sell_region": {"color": "darksalmon"},
            },
        },
    }

    # ROI table:
    minimal_roi = {"0": 0.04, "100": 0.02}

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = "5m"
    inf_timeframe = "15m"

    use_custom_stoploss = True

    leverage = 1.0
    can_short = False
    # if setting can-short to True, remember to update the config file:
    #   "trading_mode": "futures",
    #   "margin_mode": "isolated",

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2

    process_only_new_candles = True

    custom_trade_info = {}  # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    lookahead = 6

    df_coeffs: DataFrame = None
    coeff_table = None
    coeff_array = None
    gain_data = None
    merge_indicators = False # set to False to not merge indicators into prediction data



    use_rolling = True # True = rolling (slow but realistic), False = Jumping (much faster, less realistic)
    single_col_prediction = False # True = use only gain. False = use all columns (better, but much slower)

    wavelet_type:Wavelets.WaveletType = Wavelets.WaveletType.DWTA
    wavelet = None

    forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.PA
    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.SGD
    # forecaster_type:Forecasters.ForecasterType = Forecasters.ForecasterType.SVR
    forecaster = None

    data = None

    wavelet_size = 32 # needed for consistently-sized transforms
    win_size = wavelet_size # this can vary

    train_min_len = wavelet_size # longer = slower
    train_len = min(128, wavelet_size * 4) # longer = slower
    # scale_len = wavelet_size // 2 # no. recent candles to use when scaling
    scale_len = min(8, wavelet_size//2) # no. recent candles to use when scaling
    win_size = min(32, wavelet_size)
    model_window = wavelet_size # longer = slower

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

    norm_data = False # changing this requires new models

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
        "cexit_min_profit_th": 0.6,
        "cexit_profit_nstd": 1.0,
        "enable_bb_check": False,
        "entry_bb_factor": 1.09,
        "entry_bb_width": 0.026,
        "entry_guard_metric": -0.3,
        "enable_guard_metric": True,  # value loaded from strategy
        "enable_squeeze": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_loss_nstd": 2.9,
        "cexit_metric_overbought": 0.68,
        "cexit_metric_take_profit": 0.56,
        "cexit_min_loss_th": -0.3,
        "enable_exit_signal": True,
        "exit_bb_factor": 1.13,
        "exit_guard_metric": 0.6,
    }

    # Entry

    # the following flags apply to both entry and exit
    enable_guard_metric = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )

    enable_bb_check = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=True
        )

    enable_squeeze = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )

    entry_guard_metric = DecimalParameter(
        -0.8, 0.0, default=-0.2, decimals=1, space="buy", load=True, optimize=True
        )

    entry_bb_width = DecimalParameter(
        0.020, 0.100, default=0.02, decimals=3, space="buy", load=True, optimize=True
        )

    entry_bb_factor = DecimalParameter(
        0.70, 1.20, default=1.1, decimals=2, space="buy", load=True, optimize=True
        )


    # Exit
    # use exit signal? If disabled, just rely on the custom exit checks (or stoploss) to get out
    enable_exit_signal = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=True
        )

    exit_guard_metric = DecimalParameter(
        0.0, 0.8, default=0.0, decimals=1, space="sell", load=True, optimize=True
        )

    exit_bb_factor = DecimalParameter(
        0.70, 1.20, default=0.8, decimals=2, space="sell", load=True, optimize=True
        )



    # Custom Exit

    # No. Standard Deviations of profit/loss for target, and lower limit
    cexit_min_profit_th = DecimalParameter(0.0, 1.5, default=0.7, decimals=1, space="buy", load=True, optimize=True)
    cexit_profit_nstd = DecimalParameter(0.0, 3.0, default=0.9, decimals=1, space="buy", load=True, optimize=True)

    cexit_min_loss_th = DecimalParameter(-1.5, -0.0, default=-0.4, decimals=1, space="sell", load=True, optimize=True)
    cexit_loss_nstd = DecimalParameter(0.0, 3.0, default=0.7, decimals=1, space="sell", load=True, optimize=True)

    # Guard metric sell limits - used to bail out when in profit
    cexit_metric_overbought = DecimalParameter(
        0.55, 0.99, default=0.96, decimals=2, space="sell", load=True, optimize=True
        )

    cexit_metric_take_profit = DecimalParameter(
        0.55, 0.99, default=0.76, decimals=2, space="sell", load=True, optimize=True
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

        if (not self.forecaster.supports_multiple_columns()):
            print('    ****')
            print(f'    **** ERROR: forecaster ({self.forecaster_type.name}) does not support multiple indicators')
            print('    ****')

        if (not self.forecaster.supports_retrain()):
            print('    ****')
            print(f'    **** WARNING: forecaster ({self.forecaster_type.name}) does not support retrainings')
            print('    ****')

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

        if (len(removed_pairs) > 0):
            print("    Pairlist changed:")
            print(f'    old pairs: {saved_pairlist}')
            print(f'    new pairs: {curr_pairlist}')
            print(f'    pairs removed: {removed_pairs}')
            print(f'    pairs added: {added_pairs}')

            for pair in removed_pairs:
                print(f'    Removing historical data for: {pair}')
                del self.custom_trade_info[pair]

    ###################################
    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # NOTE: if you change the indicators, you need to regenerate the model

        window_size = min(32, self.win_size)

        # Base pair dataframe timeframe indicators
        curr_pair = metadata["pair"]

        self.curr_dataframe = dataframe
        self.curr_pair = curr_pair

        self.update_pairlist_data()

        # The following are needed for base functions, so do not remove.
        # Add custom indicators to add_strategy_indicators()

        # backward looking gain
        dataframe["gain"] = (
            100.0
            * (dataframe["close"] - dataframe["close"].shift(self.lookahead))
            / dataframe["close"].shift(self.lookahead)
        )
        dataframe["gain"].fillna(0.0, inplace=True)
        dataframe["gain"] = dataframe["gain"].round(4)

        # dataframe["gain"] = self.smooth(dataframe["gain"], 8)

        # need to save the gain data for later scaling
        self.gain_data = dataframe["gain"].to_numpy().copy()

        # target profit/loss thresholds
        dataframe["profit"] = dataframe["gain"].clip(lower=0.0)
        dataframe["loss"] = dataframe["gain"].clip(upper=0.0)

        dataframe = self.update_gain_targets(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=window_size, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])


        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=window_size)

        '''
        # Williams %R
        dataframe["wr"] = 0.02 * (self.williams_r(dataframe, period=window_size) + 50.0)

        # Fisher RSI
        rsi = 0.1 * (dataframe["rsi"] - 50)
        dataframe["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe["fisher_wr"] = (dataframe["wr"] + dataframe["fisher_rsi"]) / 2.0

        '''

        # init prediction column
        dataframe["predicted_gain"] = 0.0

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=window_size, mom=5)

        # scaled version for use as guard metric
        dataframe['srmi'] = 2.0 * (dataframe['rmi'] - 50.0) / 100.0

        # guard metric must be in range [-1,+1], with -ve values indicating oversold and +ve values overbought
        dataframe['guard_metric'] = dataframe['srmi']


        use_kc_squeeze = False

        if use_kc_squeeze:
            # Calculate the Keltner Channel
            keltner = qtpylib.keltner_channel(dataframe, window=window_size, atrs=4)
            upper_kc = keltner["upper"]
            lower_kc = keltner["lower"]
            upper_bb = dataframe['bb_upperband']
            lower_bb = dataframe['bb_lowerband']

            # calculate BB/KC squeeze
            dataframe['squeeze'] = np.where(((lower_bb > lower_kc) & (upper_bb < upper_kc)), 1, 0)

        # Add strategy-specific indicators
        dataframe = self.add_strategy_indicators(dataframe)

        # create and init the model, if first time (dataframe has to be populated first)
        if self.model is None:
            # print("    Loading model")
            self.load_model(np.shape(dataframe))

        # add the predictions
        # print("    Making predictions...")
        dataframe = self.add_predictions(dataframe)

        dataframe.fillna(0.0, inplace=True)

        # #DBG (cannot include this in 'real' strat because it's forward looking):
        # dataframe['dwt'] = self.get_dwt(dataframe['gain'])

        return dataframe

    def update_gain_targets(self, dataframe):
        # win_size = max(self.lookahead, 6)
        win_size = self.scale_len
        self.profit_nstd = float(self.cexit_profit_nstd.value)
        self.loss_nstd = float(self.cexit_loss_nstd.value)

        dataframe["target_profit"] = (
            dataframe["profit"].rolling(window=win_size).mean()
            + self.profit_nstd * dataframe["profit"].rolling(window=win_size).std()
        )

        dataframe["target_loss"] = dataframe["loss"].rolling(window=win_size).mean() - self.loss_nstd * abs(
            dataframe["loss"].rolling(window=win_size).std()
        )

        dataframe["target_profit"] = dataframe["target_profit"].clip(lower=float(self.cexit_min_profit_th.value))
        dataframe["target_loss"] = dataframe["target_loss"].clip(upper=float(self.cexit_min_loss_th.value))

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

    def smooth(self, y, window):
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode="same")
        # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        y_smooth = np.round(y_smooth, decimals=3)
        return np.nan_to_num(y_smooth)

    #-----------------------

    # look ahead to get future gain. Do *not* put this into the main dataframe!
    def get_future_gain(self, dataframe):
        df = self.convert_dataframe(dataframe)
        future_gain = df["gain"].shift(-self.lookahead).to_numpy()
        future_gain[-self.lookahead:] = 0.0
        future_gain = np.round(future_gain, decimals=3)
        future_gain = np.nan_to_num(future_gain)

        # future_gain = dataframe['gain'].shift(-self.lookahead).to_numpy()
        # return self.smooth(future_gain, 8)
        return future_gain

    #-------------
    # Normalisation

    array_scaler = RobustScaler()

    def update_scaler(self, data):

        if not self.array_scaler:
            self.array_scaler = RobustScaler()

        self.array_scaler.fit(data.reshape(-1,1))

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

    #-------------

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

        # convert date column so that it can be scaled.
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], utc=True)
            df["date"] = dates.astype("int64")

        df.fillna(0.0, inplace=True)

        df.set_index("date")
        df.reindex()

        # print(f'    norm_data:{self.norm_data}')
        if self.norm_data:
            # scale the dataframe
            self.scaler.fit(df)
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        return df

    ###################################

    # Model-related funcs. Override in subclass to use a different type of model

    def get_model_path(self, pair):
        category = self.__class__.__name__
        root_dir = group_dir + "/models/" + category
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
    def train_model(self, forecaster:Forecasters.base_forecaster, data: np.array, results: np.array, save_model):
        if forecaster is None:
            print("***    ERR: no forecaster ***")
            return

        x = np.nan_to_num(data)
        y = np.nan_to_num(results)

        forecaster.train(x, y)

        # print(f'   train_model() data:{np.shape(data)} results:{np.shape(results)}')

        return

    # -------------

    # initial training of the model
    def init_model(self, dataframe: DataFrame):
        # if model is not yet trained, or this is a new model and we want to combine across pairs, then train
        if (not self.model_trained) or (self.new_model and self.combine_models):
            df = dataframe

            future_gain_data = self.get_future_gain(df)
            data = self.get_data(df)

            if self.single_col_prediction:
                training_data = dataframe['gain'].to_numpy()
                training_data = self.smooth(training_data, 2)
                training_data = training_data.reshape(-1,1)
            else:
                training_data = data.copy()
            training_data = training_data[: -self.lookahead - 1]
            training_labels = future_gain_data[: -self.lookahead - 1].copy()

            if not self.model_trained:
                print(f"    initial training ({self.curr_pair})")
            else:
                print(f"    incremental training ({self.curr_pair})")

            self.train_model(self.forecaster, training_data, training_labels, True)

            self.model_trained = True

            if self.new_model:
                self.save_model()

        # print(f'    model_trained:{self.model_trained} new_model:{self.new_model}  combine_models:{self.combine_models}')

        return

    # -------------

    # set the data for this straegy. Override if necessary
    def get_data(self, dataframe):
        # default is to just normalise the dataframe and convert to numpy array
        self.curr_dataframe = dataframe
        df = dataframe.copy()
        gain = df['gain'].to_numpy()
        gain = self.smooth(gain, 2)
        df['gain'] = gain
        self.data = np.array(self.convert_dataframe(df))
        return self.data

    # -------------

    # generate predictions for an np array (intended to be overriden if needed)
    def predict_data(self, forecaster:Forecasters.base_forecaster, data):
        x = np.nan_to_num(data)

        preds = forecaster.forecast(x, self.lookahead)

        # print(f'    data:{np.shape(data)} preds:{np.shape(preds)}')

        # smooth predictions to try and avoid drastic changes
        preds = self.smooth(preds, 4)

        # scale the results to generally match the input characteristics
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
        end_row = dataframe.index.get_loc(end) + 1 # need to add the 1, don't know why!


        # if end_row < (self.wavelet_size + self.lookahead):
        if start_row < (self.wavelet_size + self.lookahead): # need buffer for training
            return 0.0

        # train on previous data
        train_end = start_row - self.lookahead - 1
        train_start = max(0, train_end-self.train_len)
        scale_start = max(0, end-self.scale_len)

        if (not self.training_mode) and (self.supports_incremental_training):
            train_data = self.training_data[train_start : start - 1].copy()
            train_results = self.training_labels[train_start : start - 1].copy()
            pair_forecaster = copy.deepcopy(self.forecaster)  # reset to avoid over-training
            self.train_model(pair_forecaster, train_data, train_results, False)
        else:
            pair_forecaster = self.forecaster

        # predict for current window
        dslice = self.training_data[start:end].copy()
        self.gain_data = np.array(dataframe["gain"].iloc[scale_start:end])  # needed for scaling
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
        scale_start = max(0, end-self.scale_len)

        # train_end = max(0, start  - self.lookahead - 1)
        # train_end = max(0, start  - 1)
        # train_end = max(0, start - self.lookahead - 1)
        train_end = min(end - self.lookahead - 1, nrows - self.lookahead - 2) # potential lookahead problem
        train_start = max(0, train_end-self.train_len)


        # get the forecaster for this pair
        if self.custom_trade_info[self.curr_pair]['forecaster'] is None:
            # make a deep copy so that we don't override the baseline model
            pair_forecaster = copy.deepcopy(self.forecaster)
        else:
            pair_forecaster = self.custom_trade_info[self.curr_pair]['forecaster']

        # loop through each row
        while end < len(x):

            if start < (self.wavelet_size + self.lookahead): # need buffer for training
                preds[end] = 0.0
            else:
                # (re-)train the model on prior data and get predictions

                if (not self.training_mode) and (self.supports_incremental_training):
                    train_data = self.training_data[train_start : train_end].copy()
                    train_results = self.training_labels[train_start : train_end].copy()
                    # pair_forecaster = copy.deepcopy(self.forecaster)  # reset to avoid over-training
                    self.train_model(pair_forecaster, train_data, train_results, False)
                    # print(f'    start:{start} end:{end} train_start:{train_start} train_end:{train_end}')

                # rebuild data up to end of current window
                dslice = self.training_data[start:end].copy()
                self.gain_data = x[scale_start:end]  # needed for scaling
                forecast = self.predict_data(pair_forecaster, dslice)

                # print(f'    forecast:{forecast}')
                preds[end] = forecast[-1]

            # move the window to the next segment
            end = end + 1
            start = start + 1
            # train_end = start - self.lookahead - 1
            # train_end = start - 1
            train_end = min(end - self.lookahead - 1, nrows - self.lookahead - 2) # potential lookahead problem
            train_start = max(0, train_end-self.train_len)

        # save the updated/trained forecaster
        self.custom_trade_info[self.curr_pair]['forecaster'] = pair_forecaster

        return preds

#----------
    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]

        # set up training data
        future_gain_data = self.get_future_gain(df)
        data = self.get_data(dataframe)

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
        train_end = max(0, start - self.lookahead - 1)
        train_size = self.train_len
        train_start = max(0, train_end - train_size)
        scale_start = max(0, end-self.scale_len)

        # get the forecaster for this pair
        if self.custom_trade_info[self.curr_pair]['forecaster'] is None:
            # make a deep copy so that we don't override the baseline model
            pair_forecaster = copy.deepcopy(self.forecaster)
        else:
            pair_forecaster = self.custom_trade_info[self.curr_pair]['forecaster']

        # loop through the rows
        while end < nrows:
            # extract the data and coefficients from the current window

            # (re-)train the model on prior data and get predictions

            if (not self.training_mode) and (self.supports_incremental_training):
                train_data = self.training_data[train_start : start - 1].copy()
                train_results = self.training_labels[train_start : start - 1].copy()
                pair_forecaster = copy.deepcopy(self.forecaster)  # reset to avoid over-training
                self.train_model(pair_forecaster, train_data, train_results, False)
                # print(f'train_data: {np.shape(train_data)}')
                # print(f'train_results: {np.shape(train_results)}')

            # rebuild data up to end of current window
            dslice = self.training_data[start:end].copy()
            self.gain_data = np.array(dataframe["gain"].iloc[scale_start:end])  # needed for scaling
            preds = self.predict_data(pair_forecaster, dslice)

            # print(f'dslice: {np.shape(dslice)}')
            # print(f'preds: {np.shape(preds)}')

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds.copy()

            # move the window to the next segment
            end = end + win_size
            start = start + win_size
            train_end = start - self.lookahead - 1
            train_start = max(0, train_end - train_size)

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data

        # predict for last window
        dslice = self.training_data[-win_size:]
        # preds = self.forecaster.predict(dslice)
        slen = min(win_size, 32)
        self.gain_data = np.array(dataframe["gain"].iloc[-slen:])  # needed for scaling
        preds = self.predict_data(pair_forecaster, dslice)
        pred_array[-slen:] = preds.copy()

        dataframe["predicted_gain"] = pred_array.copy()

        # save the updated/trained forecaster
        self.custom_trade_info[self.curr_pair]['forecaster'] = pair_forecaster

        return dataframe

    # -------------

    def add_rolling_predictions(self, dataframe: DataFrame) -> DataFrame:

        try:
            # set up training data
            future_gain_data = self.get_future_gain(dataframe)
            data = self.get_data(dataframe)

            if self.single_col_prediction:
                self.training_data = dataframe['gain'].to_numpy()
                self.training_data = self.smooth(self.training_data, 2)
                self.training_data = self.training_data.reshape(-1,1)
            else:
                self.training_data = data.copy()

            self.training_labels = np.zeros(np.shape(future_gain_data), dtype=float)
            self.training_labels = future_gain_data.copy()

            # dataframe['predicted_gain'] = dataframe['gain'].rolling(window=self.model_window).apply(self.predict, args=(dataframe,))
            dataframe['predicted_gain'] = self.rolling_predict(dataframe['gain'], self.model_window)

            # dataframe['predicted_gain'] = self.smooth(dataframe['predicted_gain'], 2)

        except Exception as e:
            print("*** Exception in add_rolling_predictions()")
            print(e) # prints the error message
            print(traceback.format_exc()) # prints the full traceback

        return dataframe

    # -------------

    # add the latest prediction, and update training periodically
    def add_latest_prediction(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe

        try:
            # set up training data
            # TODO: see if we can do this incrementally instead of rebuilding every time, or just use portion of data
            future_gain_data = self.get_future_gain(df)
            data = self.get_data(dataframe)

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
            pred_array[-clen:] = self.custom_trade_info[self.curr_pair]["predictions"][-clen:].copy()
            pred_array = np.roll(pred_array, -1)
            pred_array[-1] = 0.0

            # train on previous data
            # train_end = clen - self.model_window - self.lookahead
            train_end = np.shape(self.training_data)[0] - self.lookahead - 2
            train_start = max(0, train_end-self.train_len)

            # cannot use last portion because we are looking ahead
            tslice = self.training_data[train_start:train_end]
            lslice = self.training_labels[train_start:train_end]

            # get the forecaster for this pair
            if self.custom_trade_info[self.curr_pair]['forecaster'] is None:
                # make a deep copy so that we don't override the baseline model
                pair_forecaster = copy.deepcopy(self.forecaster)
                # forecaster should already be there, so print warning
                print(f'    *** WARNING: No pre-existing forecaster. Creating from model')
            else:
                pair_forecaster = self.custom_trade_info[self.curr_pair]['forecaster']

            # update forecaster and get predictions

            self.train_model(pair_forecaster, tslice, lslice, False)

            slen = min(clen, self.scale_len)
            self.gain_data = np.array(dataframe["gain"].iloc[-slen:])  # needed for scaling
            preds = self.predict_data(pair_forecaster, self.training_data[-self.model_window:])

            # self.forecaster = copy.deepcopy(base_forecaster) # restore original model

            # only replace last prediction (i.e. don't overwrite the historical predictions)
            pred_array[-1] = preds[-1]

            dataframe["predicted_gain"] = 0.0
            dataframe["predicted_gain"][-clen:] = pred_array[-clen:].copy()
            self.custom_trade_info[self.curr_pair]["predictions"][-clen:] = pred_array[-clen:].copy()

            # save the updated/trained forecaster
            self.custom_trade_info[self.curr_pair]['forecaster'] = pair_forecaster

            ''''''
            # Debug: print info if in buy or sell region (nothing otherwise)
            pg = preds[-1]
            if pg <= dataframe['target_loss'].iloc[-1]:
                print(f'    (v) predict {pg:6.2f}% loss for:   {self.curr_pair}')
            elif pg >= dataframe['target_profit'].iloc[-1]:
                print(f'     ^  predict {pg:6.2f}% profit for: {self.curr_pair}')

            ''''''

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
                'forecaster':  None,
                "initialised": False,
                "predictions": None,
                "curr_prediction": 0.0,
                "curr_target": 0.0,
            }

        if self.training_mode:
            print(f"    Training mode. Skipping backtest for {self.curr_pair}")
            dataframe["predicted_gain"] = 0.0
        else:
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

        # predictions can spike, so constrain range
        dataframe["predicted_gain"] = dataframe["predicted_gain"].clip(lower=-3.0, upper=3.0)

        # ad shifted version, for debug only
        dataframe['shifted_pred'] = dataframe['predicted_gain'].shift(self.lookahead)

        if run_profiler:
            prof.disable()
            # print profiling output
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(20)  # top 20 rows

        return dataframe

    ###################################

    """
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe['buy_region'] = 0

        if self.training_mode:
            return dataframe

        # update gain targets here so that we can use hyperopt parameters
        dataframe = self.update_gain_targets(dataframe)

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 1.0)

        guard_conditions = []

        if self.enable_guard_metric.value:

            # Guard metric in oversold region
            guard_conditions.append(dataframe["guard_metric"] < self.entry_guard_metric.value)

            # in lower portion of previous window
            # conditions.append(dataframe["close"] < dataframe["local_mean"])

        if self.enable_bb_check.value:
            # Bollinger band-based bull/bear indicators:
            # Done here so that we can use hyperopt to find values

            lower_limit = dataframe['bb_middleband'] - \
                self.exit_bb_factor.value * (dataframe['bb_middleband'] - dataframe['bb_lowerband'])

            dataframe['bullish'] = np.where(
                (dataframe['close'] <= lower_limit)
                , 1, 0)

            # bullish region
            guard_conditions.append(dataframe["bullish"] > 0)

            # # not bearish (looser than bullish)
            # conditions.append(dataframe["bearish"] >= 0)

        if self.enable_squeeze.value:
            if not ('squeeze' in dataframe.columns):
                dataframe['squeeze'] = np.where(
                    (dataframe['bb_width'] >= self.entry_bb_width.value)
                    , 1, 0)

            guard_conditions.append(dataframe['squeeze'] > 0)


        # add coulmn that combines guard conditions (for plotting)
        if guard_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, guard_conditions), "buy_region"] = 1

            # model triggers
            model_cond = (
                # buy region
                (dataframe["buy_region"] > 0)
                &
                (
                    # prediction crossed target
                    qtpylib.crossed_above(dataframe["predicted_gain"], dataframe["target_profit"])
                    |
                    (
                        # add this version if volume checks are enabled, because we might miss the crossing otherwise
                        (dataframe["predicted_gain"] > dataframe["target_profit"]) &
                        (dataframe["predicted_gain"].shift() > dataframe["target_profit"].shift())
                    )
                )
            )
        else:
            # model triggers
            model_cond = (
                # prediction crossed target
                qtpylib.crossed_above(dataframe["predicted_gain"], dataframe["target_profit"])
            )

        # conditions.append(metric_cond)
        conditions.append(model_cond)

        # set entry tags
        dataframe.loc[model_cond, "enter_tag"] += "model_entry "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    ###################################

    """
    exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""
        dataframe["exit_long"] = 0
        dataframe['sell_region'] = 0

        if self.training_mode or (not self.enable_exit_signal.value):
            return dataframe


        dataframe['sell_region'] = 0
        guard_conditions = []


        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 0)

        if self.enable_guard_metric.value:

            # Guard metric in overbought region
            guard_conditions.append(dataframe["guard_metric"] > self.exit_guard_metric.value)

            # in upper portion of previous window
            # guard_conditions.append(dataframe["close"] > dataframe["local_mean"])

        if self.enable_bb_check.value:

            # Bollinger band-based bull/bear indicators:
            # Done here so that we can use hyperopt to find values

            upper_limit = dataframe['bb_middleband'] + \
            self.entry_bb_factor.value * (dataframe['bb_upperband'] - dataframe['bb_middleband'])

            dataframe['bearish'] = np.where(
                (dataframe['close'] >= upper_limit)
                , -1, 0)

            # bearish region
            guard_conditions.append(dataframe["bearish"] < 0)

            # # not bullish (looser than bearish)
            # conditions.append(dataframe["bullish"] <= 0)

        if self.enable_squeeze.value:
            if not ('squeeze' in dataframe.columns):
                dataframe['squeeze'] = np.where(
                    (dataframe['bb_width'] >= self.entry_bb_width.value)
                , 1, 0)

            guard_conditions.append(dataframe['squeeze'] > 0)


        if guard_conditions:
            # add column that combines guard conditions (for plotting)
            dataframe.loc[reduce(lambda x, y: x & y, guard_conditions), "sell_region"] = -1

            # model triggers
            model_cond = (

                # sell region
                (dataframe["sell_region"] < 0)
                &
                (
                    # prediction crossed target
                    qtpylib.crossed_below(dataframe["predicted_gain"], dataframe["target_loss"])
                    |
                    (
                        # add this if volume checks are enabled, because we might miss the crossing otherwise
                        (dataframe["predicted_gain"] < dataframe["target_loss"]) &
                        (dataframe["predicted_gain"].shift() < dataframe["target_loss"].shift())
                    )
                )
            )
        else:
            # model triggers
            model_cond = (
                # prediction crossed target
                qtpylib.crossed_below(dataframe["predicted_gain"], dataframe["target_loss"])
            )

        conditions.append(model_cond)

        # set exit tags
        dataframe.loc[model_cond, "exit_tag"] += "model_exit "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe

    ###################################

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        # this only makes sense in 'live' modes
        if self.dp.runmode.value in ("backtest", "plot", "hyperopt"):
            return True

        # in 'real' systems, there is often a delay between the signal and the trade
        # double-check that predicted gain is still above threshold

        if pair in self.custom_trade_info:
            curr_pred = self.custom_trade_info[pair]["curr_prediction"]

            # check latest prediction against latest target

            curr_target = self.custom_trade_info[pair]["curr_target"]
            if curr_pred < curr_target:
                if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
                    print("")
                    print(
                        f"    *** {pair} Trade cancelled. Prediction ({curr_pred:.2f}%) below target ({curr_target:.2f}%) "
                    )
                    print("")
                return False

        # just debug
        if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
            print("")
            print(f"    Trade Entry: {pair}, rate: {rate:.4f} Predicted gain: {curr_pred:.2f}% Target: {curr_target:.2f}%")
            print("")

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        
        if self.dp.runmode.value not in ("backtest", "plot", "hyperopt"):
            print("")
            print(f"    Trade Exit: {pair}, rate: {rate:.4f)}")
            print("")

        return True

    ###################################

    """
    Custom Stoploss
    """

    # simplified version of custom trailing stoploss
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        # this is just here so that we can use custom_exit

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss

    ###################################

    """
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    """

    # simplified version of custom exit

    def custom_exit(self, pair: str, trade: Trade, current_time: "datetime", current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if not self.use_custom_stoploss:
            return None

        # check volume?!
        if last_candle['volume'] <= 1.0:
            return None

        if trade.is_short:
            print("    short trades not yet supported in custom_exit()")

        else:

            # print("    checking long trade")

            # strong sell signal, in profit
            if (current_profit > 0.0) and (last_candle["guard_metric"] >= self.cexit_metric_overbought.value):
                return "metric_overbought"

            # Above 0.5%, sell if Fisher/Williams in sell range
            if current_profit > 0.005:
                if last_candle["guard_metric"] >= self.cexit_metric_take_profit.value:
                    return "take_profit"


            # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
            if (current_profit > 0) and (last_candle["predicted_gain"] <= last_candle["target_loss"]):
                return "predict_drop"

            # if in profit and exit signal is set, sell (even if exit signals are disabled)
            if (current_profit > 0) and (last_candle["exit_long"] > 0):
                return "exit_signal"


        # The following apply to both long & short trades:

        # Sell any positions if open for >= 1 day with any level of profit
        if ((current_time - trade.open_date_utc).days >= 1) & (current_profit > 0):
            return "unclog_1"

        # Sell any positions at a loss if they are held for more than 7 days.
        if (current_time - trade.open_date_utc).days >= 7:
            return "unclog_7"

        return None
