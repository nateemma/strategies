"""
####################################################################################
TS_Simple - base class for 'simple' time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             override the model-related functions

             This strategy uses only the 'base' indicators (open, close etc.) to estimate future gain.
             Note that I use gain rather than price because it is a normalised value, and works better with prediction algorithms.
             I use the actual (future) gain to train a base model, which is then further refined for each individual pair.
             The model is created if it does not exist, and is trained on all available data before being saved.
             Models are saved in user_data/strategies/TSPredict/models/<class>/<class>.sav, where <class> is the name of the current class
             (TS_Simple if running this directly, or the name of the subclass). 
             If the model already exits, then it is just loaded and used.
             So, it makes sense to do initial training over a long period of time to create the base model. 
             If training, then no backtesting or tuning for individual pairs is performed (way faster).
             If you want to retrain (e.g. you changed indicators), then delete the model and run the strategy over a long time period

####################################################################################
"""

#pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0325, W1203

from datetime import datetime
from functools import reduce

import cProfile
import pstats

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series


# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (IStrategy, DecimalParameter, CategoricalParameter)
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

import os
import joblib
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# # this adds  ../utils
# sys.path.append("../utils")

import logging
import warnings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from utils.DataframeUtils import DataframeUtils, ScalerType # pylint: disable=E0401
# import pywt
import talib.abstract as ta





class TS_Simple(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces


    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'}
        },
        'subplots': {
            "Diff": {
                'predicted_gain': {'color': 'red'},
                'future_gain': {'color': 'lightsalmon'},
                'fisher_wr': {'color': 'lightblue'}
            },
        }
    }


    # ROI table:
    minimal_roi = {
        "0": 0.06
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '15m'

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2
    win_size = 14

    process_only_new_candles = True

    custom_trade_info = {} # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # model_window = startup_candle_count
    model_window = 128

    lookahead = 6

    training_data = None
    training_labels = None
    training_mode = False # set to true if model does not already exist
    supports_incremental_training = True
    model_per_pair = False
    combine_models = True
    model_trained = False
    new_model = False

    norm_data = False

    dataframeUtils = None
    scaler = RobustScaler()
    model = None

    curr_dataframe: DataFrame = None

    # hyperparams
    # NOTE: this strategy does not hyperopt well, no idea why. Note that some vars are turned off (optimize=False)

    # the defaults are set for fairly frequent trades, and get out quickly
    # if you want bigger trades, then increase entry_model_gain, decrese exit_model_gain and adjust profit_threshold and
    # loss_threshold accordingly. 
    # Note that there is also a corellation to self.lookahead, but that cannot be a hyperopt parameter (because it is 
    # used in populate_indicators). Larger lookahead implies bigger differences between the model and actual price
    entry_model_gain = DecimalParameter(0.5, 3.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)
    exit_model_gain = DecimalParameter(-5.0, 0.0, decimals=1, default=-1.0, space='sell', load=True, optimize=True)

    # trailing stoploss
    tstop_start = DecimalParameter(0.0, 0.06, default=0.019, decimals=3, space='sell', load=True, optimize=True)
    tstop_ratio = DecimalParameter(0.7, 0.99, default=0.8, decimals=3, space='sell', load=True, optimize=True)

    # profit threshold exit
    profit_threshold = DecimalParameter(0.005, 0.065, default=0.06, decimals=3, space='sell', load=True, optimize=True)
    use_profit_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # loss threshold exit
    loss_threshold = DecimalParameter(-0.065, -0.005, default=-0.046, decimals=3, space='sell', load=True, optimize=True)
    use_loss_threshold = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # use exit signal? 
    enable_exit_signal = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    # enable entry/exit guards (safer vs profit)
    enable_guards = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)


    ###################################

    def bot_start(self, **kwargs) -> None:

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)

        return

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    ###################################

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair dataframe timeframe indicators
        curr_pair = metadata['pair']

        self.curr_dataframe = dataframe
        self.curr_pair = curr_pair

        # print("")
        # print(curr_pair)
        # print("")

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)


        # backward looking gain
        dataframe['gain'] = 100.0 * (dataframe['close'] - dataframe['close'].shift(self.lookahead)) / dataframe['close']
        dataframe['gain'].fillna(0.0, inplace=True)

        dataframe['gain'] = self.smooth(dataframe['gain'], 8) # smooth gain to make it easier to fit
        dataframe['gain'] = self.detrend_array(dataframe['gain'])


        # add other indicators
        # NOTE: does not work well with lots of indicators. Also, avoid any indicators that oscillate wildly or have
        #       discontinuities

        '''
        rsi = ta.RSI(dataframe, timeperiod=self.win_size)
        wr = 0.02 * (self.williams_r(dataframe, period=self.win_size) + 50.0)
        rsi2 = 0.1 * (rsi - 50)
        fisher_rsi =  (np.exp(2 * rsi2) - 1) / (np.exp(2 * rsi2) + 1)
        dataframe['fisher_wr'] = (wr + fisher_rsi) / 2.0

        '''

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.win_size)

        # Williams %R
        dataframe['wr'] = 0.02 * (self.williams_r(dataframe, period=self.win_size) + 50.0)

        # Fisher RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        '''
        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)


        # Keltner Channels (these can sometimes produce inf results)
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_mid"] = keltner["mid"]

        # Stochastic
        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi = (dataframe['rsi'] - dataframe['rsi'].rolling(period).min()) / (
                dataframe['rsi'].rolling(period).max() - dataframe['rsi'].rolling(period).min())
        dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()


        '''

        dataframe['model_gain'] = 0.0

        # create and init the model, if first time (dataframe has to be populated first)
        if self.model is None:
            self.load_model(np.shape(dataframe))

        # add the predictions
        # print("    Making predictions...")
        dataframe = self.add_predictions(dataframe)

        dataframe.fillna(0.0, inplace=True)
        
        return dataframe


    def detrend_array(self, a):
            if self.norm_data:
                # de-trend the data
                w_mean = a.mean()
                w_std = a.std()
                a_notrend = (a - w_mean) / w_std
                return np.nan_to_num(a_notrend)
            else:
                return np.nan_to_num(a)
    
    def smooth(self, y, window):
        box = np.ones(window)/window
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth = np.round(y_smooth, decimals=3) #Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
        return np.nan_to_num(y_smooth)

    def get_future_gain(self, dataframe):
        future_gain = 100.0 * (dataframe['close'].shift(-self.lookahead) - dataframe['close']) / dataframe['close']
        future_gain.fillna(0.0, inplace=True)
        future_gain = np.array(future_gain)
        future_gain = self.smooth(future_gain, 8)
        # print(f'future_gain:{future_gain}')
        return self.detrend_array(future_gain)
    
    ###################################

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


    ###################################


    #-------------

    def convert_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()
        # convert date column so that it can be scaled.
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], utc=True)
            df['date'] = dates.astype('int64')

        df.fillna(0.0, inplace=True)

        df.set_index('date')
        df.reindex()

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
        else:
            self.create_model(df_shape)
            self.model_trained = False
            self.new_model = True
            self.training_mode = True

        # sklearn family of regressors sometimes support starting with an existing model (warm_start), or incrementl training (partial_fit())
        if hasattr(self.model, 'warm_start'):
            self.model.warm_start = True
            self.supports_incremental_training = True # override default

        if hasattr(self.model, 'partial_fit'):
            self.supports_incremental_training = True # override default

        if self.model is None:
            print("***    ERR: model was not created properly ***")

        return

    #-------------

    def save_model(self):

        # save trained model (but only if didn't already exist)

        model_path = self.get_model_path("")

        # create directory if it doesn't already exist
        save_dir = os.path.dirname(model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # use joblib to save model state
        print("    saving to: ", model_path)
        joblib.dump(self.model, model_path)

        return

    #-------------

    def create_model(self, df_shape):
        # self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        # params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
        #           'learning_rate': 0.1, 'loss': 'squared_error'}
        # self.model = GradientBoostingRegressor(**params)
        params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}

        print("    creating XGBRegressor")
        self.model = XGBRegressor(**params)

        # LGBMRegressor gives better/faster results, but has issues on some MacOS platforms. Hence, not using it any more
        # self.model = LGBMRegressor(**params)

        if self.model is None:
            print("***    ERR: create_model() - model was not created ***")
        return

    #-------------

    # train the model. Override if not an sklearn-compatible algorithm
    # set save_model=False if you don't want to save the model (needed for ML algorithms)
    def train_model(self, model, data: np.array, results: np.array, save_model):

        if self.model is None:
            print("***    ERR: no model ***")
            return
        
        x = np.nan_to_num(data)

        # print('    Updating existing model')
        if isinstance(model, XGBRegressor):
            # print('    Updating xgb_model')
            if self.new_model and (not self.model_trained):
                model.fit(x, results)
            else:
                model.fit(x, results, xgb_model=self.model)
        elif hasattr(model, "partial_fit"):
            # print('    partial_fit()')
            model.partial_fit(x, results)
        else:
            # print('    fit()')
            model.fit(x, results)


        return


    #-------------

    # single prediction (for use in rolling calculation)
    def predict(self, df) -> float:

        data = np.array(self.convert_dataframe(df))

        x = np.nan_to_num(data)

        # y_pred = self.model.predict(data)[-1]
        y_pred = self.predict_data(x)[-1]

        return y_pred


    # initial training of the model
    def init_model(self, dataframe: DataFrame):
        # if not self.training_mode:
        #     # limit training to what we would see in a live environment, otherwise results are too good
        #     live_buffer_size = 974
        #     if dataframe.shape[0] > live_buffer_size:
        #         df = dataframe.iloc[-live_buffer_size:]
        #     else:
        #         df = dataframe
        # else:
        #     df = dataframe

        # if model is not yet trained, or this is a new model and we want to combine across pairs, then train
        if (not self.model_trained) or (self.new_model and self.combine_models):
            df = dataframe

            data = np.array(self.convert_dataframe(df))
            future_gain_data = self.get_future_gain(df)

            training_data = data[:-self.lookahead-1].copy()
            training_labels = future_gain_data[:-self.lookahead-1].copy()

            if not self.model_trained:
                print(f'    initial training ({self.curr_pair})')
            else:
                print(f'    inceremental training ({self.curr_pair})')

            self.train_model(self.model, training_data, training_labels, True)

            self.model_trained = True

            if self.new_model:
                self.save_model()
        
        # print(f'    model_trained:{self.model_trained} new_model:{self.new_model}  combine_models:{self.combine_models}')

        return
    
    # generate predictions for an np array (intended to be overriden if needed)
    def predict_data(self, data):
        x = np.nan_to_num(data)
        preds = self.model.predict(x)
        preds = np.clip(preds, -5.0, 5.0)
        return preds

    # get predictions for the entire dataframe
    def get_predictions(self, dataframe: DataFrame):

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        if dataframe.shape[0] > live_buffer_size:
            df = dataframe.iloc[-live_buffer_size:]
        else:
            df = dataframe

        model = self.custom_trade_info[self.curr_pair]

        data = np.array(self.convert_dataframe(df))
        future_gain_data = self.get_future_gain(df)
        # gain_data = df['gain'].to_numpy()

        self.training_data = data.copy()
        self.training_labels = future_gain_data.copy()

        if (not self.training_mode) and (self.supports_incremental_training):
            self.train_model(model, self.training_data, self.training_labels, False)

        # predictions = self.model.predict(data)
        predictions = self.predict_data(data)
        dataframe['predicted_gain'] = predictions
        
        return dataframe


    # add predictions in a rolling fashion. Use this when future data is present (e.g. backtest)
    def add_rolling_predictions(self, dataframe: DataFrame):

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974

        # roll through the close data and predict for each step
        nrows = np.shape(dataframe)[0]

        # build the coefficient table and merge into the dataframe (outside the main loop)

        data = np.array(self.convert_dataframe(dataframe))
        future_gain_data = self.get_future_gain(dataframe)
        gain_data = dataframe['gain'].to_numpy()

        # set up training data
        self.training_data = data.copy()
        self.training_labels = future_gain_data.copy()


        # initialise the prediction array, using the close data
        pred_array = gain_data.copy()


        # if training withion loop, we need to use a buffer size at least 2x the DWT/DWT window size + lookahead
        win_size = live_buffer_size
        # win_size = 512
        max_win_size = live_buffer_size 

        start = 0
        end = start + win_size
        first_time = True

        model = self.custom_trade_info[self.curr_pair]

        while end < nrows:

            start = end - win_size
            dslice = self.training_data[start:end].copy()
            cslice = self.training_labels[start:end].copy()

            # print(f"start:{start} end:{end} win_size:{win_size} dslice:{np.shape(dslice)}")

            if (not self.training_mode) and (self.supports_incremental_training):
                self.train_model(model, dslice, cslice, False)
            # preds = self.model.predict(dslice)
            preds = self.predict_data(dslice)

            if first_time:
                pred_array[:win_size] = preds.copy()
                first_time = False
            else:
                pred_array[end] = preds[-1]

            end = end + 1

        dataframe['predicted_gain'] = pred_array.copy()

        return dataframe



    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]

        data = np.array(self.convert_dataframe(dataframe))
        future_gain_data = self.get_future_gain(dataframe)
        gain_data = dataframe['gain'].to_numpy()

        # set up training data
        self.training_data = data.copy()
        self.training_labels = future_gain_data.copy()

        # initialise the prediction array, using the close data
        pred_array = np.zeros(np.shape(gain_data), dtype=float)

        # win_size = 974
        win_size = 128

        # loop until we get to/past the end of the buffer
        # start = 0
        # end = start + win_size
        start = win_size
        end = start + win_size
        train_start = 0

        model = self.custom_trade_info[self.curr_pair]

        while end < nrows:

            # extract the data and coefficients from the current window
            start = end - win_size
            dslice = self.training_data[start:end].copy()

            # get the training data. Use data prior to current prediction data, limited to live buffer size (so backtest resembles live modes)


            # if start > live_buffer_size:
            #     train_start = start - live_buffer_size
            # print(f'train_start: {train_start} start:{start}')

            train_data = self.training_data[train_start:start-1].copy()
            train_results = self.training_labels[train_start:start-1].copy()

            # print(f"start:{start} end:{end} train_start:{train_start}")
            # print(f'train_data: {train_data}')
            # print(f'train_results: {train_results}')

            # (re-)train the model on prior data and get predictions

            if (not self.training_mode) and (self.supports_incremental_training):
                self.train_model(model, train_data, train_results, False)

            preds = self.predict_data(dslice)

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds.copy()

            # move the window to the next segment
            end = end + win_size - 1

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data
        dslice = self.training_data[-(win_size+self.lookahead):-self.lookahead]
        cslice = self.training_labels[-(win_size+self.lookahead):-self.lookahead]

        if (not self.training_mode) and (self.supports_incremental_training):
            self.train_model(model, dslice, cslice, False)

        # predict for last window
        dslice = data[-win_size:].copy()
        # preds = self.model.predict(dslice)
        preds = self.predict_data(dslice)
        pred_array[-win_size:] = preds.copy()

        dataframe['predicted_gain'] = pred_array.copy()

        # add gain to dataframe for display purposes
        dataframe['future_gain'] = gain_data.copy()

        return dataframe

    
    # add predictions to dataframe['predicted_gain']
    def add_predictions(self, dataframe: DataFrame) -> DataFrame:

        # print(f"    {self.curr_pair} adding predictions")

        run_profiler = False

        if run_profiler:
            prof = cProfile.Profile()
            prof.enable()

        self.scaler = RobustScaler() # reset scaler each time

        self.init_model(dataframe)

        if self.curr_pair not in self.custom_trade_info:
            self.custom_trade_info[self.curr_pair] = None

        if self.custom_trade_info[self.curr_pair] is None:
            self.custom_trade_info[self.curr_pair] = self.model


        if self.training_mode:
            print(f'    Training mode. Skipping backtest for {self.curr_pair}')
            dataframe['predicted_gain'] = 0.0
        else:
            print(f'    backtesting {self.curr_pair}')
            dataframe = self.add_jumping_predictions(dataframe)
            # dataframe = self.add_rolling_predictions(dataframe)


        if run_profiler:
            prof.disable()
            # print profiling output
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(20) # top 20 rows

        return dataframe
    
    ###################################

    """
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
       

        if self.enable_guards.value:
            # Fisher/Williams in oversold region
            conditions.append(dataframe['fisher_wr'] < 0.0)

            # some trading volume
            conditions.append(dataframe['volume'] > 0)


        fwr_cond = (
            (dataframe['fisher_wr'] < -0.98)
        )

        # model triggers
        model_cond = (
            (
                # model predicts a rise above the entry threshold
                (dataframe['predicted_gain'] >= self.entry_model_gain.value) &

                # Fisher/Williams in oversold region
                (dataframe['fisher_wr'] < -0.5)
            )
            |
            # (
            #     # model predicts some gain
            #     (dataframe['predicted_gain'] > 0.0) &

            #     # strong oversold condition
            #     (dataframe['fisher_wr'] <= -0.99)
            # )
            # |
            (
                # large gain predicted (ignore fisher_wr)
                (dataframe['predicted_gain'] >= 2.0 * self.entry_model_gain.value) 
            )
        )
        

        # conditions.append(fwr_cond)
        conditions.append(model_cond)


        # set entry tags
        dataframe.loc[fwr_cond, 'enter_tag'] += 'fwr_entry '
        dataframe.loc[model_cond, 'enter_tag'] += 'model_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe


    '''
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        prev_candle = dataframe.iloc[-2].squeeze()

        # don't buy if the purchase price is above the current prediction (both can change)
        # pred = round(last_candle['predicted_gain'], 4)
        # price = round(rate, 4)

        pred = last_candle['predicted_gain']
        price = rate

        print("")
        if pred > price:
            # if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
            if self.dp.runmode.value not in ('plot', 'hyperopt'):
                print(f'Entry Signal: {pair}, Prediction:{pred:.4f} Price: {price:.4f}')
            result = True
        else:
            if self.dp.runmode.value not in ('hyperopt'):
                print(f"Entry rejected: {pair}. Prediction:{pred:.4f} <= Price:{price:.4f}")
            result = False

        # don't buy if sell signal active nearby (it can happen)
        if (last_candle['exit_long'] > 0) or (prev_candle['exit_long'] > 0):
            if self.dp.runmode.value not in ('hyperopt'):
                print(f"Entry rejected: {pair} sell active")
            result = False
        
        print("")

        return result
    
    '''

    ###################################

    """
    exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        if not self.enable_exit_signal.value:
            dataframe['exit_long'] = 0
            return dataframe

        # if self.enable_guards.value:

        if self.enable_guards.value:
            # Fisher/Williams in overbought region
            conditions.append(dataframe['fisher_wr'] > 0.0)

            # some trading volume
            conditions.append(dataframe['volume'] > 0)

        # model triggers
        model_cond = (
            (

                (dataframe['predicted_gain'] <= self.exit_model_gain.value) 
            )
        )

        conditions.append(model_cond)


        # set exit tags
        dataframe.loc[model_cond, 'exit_tag'] += 'model_exit '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1

        return dataframe


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
                
        if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
            print(f'Trade Exit: {pair}, rate: {round(rate, 4)}')

        return True
    
    ###################################

    """
    Custom Stoploss
    """

    # simplified version of custom trailing stoploss
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        # if current profit is above start value, then set stoploss at fraction of current profit
        if current_profit > self.tstop_start.value:
            return current_profit * self.tstop_ratio.value

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss


    ###################################

    """
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    """

    # simplified version of custom exit

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        

        if not self.use_custom_stoploss:
            return None


        # strong sell signal, in profit
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.96):
            return 'fwr_overbought'

        # Above 1%, sell if Fisher/Williams in sell range
        if current_profit > 0.01:
            if last_candle['fisher_wr'] > 0.8:
                return 'take_profit'
 

        # check profit against ROI target. This sort of emulates the freqtrade roi approach, but is much simpler
        if self.use_profit_threshold.value:
            if (current_profit >= self.profit_threshold.value):
                return 'profit_threshold'

        # check loss against threshold. This sort of emulates the freqtrade stoploss approach, but is much simpler
        if self.use_loss_threshold.value:
            if (current_profit <= self.loss_threshold.value):
                return 'loss_threshold'
              
        # Sell any positions if open for >= 1 day with any level of profit
        if ((current_time - trade.open_date_utc).days >= 1) & (current_profit > 0):
            return 'unclog_1'
        
        # Sell any positions at a loss if they are held for more than 7 days.
        if (current_time - trade.open_date_utc).days >= 7:
            return 'unclog_7'
        
        
        # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
        if (current_profit > 0) and (last_candle['predicted_gain'] <= self.exit_model_gain.value):
            return 'predict_drop'
        

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'

        return None
