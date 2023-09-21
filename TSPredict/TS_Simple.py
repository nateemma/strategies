#pragma pylint: disable=W0105, C0103, C0301, W1203

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

sys.path.append(str(Path(__file__).parent))

# this adds  ../utils
sys.path.append("../utils")

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from utils.DataframeUtils import DataframeUtils, ScalerType # pylint: disable=reportMissingImports
import pywt
import talib.abstract as ta

"""
####################################################################################
TS_Simple - base class for time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             overide the model-related functions

####################################################################################
"""


class TS_Simple(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces


    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
            # 'tema': {'color': 'lightsalmon'},
            # 'dwt': {'color': 'lightsalmon'},
            'model_predict': {'color': 'mediumaquamarine'},
        },
        'subplots': {
            "Diff": {
                'model_diff': {'color': 'brown'},
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

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables

    # model_window = startup_candle_count
    model_window = 128

    lookahead = 6

    training_data = None
    training_labels = None

    dataframeUtils = None
    scaler = RobustScaler()
    model = None

    curr_dataframe: DataFrame = None

    # hyperparams
    # NOTE: this strategy does not hyperopt well, no idea why. Note that some vars are turned off (optimize=False)

    # the defaults are set for fairly frequent trades, and get out quickly
    # if you want bigger trades, then increase entry_model_diff, decrese exit_model_diff and adjust profit_threshold and
    # loss_threshold accordingly. 
    # Note that there is also a corellation to self.lookahead, but that cannot be a hyperopt parameter (because it is 
    # used in populate_indicators). Larger lookahead implies bigger differences between the model and actual price
    entry_model_diff = DecimalParameter(0.5, 3.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)
    exit_model_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-1.0, space='sell', load=True, optimize=True)

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
    enable_guards = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=False)


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

        # print("")
        # print(curr_pair)
        # print("")

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)


        # # build the DWT
        # print("    Building DWT...")
        # dataframe['model_model'] = dataframe['close'].rolling(window=self.model_window).apply(self.model)

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

        # Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)


        dataframe['model_diff'] = 0.0

        # create and init the model, if first time (dataframe has to be populated first)
        if self.model is None:
            print("    creating model...")
            self.create_model()
            self.init_model(dataframe)

        # add the predictions
        # print("    Making predictions...")
        dataframe = self.add_predictions(dataframe)

        # % difference between prediction and curent close
        dataframe['model_diff'] = 100.0 * (dataframe['model_predict'] - dataframe['close']) / dataframe['close']

        
        return dataframe

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

    def create_model(self):
                # self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        # params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
        #           'learning_rate': 0.1, 'loss': 'squared_error'}
        # self.model = GradientBoostingRegressor(**params)
        params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}

        self.model = XGBRegressor(**params)

        # LGBMRegressor gives better/faster results, but has issues on some MacOS platforms. Hence, noy using it any more
        # self.model = LGBMRegressor(**params)
        return

    #-------------

    def train_model(self, data: np.array, train: np.array):

        # data = np.array(self.convert_dataframe(dataframe))

        # print(f"df: {np.shape(x)} y:{np.shape(y)}")


        self.model.fit(data, train)

        return


    #-------------

    def predict(self, df) -> float:

        data = np.array(self.convert_dataframe(df))
              
        y_pred = self.model.predict(data)[-1]

        return y_pred


    # initial training of the model
    def init_model(self, dataframe: DataFrame):

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        if dataframe.shape[0] > live_buffer_size:
            df = dataframe.iloc[-live_buffer_size:]
        else:
            df = dataframe

        close_data = np.array(df['close']).copy()
        data = np.array(self.convert_dataframe(df))

        training_data = data[:-self.lookahead]
        training_labels = close_data[self.lookahead:]
        self.train_model(training_data, training_labels)
        
        return 
    
    # get predictions for the entire dataframe
    def get_predictions(self, dataframe: DataFrame):

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        if dataframe.shape[0] > live_buffer_size:
            df = dataframe.iloc[-live_buffer_size:]
        else:
            df = dataframe

        close_data = np.array(df['close']).copy()
        data = np.array(self.convert_dataframe(df))

        self.training_data = data[:-self.lookahead]
        self.training_labels = close_data[self.lookahead:]
        self.train_model(self.training_data, self.training_labels)

        predictions = self.model.predict(data)
        dataframe['model_predict'] = predictions
        
        return dataframe


    # add predictions in a rolling fashion. Use this when future data is present (e.g. backtest)
    def add_rolling_predictions(self, dataframe: DataFrame):

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]

        # build the coefficient table and merge into the dataframe (outside the main loop)

        close_data = np.array(df['close']).copy()

        data = np.array(self.convert_dataframe(df)) # much faster using np.array vs DataFrame

        # set up training data
        self.training_data = data
        self.training_labels = close_data
        self.training_labels[:-self.lookahead] = close_data[self.lookahead:]

        # initialise the prediction array, using the close data
        pred_array = close_data


        # if training withion loop, we need to use a buffer size at least 2x the DWT/DWT window size + lookahead
        win_size = live_buffer_size
        max_win_size = live_buffer_size 

        start = 0
        end = start + win_size
        first_time = True

        while end < nrows:

            start = end - win_size
            dslice = self.training_data[start:end]
            cslice = self.training_labels[start:end]

            # print(f"start:{start} end:{end} win_size:{win_size} dslice:{np.shape(dslice)}")

            self.train_model(dslice, cslice)
            preds = self.model.predict(dslice)

            if first_time:
                pred_array[:win_size] = preds
                first_time = False
            else:
                pred_array[end] = preds[-1]

            end = end + 1

        dataframe['model_predict'] = pred_array

        return dataframe



    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, dataframe: DataFrame) -> DataFrame:

        # limit training to what we would see in a live environment, otherwise results are too good
        live_buffer_size = 974
        df = dataframe

        # roll through the close data and predict for each step
        nrows = np.shape(df)[0]


        # build the coefficient table and merge into the dataframe (outside the main loop)

        close_data = np.array(df['close']).copy()

        data = np.array(self.convert_dataframe(df)) # much faster using np.array vs DataFrame

        # set up training data
        self.training_data = data
        self.training_labels = close_data
        self.training_labels[:-self.lookahead] = close_data[self.lookahead:]

        # initialise the prediction array, using the close data
        pred_array = close_data

        # win_size = 974
        win_size = 128

        # loop until we get to/past the end of the buffer
        start = 0
        end = start + win_size

        while end < nrows:

            # extract the data and coefficients from the current window
            start = end - win_size
            dslice = self.training_data[start:end]
            cslice = self.training_labels[start:end]

            # print(f"start:{start} end:{end} win_size:{win_size} dslice:{np.shape(dslice)}")

            # (re-)train the model and get predictions
            self.train_model(dslice, cslice)
            preds = self.model.predict(dslice)

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds

            # move the window to the next segment
            end = end + win_size - 1

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        # Note that we cannot use the last section for training because we don't have forward looking data
        dslice = self.training_data[-(win_size+self.lookahead):-self.lookahead]
        cslice = self.training_labels[-(win_size+self.lookahead):-self.lookahead]
        self.train_model(dslice, cslice)

        # predict for last window
        dslice = data[-win_size:]
        preds = self.model.predict(dslice)
        pred_array[-win_size:] = preds

        dataframe['model_predict'] = pred_array

        # #DBG:
        # dataframe['coeff_2'] = df_merged['coeff_2']

        return dataframe

    
    # add predictions to dataframe['model_predict']
    def add_predictions(self, dataframe: DataFrame) -> DataFrame:

        run_profiler = False

        if run_profiler:
            prof = cProfile.Profile()
            prof.enable()

        self.scaler = RobustScaler() # reset scaler each time

        dataframe['model_predict'] = dataframe['close']

        # for hyperopt, dryrun and plot modes, we need to add/replace data using a rolling/jumping window
        if self.dp.runmode.value in ('hyperopt' 'backtest' 'plot'):
            # dataframe = self.add_rolling_predictions(dataframe)
            dataframe = self.add_jumping_predictions(dataframe)
        # else:
        #     # for other modes, we can just directly process the data (because it is not forward looking)
        #     dataframe = self.get_predictions(dataframe)

        # #DBG:
        # dataframe = self.get_predictions(dataframe)

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
       
        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        if self.enable_guards.value:

            # Fisher/Williams in buy region
            conditions.append(dataframe['fisher_wr'] <= -0.5)
        else:
            conditions.append(dataframe['fisher_wr'] < 0.0)

        # model triggers
        model_cond = (
            # model predicts a rise above the entry threshold
            (dataframe['model_diff'] >= self.entry_model_diff.value) #&

            # model prediction is going up
            # (dataframe['model_predict'] > dataframe['model_predict'].shift())   
            )

        conditions.append(model_cond)

        # # DWTs will spike on big gains, so try to constrain
        # spike_cond = (
        #         dataframe['model_diff'] < 2.0 * self.entry_model_diff.value
        # )
        # conditions.append(spike_cond)

        # set entry tags
        dataframe.loc[model_cond, 'enter_tag'] += 'model_entry '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # don't buy if the purchase price is above the current prediction (both can change)
        # pred = round(last_candle['model_predict'], 4)
        # price = round(rate, 4)

        pred = last_candle['model_predict']
        price = rate

        if pred > price:
            if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
                print(f'Entry: {pair:.4f}, rate: {price:.4f}')
            result = True
        else:
            if self.dp.runmode.value not in ('hyperopt'):
                print(f"Entry rejected: {pair}. Prediction:{pred:.4f} <= rate:{price:.4f}")
            result = False

        # don't buy if sell signal active (it can happen)
        if last_candle['exit_long'] > 0:
            if self.dp.runmode.value not in ('hyperopt'):
                print(f"Entry rejected: sell active")
            result = False
            
        return result
    
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

        if self.enable_guards.value:
 
            # some volume
            conditions.append(dataframe['volume'] > 0)

             # Fisher/Williams in sell region
            conditions.append(dataframe['fisher_wr'] >= 0.5)
        else:
            conditions.append(dataframe['fisher_wr'] > 0.0)

        # model triggers
        model_cond = (
            (dataframe['model_diff'] <= self.exit_model_diff.value) 
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
        if (current_profit > 0) and (last_candle['fisher_wr'] > 0.95):
            return 'fwr_overbought'

        # Above 1%, sell if Fisher/Williams in sell range
        if current_profit > 0.01:
            if last_candle['fisher_wr'] > 0.8:
                return 'take_profit'
                  
        # if model prediction is above threshold, don't exit
        if last_candle['model_diff'] >= self.entry_model_diff.value:
            return None

        # check profit against ROI target. This sort of emulates the freqtrade roi approach, but is much simpler
        if self.use_profit_threshold.value:
            if (current_profit >= self.profit_threshold.value):
                return 'profit_threshold'

        # check loss against threshold. This sort of emulates the freqtrade stoploss approach, but is much simpler
        if self.use_loss_threshold.value:
            if (current_profit <= self.loss_threshold.value):
                return 'loss_threshold'
              
        # Sell any positions at a loss if they are held for more than 'N' days.
        if (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'
        
        # big drop predicted. Should also trigger an exit signal, but this might be quicker (and will likely be 'market' sell)
        if (current_profit > 0) and (last_candle['model_diff'] <= self.exit_model_diff.value):
            return 'predict_drop'
        

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'

        return None
