#pragma pylint: disable=W0105, C0103, C0301, W1203

from datetime import datetime
from functools import reduce
# import timeit

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series
import scipy


# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (IStrategy, DecimalParameter, CategoricalParameter)
from freqtrade.persistence import Trade

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from DataframeUtils import DataframeUtils, ScalerType
import pywt
import talib.abstract as ta

'''

####################################################################################
DWT_Predict2 - use a Discreet Wavelet Transform to model the price, and a
              regression algorithm trained on the DWT coefficients, which is then used
              to predict future prices.
              Unfortunately, this must all be done in a rolling fashion to avoid lookahead
              bias - so it is pretty slow

              This variant uses the just the DWT coefficients, ot the whole dataframe

####################################################################################
'''


class DWT_Predict2(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {
        "0": 0.04, "100": 0.02
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

    model_window = startup_candle_count

    lookahead = 6

    # df_coeffs: DataFrame = None
    coeff_array = None
    coeff_model = None
    dataframeUtils = None
    scaler = RobustScaler()

    # DWT  hyperparams
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


    plot_config = {
        'main_plot': {
            'close': {'color': 'cornflowerblue'},
            # 'model_model': {'color': 'lightsalmon'},
            'model_predict': {'color': 'mediumaquamarine'},
        },
        'subplots': {
            "Diff": {
                'model_diff': {'color': 'brown'},
            },
        }
    }

    ###################################

    def bot_start(self, **kwargs) -> None:

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)

        if self.coeff_model is None:
            self.create_model()

        return

    ###################################

    '''
    Informative Pair Definitions
    '''

    def informative_pairs(self):
        return []

    ###################################

    '''
    Indicator Definitions
    '''

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Base pair dataframe timeframe indicators
        curr_pair = metadata['pair']

        # print("")
        # print(curr_pair)
        # print("")


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

        # add predictions
        dataframe = self.add_predictions(dataframe)
            
        # calculation % diff between model and actual closing price
        dataframe['model_diff'] = 100.0 * (dataframe['model_predict'] - dataframe['close']) / dataframe['close']

        return dataframe

    ###################################

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 14) -> Series:
        '''
        Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
            of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
            Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
            of its recent trading range.
            The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        '''

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100


    ###################################

 
    def madev(self, d, axis=None):
        ''' Mean absolute deviation of a signal'''
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    ###################################
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        # print(pywt.wavelist(kind='discrete'))

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # get the DWT coefficients
        wavelet = 'haar'
         # wavelet = 'db12'
        # wavelet = 'db4'
        coeffs = pywt.wavedec(x, wavelet, mode='smooth')

        # # remove higher harmonics
        # level = 2
        # length = len(data)
        # sigma = (1 / 0.6745) * self.madev(coeffs[-level])
        # uthresh = sigma * np.sqrt(2 * np.log(length))
        # coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])

        # flatten the coefficient arrays
        features = np.concatenate(np.array(coeffs, dtype=object))

        # # trim down to max 128 entries
        # if len(features) > 128:
        #     features = features[:128]

        return features
    

    # builds a numpy array of coefficients
    def add_coefficients(self, dataframe) -> DataFrame:

        # normalise the dataframe
        df_norm = self.convert_dataframe(dataframe)

        # copy the close data into an np.array (faster)
        close_data = np.array(df_norm['close'])

        init_done = False
        
        # roll through the close data and create DWT coefficients for each step
        nrows = np.shape(close_data)[0]

        start = 0
        end = start + self.model_window
        dest = end

        # print(f"nrows:{nrows} start:{start} end:{end} dest:{dest} nbuffs:{nbuffs}")

        self.coeff_array = None
        num_coeffs = 0

        while end < nrows:
            dslice = close_data[start:end]

            # print(f"start:{start} end:{end} dest:{dest} len:{len(dslice)}")

            features = self.get_coeffs(dslice)
            
            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_array = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_array:{np.shape(self.coeff_array)}")

            # copy the features to the appropriate row of the coefficient array (offset due to startup window)
            self.coeff_array[dest] = features

            start = start + 1
            dest = dest + 1
            end = end + 1

        # # normalise the coefficients
        # self.scaler.fit(self.coeff_array)
        # self.coeff_array = self.scaler.transform(self.coeff_array)

        return dataframe
    


    # builds a numpy array of coefficients
    def get_coefficient_table(self, data: np.array):

        init_done = False
        
        # roll through the  data and create DWT coefficients for each step
        nrows = np.shape(data)[0]

        # print(f'nrows:{nrows}')

        start = 0
        end = start + self.model_window - 1
        dest = end

        # print(f"nrows:{nrows} start:{start} end:{end} dest:{dest} nbuffs:{nbuffs}")

        coeff_table = None
        num_coeffs = 0

        while end < nrows:
            dslice = data[start:end]

            # print(f"start:{start} end:{end} dest:{dest} len:{len(dslice)}")

            features = self.get_coeffs(dslice)
            
            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                coeff_table = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_table:{np.shape(coeff_table)}")

            # copy the features to the appropriate row of the coefficient array (offset due to startup window)
            coeff_table[dest] = features

            start = start + 1
            dest = dest + 1
            end = end + 1

        return coeff_table


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

    #-------------

    def create_model(self):
                # self.coeff_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        # params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
        #           'learning_rate': 0.1, 'loss': 'squared_error'}
        # self.coeff_model = GradientBoostingRegressor(**params)
        params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1}

        self.coeff_model = XGBRegressor(**params)

        # LGBMRegressor gives better/faster results, but has issues on some MacOS platforms. Hence, not using it any more
        # self.coeff_model = LGBMRegressor(**params)
        return

    #-------------

    def train_model(self, data: np.array, coeff_table: np.array):

        #TODO: do grid search for best parameters first time through (or just during debug)

        # print(f"data: {np.shape(data)} coeff_table:{np.shape(coeff_table)}")

        # need to exclude the startup period at the front, and the lookahead period at the end

        # x = coeff_table[self.startup_candle_count:-self.lookahead]
        # y = data[self.startup_candle_count+self.lookahead:]
        x = coeff_table[:-self.lookahead]
        y = data[self.lookahead:]

        # print(f"df: {np.shape(x)} y:{np.shape(y)}")

        self.coeff_model.fit(x, y)

        return


    #-------------

    def get_modal_value(self, data):
        # round to 6 decimal places (otherwise there is unlikely to be a modal value)
        rnd = np.round(data, decimals=6)

        # get the mode
        mode = scipy.stats.mode(rnd, keepdims=False).mode

        return mode


    # get predictions based on array of close data
    def get_predictions(self, data: np.array):
        coeff_table = self.get_coefficient_table(data)
        self.train_model(data, coeff_table)
        predictions = self.coeff_model.predict(coeff_table)
        return predictions


    # add predictions in a non-rolling fashion. Use this when future data is *not* present (e.g. dry run)
    def add_predictions(self, dataframe: DataFrame) -> DataFrame:

        dataframe['model_predict'] = dataframe['close']

        # normalise the dataframe
        # df_norm = self.convert_dataframe(dataframe)
        close_data = np.array(dataframe['close'])

        # for hyperopt, dryrun and plot modes, we need to add data using a rolling window
        if self.dp.runmode.value in ('hyperopt' 'backtest' 'plot'):
            # dataframe['model_predict'] = self.add_rolling_predictions(close_data)
            dataframe['model_predict'] = self.add_jumping_predictions(close_data)
        else:
            # for other modes, we can just directly process the data (because it is not forward looking)
            dataframe['model_predict'] = self.get_predictions(close_data)
       
        return dataframe


    # add predictions in a rolling fashion. Use this when future data is present (e.g. backtest)
    def add_rolling_predictions(self, data: np.array) -> DataFrame:

        use_expanding_window = False

        # add data in a rolling fashion.
        # Note that this is not quite the same as the usual (pandas) rolling window, since we can use an expanding window
        # that includes any data that lies in the 'past'.
        # Also, I restrict the window size to 974, which is the size of the data buffer in dry run and live run modes

        live_buffer_size = 974

        # roll through the close data and predict for each step
        nrows = np.shape(data)[0]

        # initialise the prediction array, using the close data
        pred_array = data.copy()
        
        # build the coefficient table (outside the main loop)
        coeff_table = self.get_coefficient_table(data)

        # win_size = self.model_window
        # if (nrows > live_buffer_size):
        #     max_win_size = live_buffer_size 
        # else:
        #     max_win_size = self.model_window

        # if training withion loop, we need to use a buffer size at least 2x the DWT/DWT window size + lookahead
        win_size = live_buffer_size
        max_win_size = live_buffer_size 

        start = 0
        end = start + win_size
        first_time = True

        while end < nrows:

            start = end - win_size
            dslice = data[start:end]
            cslice = coeff_table[start:end]

            # print(f"start:{start} end:{end} win_size:{win_size} dslice:{np.shape(dslice)}")

            self.train_model(dslice, cslice)
            preds = self.coeff_model.predict(cslice)

            if first_time:
                pred_array[:win_size] = preds
                first_time = False
            else:
                pred_array[end] = preds[-1]

            end = end + 1
            if use_expanding_window:
                win_size = min(win_size+1, max_win_size)

        return pred_array


    # add predictions in a jumping fashion. This is a compromise - the rolling version is very slow
    # Note: you probably need to manually tune the parameters, since there is some limited lookahead here
    def add_jumping_predictions(self, data: np.array) -> DataFrame:

        use_expanding_window = False

        # roll through the close data and predict for each step
        nrows = np.shape(data)[0]

        # initialise the prediction array, using the close data
        pred_array = data.copy()
        
        # build the coefficient table (outside the main loop)
        coeff_table = self.get_coefficient_table(data)

        # if training within loop, we need to use a buffer size at least 2x the DWT/DWT window size + lookahead
        # here, just using same size as the live run buffer
        win_size = 974

        # loop until we get to/past the end of the buffer
        start = 0
        end = start + win_size

        while end < nrows:

            # extract the data and coefficients from the current window
            start = end - win_size
            dslice = data[start:end]
            cslice = coeff_table[start:end]

            # print(f"start:{start} end:{end} win_size:{win_size} dslice:{np.shape(dslice)}")

            # (re-)train the model and get predictions
            self.train_model(dslice, cslice)
            preds = self.coeff_model.predict(cslice)

            # copy the predictions for this window into the main predictions array
            pred_array[start:end] = preds

            # move the window to the next segment
            end = end + win_size - 1

        # make sure the last section gets processed (the loop above may not exactly fit the data)
        dslice = data[-win_size:]
        cslice = coeff_table[-win_size:]
        self.train_model(dslice, cslice)
        preds = self.coeff_model.predict(cslice)
        pred_array[-win_size:] = preds

        return pred_array

    ###################################

    '''
    entry Signal
    '''

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
       
        # some trading volume
        conditions.append(dataframe['volume'] > 0)

        if self.enable_guards.value:

            # Fisher/Williams in buy region
            conditions.append(dataframe['fisher_wr'] <= -0.5)
        else:
            conditions.append(dataframe['fisher_wr'] < 0.0) # very loose guard

        # Model triggers
        model_cond = (
            # model predicts a rise above the entry threshold
            (dataframe['model_diff'] >= self.entry_model_diff.value) &

            # model prediction is going up
            (dataframe['model_predict'] >= dataframe['model_predict'].shift())

        )

        conditions.append(model_cond)

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
        pred = round(last_candle['model_predict'], 4)
        price = round(rate, 4)
        if pred > price:
            if self.dp.runmode.value not in ('backtest', 'plot', 'hyperopt'):
                print(f'Entry: {pair}, rate: {price}')
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

    '''
    exit Signal
    '''

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
            conditions.append(dataframe['fisher_wr'] > 0.0) # very loose guard

        # Model triggers
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

    '''
    Custom Stoploss
    '''

    # simplified version of custom trailing stoploss
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        # if current profit is above start value, then set stoploss at fraction of current profit
        if current_profit > self.tstop_start.value:
            return current_profit * self.tstop_ratio.value

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss


    ###################################

    '''
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    '''

    # simplified version of custom exit

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
                  
        if not self.use_custom_stoploss:
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()


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
        if last_candle['model_diff'] <= min(-1.0, self.exit_model_diff.value):
            return 'predict_drop'
        

        # if in profit and exit signal is set, sell (even if exit signals are disabled)
        if (current_profit > 0) and (last_candle['exit_long'] > 0):
            return 'exit_signal'

        return None