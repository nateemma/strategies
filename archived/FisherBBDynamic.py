
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from cachetools import TTLCache
from pandas import DataFrame, Series
import numpy as np

## Indicator libs
import talib.abstract as ta
from finta import TA as fta
import technical.indicators as ftt
from technical.indicators import hull_moving_average
from technical.indicators import PMAX, zema
from technical.indicators import cmf

## FT stuffs
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open, IntParameter, DecimalParameter, CategoricalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from skopt.space import Dimension


"""
NOTE:
docker-compose run --rm freqtrade hyperopt -c user_data/config-backtesting.json --strategy IchimokuHaulingV8a --hyperopt-loss SortinoHyperOptLossDaily --spaces roi buy sell --timerange=1624940400-1630447200 -j 4 -e 1000
"""
class FisherBBDynamic(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_gain": 0.01,
        "buy_enable_signal_fisher_bb": True,
        "buy_enable_signal_fisher_hull": True,
        "buy_fisher": 0.9,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_cstp_bail_how": "any",
        "sell_cstp_bail_roc": -0.047,
        "sell_cstp_bail_time": 780,
        "sell_cstp_bb_trailing_input": "bb_lowerband_trend_inf",
        "sell_cstp_threshold": -0.016,
        "sell_cstp_trailing_max_stoploss": 0.062,
        "sell_cstp_trailing_only_offset_is_reached": 0.035,
        "sell_cstp_trailing_stop_profit_devider": 2,
        "sell_droi_pullback": True,
        "sell_droi_pullback_amount": 0.005,
        "sell_droi_pullback_respect_table": False,
        "sell_droi_trend_type": "ssl",
    }


    # ROI table: 
    minimal_roi = {
        "0": 0.282,
        "37": 0.082,
        "94": 0.04,
        "208": 0
    }

    # Stoploss:
    stoploss = -0.028 

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.181
    trailing_stop_positive_offset = 0.202
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True

    # buy signal
    buy_enable_signal_fisher_hull = CategoricalParameter([True, False], default=buy_params['buy_enable_signal_fisher_hull'], space='buy', optimize=True)

    buy_enable_signal_fisher_bb = CategoricalParameter([True, False], default=buy_params['buy_enable_signal_fisher_bb'], space='buy', optimize=True)
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.10, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=0.02, space="buy")

    # Dynamic ROI
    sell_droi_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any'], default=sell_params['sell_droi_trend_type'], space='sell', optimize=True)
    sell_droi_pullback = CategoricalParameter([True, False], default=sell_params['sell_droi_pullback'], space='sell', optimize=True)
    sell_droi_pullback_amount = DecimalParameter(0.005, 0.02, default=sell_params['sell_droi_pullback_amount'], space='sell')
    sell_droi_pullback_respect_table = CategoricalParameter([True, False], default=sell_params['sell_droi_pullback_respect_table'], space='sell', optimize=True)

    # Custom Stoploss
    sell_cstp_threshold = DecimalParameter(-0.05, 0, default=sell_params['sell_cstp_threshold'], space='sell')
    sell_cstp_bail_how = CategoricalParameter(['roc', 'time', 'any'], default=sell_params['sell_cstp_bail_how'], space='sell', optimize=True)
    sell_cstp_bail_roc = DecimalParameter(-0.05, -0.01, default=sell_params['sell_cstp_bail_roc'], space='sell')
    sell_cstp_bail_time = IntParameter(720, 1440, default=sell_params['sell_cstp_bail_time'], space='sell')
    sell_cstp_trailing_only_offset_is_reached = DecimalParameter(0.01, 0.06, default=sell_params['sell_cstp_trailing_only_offset_is_reached'], space='sell')
    sell_cstp_trailing_stop_profit_devider = IntParameter(2, 4, default=sell_params['sell_cstp_trailing_stop_profit_devider'], space='sell')
    sell_cstp_trailing_max_stoploss = DecimalParameter(0.02, 0.08, default=sell_params['sell_cstp_trailing_max_stoploss'], space='sell')
    # sell_cstp_bb_trailing_input = CategoricalParameter(['bb_lowerband_trend', 'bb_lowerband_trend_inf', 'bb_lowerband_neutral', 'bb_lowerband_neutral_inf', 'bb_upperband_neutral_inf'], default=sell_params['sell_cstp_bb_trailing_input'], space='sell', optimize=True)
    sell_cstp_bb_trailing_input = CategoricalParameter(['bb_lowerband_trend_inf', 'bb_lowerband_neutral_inf', 'bb_upperband_neutral_inf'], default=sell_params['sell_cstp_bb_trailing_input'], space='sell', optimize=True)

    # nested hyperopt class
    class HyperOpt:

        # defining as dummy, so that no error is thrown about missing
        # sell indicator space when hyperopting for all spaces
        @staticmethod
        def indicator_space() -> List[Dimension]:
            return []

    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes
    custom_current_low_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count = 149

    use_dynamic_roi = True

    # timeframe = '15m'
    timeframe = '5m'
    informative_timeframe = '1h'

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    plot_config = {
        'main_plot': {
            'chikou_span_inf': {'color': 'green'},
            'tenkan_sen_inf': {'color': 'blue'},
            'kijun_sen_inf': {'color': 'red'},
            'senkou_a_inf': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Kumo',
                'fill_color': 'rgba(51, 255, 117, 0.2)',
            },
            'senkou_b_inf': {'color': 'red'},
            'leading_senkou_span_a_inf': {'color': 'green'},
            'leading_senkou_span_b_inf': {'color': 'red'},
            'sslUp_inf': {'color': 'green'},
            'sslDown_inf': {'color': 'red'}
        },
        'subplots': {
            'summary': {
                'cloud_green_inf': {},
                'cloud_red_inf': {},
                'future_green_inf': {},
                'chikou_high_inf': {},
                'go_long_inf': {}
            }
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs


    #
    # Processing indicators
    #

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])

        if not self.dp:
            return dataframe

        dataframe = self.get_buy_signal_indicators(dataframe, metadata)

        informative_tmp = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        informative = self.get_market_condition_indicators(informative_tmp.copy(), metadata)
        informative = self.get_custom_stoploss_indicators(informative, metadata)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

        dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "_inf"), inplace=True)

        # Slam some indicators into the trade_info dict so we can dynamic roi and custom stoploss in backtest
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['roc_inf'] = dataframe[['date', 'roc_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['atr_inf'] = dataframe[['date', 'atr_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['sroc_inf'] = dataframe[['date', 'sroc_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['ssl-dir_inf'] = dataframe[['date', 'ssl-dir_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['rmi-up-trend_inf'] = dataframe[['date', 'rmi-up-trend_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['candle-up-trend_inf'] = dataframe[['date', 'candle-up-trend_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['bb_lowerband_trend_inf'] = dataframe[['date', 'bb_lowerband_trend_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['bb_lowerband_trend_inf'] = dataframe[['date', 'bb_lowerband_trend_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['bb_lowerband_neutral_inf'] = dataframe[['date', 'bb_lowerband_neutral_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['bb_lowerband_neutral_inf'] = dataframe[['date', 'bb_lowerband_neutral_inf']].copy().set_index('date')
            self.custom_trade_info[metadata['pair']]['bb_upperband_neutral_inf'] = dataframe[['date', 'bb_upperband_neutral_inf']].copy().set_index('date')

        return dataframe

    def get_buy_signal_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # # get_buy_signal_awesome_macd
        # dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        #
        # # get_buy_signal_adx_momentum
        # #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        # dataframe['sar'] = ta.SAR(dataframe)
        # dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)
        #
        # # get_buy_signal_adx_smas
        # #dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        # dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['long'] = ta.SMA(dataframe, timeperiod=6)
        #
        # # get_buy_signal_asdts_rockwelltrading
        # #macd = ta.MACD(dataframe)
        # #dataframe['macd'] = macd['macd']
        # #dataframe['macdsignal'] = macd['macdsignal']
        # #dataframe['macdhist'] = macd['macdhist']
        #
        # # get_buy_signal_averages_strategy
        # dataframe['maShort'] = ta.EMA(dataframe, timeperiod=8)
        # dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=21)
        #

        # get_buy_signal_fisher_bb
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # get_buy_signal_fisher_hull
        dataframe['hma'] = hull_moving_average(dataframe, 14, 'close')
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        #
        # # get_buy_signal_gettin_moist
        # dataframe['color'] = dataframe['close'] > dataframe['open']
        # #macd = ta.MACD(dataframe)
        # #dataframe['macd'] = macd['macd']
        # #dataframe['macdsignal'] = macd['macdsignal']
        # #dataframe['macdhist'] = macd['macdhist']
        # dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        # dataframe['roc_6'] = ta.ROC(dataframe, timeperiod=6)
        # dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3,1,0)
        # dataframe['in-the-mood'] = dataframe['rsi_7'] > dataframe['rsi_7'].rolling(12).mean()
        # dataframe['moist'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        # dataframe['throbbing'] = dataframe['roc_6'] > dataframe['roc_6'].rolling(12).mean()
        # dataframe['ready-to-go'] = np.where(dataframe['close'] > dataframe['open'].rolling(12).mean(), 1,0)
        #
        # # get_buy_signal_hlhb
        # dataframe['hl2'] = (dataframe["close"] + dataframe["open"]) / 2
        # dataframe['rsi_10'] = ta.RSI(dataframe, timeperiod=10, price='hl2')
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # #dataframe['adx'] = ta.ADX(dataframe)

        # get_buy_signal_macd_strategy_crossed
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        #dataframe['cci'] = ta.CCI(dataframe)

        # get_buy_signal_macd_strategy
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        #dataframe['cci'] = ta.CCI(dataframe)

        # # get_buy_signal_pmax
        # dataframe['ZLEMA'] = zema(dataframe, period=10)
        # dataframe = PMAX(dataframe, period=10, multiplier=3, length=10, MAtype=9, src=2)
        #
        # # get_buy_signal_quickie
        # #macd = ta.MACD(dataframe)
        # #dataframe['macd'] = macd['macd']
        # #dataframe['macdsignal'] = macd['macdsignal']
        # #dataframe['macdhist'] = macd['macdhist']
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        # dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        # dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=200)
        # #dataframe['adx'] = ta.ADX(dataframe)
        # bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']
        #
        # # get_buy_signal_scalp
        # dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        # dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        # dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        # stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']
        # #dataframe['adx'] = ta.ADX(dataframe)

        # get_buy_signal_simple
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']
        #dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        #bollinger = qtpylib.bollinger_bands(dataframe['close'], window=12, stds=2)
        #dataframe['bb_lowerband'] = bollinger['lower']
        #dataframe['bb_upperband'] = bollinger['upper']
        #dataframe['bb_middleband'] = bollinger['mid']

        # # get_buy_signal_strategy001
        # dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        #
        # # get_buy_signal_technical_example_strategy
        # dataframe['cmf'] = cmf(dataframe, 21)

        # get_buy_signal_tema_rsi_strategy
        #dataframe['rsi'] = ta.RSI(dataframe)
        #bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        #dataframe['bb_lowerband'] = bollinger['lower']
        #dataframe['bb_middleband'] = bollinger['mid']
        #dataframe['bb_upperband'] = bollinger['upper']
        #dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def get_market_condition_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=displacement)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        ssl = SSLChannels_ATR(dataframe, 10)
        dataframe['sslDown'] = ssl[0]
        dataframe['sslUp'] = ssl[1]

        dataframe['vfi'] = fta.VFI(dataframe, period=14)

        # Summary indicators
        dataframe['future_green'] = ichimoku['cloud_green'].shift(displacement).fillna(0).astype('int') * 2
        dataframe['chikou_high'] = ((dataframe['chikou_span'] > dataframe['senkou_a']) & (dataframe['chikou_span'] > dataframe['senkou_b'])).shift(displacement).fillna(0).astype('int')
        dataframe['go_long'] = ((dataframe['tenkan_sen'] > dataframe['kijun_sen']) & (dataframe['close'] > dataframe['leading_senkou_span_a']) & (dataframe['close'] > dataframe['leading_senkou_span_b']) & (dataframe['future_green'] > 0) & (dataframe['chikou_high'] > 0)).fillna(0).astype('int') * 3
        dataframe['max'] = dataframe['high'].rolling(3).max()
        dataframe['min'] = dataframe['low'].rolling(6).min()
        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(),1,0)
        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(),1,0)
        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(5, min_periods=1).sum() != 0,1,0)
        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(5, min_periods=1).sum() != 0,1,0)

        return dataframe

    def get_custom_stoploss_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger_neutral = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband_neutral'] = bollinger_neutral['lower']
        dataframe['bb_middleband_neutral'] = bollinger_neutral['mid']
        dataframe['bb_upperband_neutral'] = bollinger_neutral['upper']
        bollinger_trend = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband_trend'] = bollinger_trend['lower']
        dataframe['bb_middleband_trend'] = bollinger_trend['mid']
        dataframe['bb_upperband_trend'] = bollinger_trend['upper']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=9)
        dataframe['rmi'] = RMI(dataframe, length=24, mom=5)
        ssldown, sslup = SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(),1,0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3,1,0)
        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['close'].shift(),1,0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3,1,0)

        return dataframe


    #
    # Processing buy signals
    #

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (self.get_buy_signal_fisher_bb(dataframe) == True) |
                (self.get_buy_signal_fisher_hull(dataframe) == True)
            ) &
            (dataframe['sslUp_inf'] > dataframe['sslDown_inf']) &
            (dataframe['up_trend_inf'] > 0) &
            (dataframe['go_long_inf'] > 0) &
            # NOTE: I keep the volume checks of feels like it has not much benifit when trading leverage tokens,
            # maybe im wrong!?
            # (dataframe['vfi'] < 0.0) &
            (dataframe['volume'] > 0)
        ,'buy'] = 1

        return dataframe

    def get_buy_signal_fisher_bb(self, dataframe: DataFrame):
        signal = (
            (self.buy_enable_signal_fisher_bb.value == True) &
            (dataframe['fisher_rsi'] <= self.buy_fisher.value) &
            (dataframe['bb_gain'] >= self.buy_bb_gain.value)
        )
        return signal


    def get_buy_signal_fisher_hull(self, dataframe: DataFrame):
        signal = (
            (self.buy_enable_signal_fisher_hull.value == True) &
            (dataframe['hma'] < dataframe['hma'].shift()) &
            (dataframe['cci'] <= -50.0) &
            (dataframe['fisher_rsi'] < -0.5)
        )
        return signal


    #
    # Processing sell signals
    #

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['sslDown_inf'], dataframe['sslUp_inf']))
            & (
                (qtpylib.crossed_below(dataframe['tenkan_sen_inf'], dataframe['kijun_sen_inf']))
                |(qtpylib.crossed_below(dataframe['close_inf'], dataframe['kijun_sen_inf']))
            ) &

            # NOTE: I keep the volume checks of feels like it has not much benifit when trading leverage tokens,
            # maybe im wrong!?
            # (dataframe['vfi'] < 0.0) &
            (dataframe['volume'] > 0)

        ,'sell'] = 1

        return dataframe


    #
    # Custom Stoploss
    #

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            sroc = dataframe['sroc_inf'].iat[-1]
            bb_trailing = dataframe[self.sell_cstp_bb_trailing_input.value].iat[-1]
        # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
        else:
            sroc = self.custom_trade_info[trade.pair]['sroc_inf'].loc[current_time]['sroc_inf']
            bb_trailing = self.custom_trade_info[trade.pair][self.sell_cstp_bb_trailing_input.value].loc[current_time][self.sell_cstp_bb_trailing_input.value]

        if current_profit < self.sell_cstp_threshold.value:
            if self.sell_cstp_bail_how.value == 'roc' or self.sell_cstp_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if (sroc/100) <= self.sell_cstp_bail_roc.value:
                    return 0.001
            if self.sell_cstp_bail_how.value == 'time' or self.sell_cstp_bail_how.value == 'any':
                # Dynamic bailout based on time
                if trade_dur > self.sell_cstp_bail_time.value:
                    return 0.001

        if current_profit < self.sell_cstp_trailing_only_offset_is_reached.value:
            if current_rate <= bb_trailing:
                return 0.001
            else:
                return -1

        desired_stoploss = current_profit / self.sell_cstp_trailing_stop_profit_devider.value
        return max(min(desired_stoploss, self.sell_cstp_trailing_max_stoploss.value), 0.025)


    #
    # Dynamic ROI
    #

    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:

        minimal_roi = self.minimal_roi
        _, table_roi = self.min_roi_reached_entry(trade_dur)

        # see if we have the data we need to do this, otherwise fall back to the standard table
        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                rmi_trend = dataframe['rmi-up-trend_inf'].iat[-1]
                candle_trend = dataframe['candle-up-trend_inf'].iat[-1]
                ssl_dir = dataframe['ssl-dir_inf'].iat[-1]
            # If in backtest or hyperopt, get the indicator values out of the trades dict (Thanks @JoeSchr!)
            else:
                rmi_trend = self.custom_trade_info[trade.pair]['rmi-up-trend_inf'].loc[current_time]['rmi-up-trend_inf']
                candle_trend = self.custom_trade_info[trade.pair]['candle-up-trend_inf'].loc[current_time]['candle-up-trend_inf']
                ssl_dir = self.custom_trade_info[trade.pair]['ssl-dir_inf'].loc[current_time]['ssl-dir_inf']

            min_roi = table_roi
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            pullback_value = (max_profit - self.sell_droi_pullback_amount.value)
            in_trend = False

            if self.sell_droi_trend_type.value == 'rmi' or self.sell_droi_trend_type.value == 'any':
                if rmi_trend == 1:
                    in_trend = True
            if self.sell_droi_trend_type.value == 'ssl' or self.sell_droi_trend_type.value == 'any':
                if ssl_dir == 'up':
                    in_trend = True
            if self.sell_droi_trend_type.value == 'candle' or self.sell_droi_trend_type.value == 'any':
                if candle_trend == 1:
                    in_trend = True

            # Force the ROI value high if in trend
            if (in_trend == True):
                min_roi = 100
                # If pullback is enabled, allow to sell if a pullback from peak has happened regardless of trend
                if self.sell_droi_pullback.value == True and (current_profit < pullback_value):
                    if self.sell_droi_pullback_respect_table.value == True:
                        min_roi = table_roi
                    else:
                        min_roi = current_profit / 2

        else:
            min_roi = table_roi

        return trade_dur, min_roi

    # Change here to allow loading of the dynamic_roi settings
    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        if self.use_dynamic_roi:
            _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)
        else:
            _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    # Get the current price from the exchange (or local cache)
    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            # Check if cache has been invalidated
            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate


    #
    # Custom trade info
    #

    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = False

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):

            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate, current_rate)

        return trade_data


#
# Custom indicators
#

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc