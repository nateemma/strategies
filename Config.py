# Common configuration items to be used across Strategies

minimal_roi = {
    "0": 0.038,
    "20": 0.026,
    "117": 0.016,
    "286": 0
}

trailing_stop = True
trailing_stop_positive = 0.034
trailing_stop_positive_offset = 0.037
trailing_only_offset_is_reached = True

# Stoploss:
stoploss = -0.349
#
# minimal_roi = {
#     "0": 0.038,
#     "20": 0.026,
#     "117": 0.016,
#     "286": 0
# }
#
# trailing_stop = True
# trailing_stop_positive = 0.034
# trailing_stop_positive_offset = 0.037
# trailing_only_offset_is_reached = True
#
# # Stoploss:
# stoploss = -0.349

# Optimal timeframe for the strategy
timeframe = '5m'

# run "populate_indicators" only for new candle
process_only_new_candles = False

# Experimental settings (configuration will overide these if set)
use_sell_signal = True
sell_profit_only = True
ignore_roi_if_buy_signal = False

# Optional order type mapping
order_types = {
    'buy': 'limit',
    'sell': 'limit',
    'stoploss': 'market',
    'stoploss_on_exchange': False
}

# Optional order time in force.
order_time_in_force = {
    'buy': 'gtc',
    'sell': 'gtc'
}