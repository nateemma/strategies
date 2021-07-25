# Common configuration items to be used across Strategies

# # ROI table:
# minimal_roi = {
#     "0": 0.278,
#     "39": 0.087,
#     "124": 0.038,
#     "135": 0
# }
#
# # Trailing stop:
# trailing_stop = True
# trailing_stop_positive = 0.172
# trailing_stop_positive_offset = 0.212
# trailing_only_offset_is_reached = False

# ROI table:
minimal_roi = {
    "0": 0.037,
    "15": 0.021,
    "47": 0.011,
    "86": 0
}

# Trailing stop:
trailing_stop = True
trailing_stop_positive = 0.105
trailing_stop_positive_offset = 0.144
trailing_only_offset_is_reached = True

# Stoploss:
stoploss = -0.349

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