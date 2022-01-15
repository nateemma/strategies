# Common configuration items to be used across Strategies

from freqtrade.strategy.interface import IStrategy

exchange_name = "ftx"
informative_pair = "BTC/USD"

# ROI table:
minimal_roi = {
    "0": 0.256,
    "61": 0.102,
    "119": 0.039,
    "324": 0
}

# Stoploss:
stoploss = -0.191

# Trailing stop:
trailing_stop = True
trailing_stop_positive = 0.288
trailing_stop_positive_offset = 0.302
trailing_only_offset_is_reached = False

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

# HYPERPARAMETERS FOR STRATEGIES

# These are typically generated with hyperopt

# To update strategy parameters, just cut&paste the "buy_params" contents from the strategy into the entry of the
# dictionary indexed by strategy name.
# It's done this way to make it easier to update (frequently)

strategyParameters = {}

# EMABounce presents an issue since sometimes there just is no solution. Set a default set of values:
strategyParameters["EMABounce"] = {
    "buy_diff": 0.065,
    "buy_long_period": 50,
    "buy_macd_enabled": False,
    "buy_short_period": 10,
}
#====== AUTO-GENERATED PARAMETERS =========

# ComboHold
strategyParameters["ComboHold"] = {
    "buy_bbbhold_enabled": False,
    "buy_bigdrop_enabled": False,
    "buy_btcjump_enabled": True,
    "buy_btcndrop_enabled": True,
    "buy_btcnseq_enabled": True,
    "buy_emabounce_enabled": False,
    "buy_fisherbb_enabled": False,
    "buy_macdcross_enabled": False,
    "buy_ndrop_enabled": False,
    "buy_nseq_enabled": True,
}


# BBBHold
strategyParameters["BBBHold"] = {
    "buy_bb_gain": 0.02,
    "buy_fisher": 0.17,
    "buy_fisher_enabled": False,
    "buy_mfi": 40.0,
    "buy_mfi_enabled": False,
}

# BigDrop
strategyParameters["BigDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.011,
    "buy_fisher": 0.23,
    "buy_fisher_enabled": True,
    "buy_mfi": 25.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 4,
}

# BTCBigDrop
strategyParameters["BTCBigDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": -0.5,
    "buy_fisher_enabled": True,
    "buy_mfi": 38.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 7,
}

# BTCJump
strategyParameters["BTCJump"] = {
    "buy_bb_gain": 0.01,
    "buy_btc_jump": 0.009,
    "buy_fisher": -0.08,
}

# BTCNDrop
strategyParameters["BTCNDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": 0.99,
    "buy_fisher_enabled": False,
    "buy_mfi": 22.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 3,
}

# BTCNSeq
strategyParameters["BTCNSeq"] = {
    "buy_bb_enabled": False,
    "buy_bb_gain": 0.02,
    "buy_drop": 0.01,
    "buy_fisher": 0.75,
    "buy_fisher_enabled": True,
    "buy_num_candles": 4,
}

# EMABounce
# No parameters found!

# FBB_
strategyParameters["FBB_"] = {
    "buy_bb_gain": 0.02,
    "buy_fisher": -0.68,
}

# FBB_2
strategyParameters["FBB_2"] = {
    "buy_bb_gain": 0.02,
    "buy_fisher": -0.91,
}

# MACDCross
strategyParameters["MACDCross"] = {
    "buy_adx": 18.0,
    "buy_adx_enabled": False,
    "buy_bb_enabled": True,
    "buy_bb_gain": 0.01,
    "buy_dm_enabled": False,
    "buy_fisher": 0.42,
    "buy_fisher_enabled": False,
    "buy_mfi": 19.0,
    "buy_mfi_enabled": False,
    "buy_neg_macd_enabled": True,
    "buy_period": 8,
    "buy_sar_enabled": False,
}

# NDrop
strategyParameters["NDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": 0.19,
    "buy_fisher_enabled": True,
    "buy_mfi": 10.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 4,
}

# NSeq
strategyParameters["NSeq"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.006,
    "buy_fisher": -0.74,
    "buy_fisher_enabled": False,
    "buy_mfi": 15.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 7,
}

#==========================================