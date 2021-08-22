# Common configuration items to be used across Strategies

from freqtrade.strategy.interface import IStrategy

exchange_name = "binanceus"

# ROI table:
minimal_roi = {
    "0": 0.155,
    "65": 0.075,
    "87": 0.039,
    "305": 0
}

# Stoploss:
stoploss = -0.253

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
    "buy_btcndrop_enabled": False,
    "buy_btcnseq_enabled": True,
    "buy_emabounce_enabled": False,
    "buy_fisherbb_enabled": True,
    "buy_macdcross_enabled": False,
    "buy_ndrop_enabled": False,
    "buy_nseq_enabled": True,
}

# BBBHold
strategyParameters["BBBHold"] = {
    "buy_bb_gain": 0.01,
    "buy_fisher": -0.8,
    "buy_fisher_enabled": True,
    "buy_mfi": 25.0,
    "buy_mfi_enabled": False,
}

# BigDrop
strategyParameters["BigDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.015,
    "buy_fisher": -0.85,
    "buy_fisher_enabled": True,
    "buy_mfi": 11.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 7,
}

# BTCBigDrop
strategyParameters["BTCBigDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": -0.57,
    "buy_fisher_enabled": False,
    "buy_mfi": 20.0,
    "buy_mfi_enabled": True,
    "buy_num_candles": 8,
}

# BTCJump
strategyParameters["BTCJump"] = {
    "buy_bb_gain": 0.01,
    "buy_btc_jump": 0.005,
    "buy_fisher": 0.45,
}

# BTCNDrop
strategyParameters["BTCNDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": -0.6,
    "buy_fisher_enabled": False,
    "buy_mfi": 39.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 3,
}

# BTCNSeq
strategyParameters["BTCNSeq"] = {
    "buy_bb_enabled": False,
    "buy_bb_gain": 0.04,
    "buy_drop": 0.01,
    "buy_fisher": -0.82,
    "buy_fisher_enabled": True,
    "buy_num_candles": 7,
}

# EMABounce
# No parameters found!

# FisherBB
strategyParameters["FisherBB"] = {
    "buy_bb_gain": 0.03,
    "buy_fisher": 0.77,
}

# FisherBB2
strategyParameters["FisherBB2"] = {
    "buy_bb_gain": 0.02,
    "buy_fisher": -0.93,
}

# MACDCross
strategyParameters["MACDCross"] = {
    "buy_adx": 25.0,
    "buy_adx_enabled": False,
    "buy_bb_enabled": True,
    "buy_bb_gain": 0.01,
    "buy_dm_enabled": False,
    "buy_fisher": 0.15,
    "buy_fisher_enabled": False,
    "buy_mfi": 10.0,
    "buy_mfi_enabled": False,
    "buy_neg_macd_enabled": False,
    "buy_period": 13,
    "buy_sar_enabled": True,
}

# NDrop
strategyParameters["NDrop"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.01,
    "buy_fisher": -0.32,
    "buy_fisher_enabled": False,
    "buy_mfi": 38.0,
    "buy_mfi_enabled": False,
    "buy_num_candles": 3,
}

# NSeq
strategyParameters["NSeq"] = {
    "buy_bb_enabled": False,
    "buy_drop": 0.009,
    "buy_fisher": 0.98,
    "buy_fisher_enabled": True,
    "buy_mfi": 35.0,
    "buy_mfi_enabled": True,
    "buy_num_candles": 6,
}