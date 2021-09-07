"""
QuickProfitHyperOptLoss

This module is a custom HyperoptLoss class

The goal is to primarily use profit as a metric, but also take into account the number of trades,
trade duration, average profit, win/loss %, Sharpe ratio, Sortino ratio etc.
This version prioritises a short duration

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict

# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 10          # used to set target goals
MIN_TRADES_PER_DAY = 2                # used to filter out scenarios where there are not enough trades
EXPECTED_PROFIT_PER_TRADE = 0.010     # be realistic. Setting this too high will eliminate potentially good solutions
EXPECTED_AVE_PROFIT = 0.050           # used to assess actual profit vs desired profit. OK to set high
MAX_ACCEPTED_TRADE_DURATION = 240    # 240 = 4 hours

class QuickProfitHyperOptLoss(IHyperOptLoss):
    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        days_period = (max_date - min_date).days
        # target_trades = days_period*EXPECTED_TRADES_PER_DAY
        if config['max_open_trades']:
            target_trades = days_period*config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # trade loss
        if trade_count > MIN_TRADES_PER_DAY*days_period:
            num_trades_loss = (target_trades-trade_count)/target_trades
        else:
            # just return a large number if insufficient trades. Makes other calculations easier/safer
            num_trades_loss = 20.0
            return 20.0

        # Absoulte Profit
        if config['max_open_trades'] and config['stake_amount']:
            abs_profit_loss = -1.0 * results["profit_abs"].sum() / (config['max_open_trades']*config['stake_amount'])
        else:
            abs_profit_loss = -1.0 * results["profit_abs"].sum() / 10000.0


        # total profit (relative to expected profit)
        # note that we don't have enough info to calculate profit % because we don't know the original investment
        # so, we approximate
        total_profit = results["profit_abs"]
        total_profit = total_profit - 0.0005 # adding slippage of 0.1% per trade
        # guess the expected profit using mean stake amount
        # results['expected_profit'] = results['stake_amount']*EXPECTED_PROFIT_PER_TRADE
        # profit_diff = (results['expected_profit'] - total_profit)/results['expected_profit']
        profit_sum = results["profit_abs"].sum()
        expected_sum = results['stake_amount'].mean() * trade_count * EXPECTED_PROFIT_PER_TRADE
        day_profit_loss = -1.0 * (profit_sum / days_period) / 100.0


        exp_profit_loss = (expected_sum - profit_sum) / expected_sum


        # if num_trades_loss < 0.0:
        #     print("profit_sum:{:.3f} expected_sum:{:.3f} day_profit_loss:{:.3f} exp_profit_loss:{:.3f}" \
        #           .format(profit_sum, expected_sum, day_profit_loss, exp_profit_loss))

        # trade duration (taken from default loss function)
        trade_duration = results['trade_duration'].mean()
        duration_loss = (trade_duration-MAX_ACCEPTED_TRADE_DURATION)/MAX_ACCEPTED_TRADE_DURATION

        # Losing trades
        results['downside_returns'] = 0
        # results.loc[total_profit < 0, 'downside_returns'] = results['profit_ratio']
        results.loc[total_profit < 0, 'downside_returns'] = 1.0
        losing_count = results['downside_returns'].sum()
        losing_loss = losing_count/trade_count - 0.25 # relative to 25%

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit > 0.001, 'upside_returns'] = 1.0
        # results.loc[total_profit > 0, 'upside_returns'] = results['profit_ratio']
        winning_loss = 0.7 - results['upside_returns'].sum()/trade_count # goal is 70%

        # Win/loss ratio
        if (losing_count > 0):
            win_ratio_loss = -(results['upside_returns'].sum() / losing_count)/10.0
        else:
            win_ratio_loss = -1.0 # 0 is v.good if sufficient trades

        # Ave. Profit
        ave_profit_loss = 10.0 * (EXPECTED_AVE_PROFIT - results['profit_ratio'].mean())

        # Profit Standard Deviation
        profit_std_loss = np.std(results['profit_ratio']) - 0.02

        # Sharpe Ratio
        expected_returns_mean = total_profit.sum() / days_period
        up_stdev = np.std(total_profit)
        if up_stdev != 0:
            # calculate Sharpe ratio, but scale down to match other parameters
            # sharp_ratio_loss = 0.8 - (expected_returns_mean / up_stdev * np.sqrt(365)) / 100.0
            sharp_ratio_loss = 0.1 - (expected_returns_mean / up_stdev * np.sqrt(365)) / 100.0
        else:
            # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
            sharp_ratio_loss = 2.0

        # Sortino Ratio
        down_stdev = np.std(results['downside_returns'])
        if down_stdev != 0:
            sortino_ratio_loss = -1.0 * (expected_returns_mean / down_stdev * np.sqrt(365)) / 10000.0
        else:
            # Define high (negative) sortino ratio to be clear that this is NOT optimal.
            sortino_ratio_loss = 2.0

        # amplify if both Sharpe and Sortino are -ve or both +ve
        if ((sharp_ratio_loss < 0.0) and (sortino_ratio_loss < 0.0)) or \
                ((sharp_ratio_loss > 0.0) and (sortino_ratio_loss > 0.0)):
            sharp_ratio_loss = 2.0 * sharp_ratio_loss
            sortino_ratio_loss = 2.0 * sortino_ratio_loss


        # Stoploss ratio
        stoploss_loss = -results['stop_loss_ratio'].mean()
        # sell_reason = results['sell_reason']
        # print('sell_reason: ', sell_reason)

        # weight the results (values are based on trial & error). Goal is for anything -ve to be a decent  solution
        num_trades_loss    = 1.0 * num_trades_loss
        duration_loss      = 1.0 * duration_loss
        abs_profit_loss    = 1.0 * abs_profit_loss
        exp_profit_loss    = 0.5 * exp_profit_loss
        day_profit_loss    = 0.5 * day_profit_loss
        ave_profit_loss    = 0.5 * ave_profit_loss
        profit_std_loss    = 0.0 * profit_std_loss
        losing_loss        = 2.0 * losing_loss
        winning_loss       = 2.0 * winning_loss
        win_ratio_loss     = 0.25 * win_ratio_loss
        sharp_ratio_loss   = 1.0 * sharp_ratio_loss
        sortino_ratio_loss = 0.75 * sortino_ratio_loss
        # stoploss_loss      = 1.0 * stoploss_loss


        result = abs_profit_loss + num_trades_loss + duration_loss + exp_profit_loss + day_profit_loss + \
                 ave_profit_loss + losing_loss +  winning_loss + win_ratio_loss + sharp_ratio_loss + \
                 sortino_ratio_loss

        #if result < 0.0:
        # print("abs:{:.3f} trade:{:.3f} dur:{:.3f}  exp_profit:{:.3f} daily:{:.3f} ave_profit:{:.3f} " \
        #       "losing:{:.3f}  winning:{:.3f} win_ratio:{:.3f} sharpe:{:.3f} sortino:{:.3f} " \
        #       " Total:{:.3f}"\
        #       .format(abs_profit_loss, num_trades_loss, duration_loss, exp_profit_loss, day_profit_loss, \
        #               ave_profit_loss, losing_loss, winning_loss, win_ratio_loss, sharp_ratio_loss, \
        #               sortino_ratio_loss, result))

        return result
