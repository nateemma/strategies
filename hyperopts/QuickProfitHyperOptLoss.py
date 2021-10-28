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

# Constants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration

EXPECTED_TRADES_PER_DAY = 10          # used to set target goals
MIN_TRADES_PER_DAY = 2                # used to filter out scenarios where there are not enough trades
EXPECTED_PROFIT_PER_TRADE = 0.010     # be realistic. Setting this too high will eliminate potentially good solutions
EXPECTED_AVE_PROFIT = 0.050           # used to assess actual profit vs desired profit. OK to set high
EXPECTED_TRADE_DURATION = 120         # goal for duration (or shorter) in seconds
MAX_TRADE_DURATION = 300              # max allowable duration

UNDESIRED_SOLUTION = 20.0             # indicates that we don't want this solution (so hyperopt will avoid)

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
            target_trades = 2.0 * days_period * config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # trade loss
        if trade_count > MIN_TRADES_PER_DAY*days_period:
            num_trades_loss = (target_trades-trade_count)/target_trades
        else:
            # just return a large number if insufficient trades. Makes other calculations easier/safer
            return UNDESIRED_SOLUTION

        # Absoulte Profit
        num_months = days_period / 30.0
        profit_sum = results["profit_abs"].sum()

        if profit_sum < 0.0:
            # print("profit_sum too low:{:.3f} ".format(profit_sum))
            return UNDESIRED_SOLUTION # if -ve profit, just return

        if config['dry_run_wallet']:
            abs_profit_loss = profit_sum / config['dry_run_wallet']
        elif config['max_open_trades'] and config['stake_amount']:
            abs_profit_loss = profit_sum / (config['max_open_trades']*config['stake_amount']*num_months)
        else:
            abs_profit_loss = profit_sum / (10000.0 * num_months)

        # scale loss by #months so that it's consistent no matter the length of the run
        # use 15% per month as goal, scale by 10
        abs_profit_loss = 10.0 * (0.15 - (abs_profit_loss / num_months))

        # punish if below goal
        if abs_profit_loss > 0.0:
            return UNDESIRED_SOLUTION

        # total profit (relative to expected profit)
        # note that we don't have enough info to calculate profit % because we don't know the original investment
        # so, we approximate
        total_profit = results["profit_abs"]
        expected_sum = results['stake_amount'].mean() * trade_count * EXPECTED_PROFIT_PER_TRADE
        day_profit_loss = -1.0 * (profit_sum / days_period) / 100.0


        exp_profit_loss = (expected_sum - profit_sum) / expected_sum


        # if num_trades_loss < 0.0:
        #     print("profit_sum:{:.3f} expected_sum:{:.3f} day_profit_loss:{:.3f} exp_profit_loss:{:.3f}" \
        #           .format(profit_sum, expected_sum, day_profit_loss, exp_profit_loss))

        # trade duration (taken from default loss function)
        trade_duration = results['trade_duration'].mean()
        duration_loss = (trade_duration-EXPECTED_TRADE_DURATION)/EXPECTED_TRADE_DURATION

        # punish if below goal
        if trade_duration > MAX_TRADE_DURATION:
            return UNDESIRED_SOLUTION

        # Losing trades
        results['downside_returns'] = 0
        results.loc[total_profit < 0, 'downside_returns'] = 1.0
        losing_count = results['downside_returns'].sum()

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit > 0.0001, 'upside_returns'] = 1.0
        winning_count = results['upside_returns'].sum()

        # if winning_count < (2.0 * losing_count):
        if winning_count < (1.5 * losing_count):
            # print("winning_count too low:{:.3f} vs losing_count:{:.3f}".format(winning_count, losing_count))
            return UNDESIRED_SOLUTION

        # Expectancy (refer to freqtrade edge page for info)
        w = winning_count / trade_count
        l = 1.0 - w
        results['net_gain'] = results['profit_abs'] * results['upside_returns']
        results['net_loss'] = results['profit_abs'] * results['downside_returns']
        ave_profit = results['net_gain'].sum() / trade_count
        ave_loss = results['net_loss'].sum() / trade_count
        if abs(ave_loss) < 0.001:
            ave_loss = 0.001
        r = ave_profit / abs(ave_loss)
        e = r*w - l

        # Assess relative to 30% gain (arbitrary) and scale up to make it easier to read
        # expectancy_loss = 10.0*(0.3-e)
        # expectancy_loss = 1.0*(0.3-e)
        expectancy_loss = -e
        if expectancy_loss > 0.0:
            return UNDESIRED_SOLUTION

        # Win/Loss ratio (losses here are draws & losses)
        win_loss_ratio_loss = 1.5 - (winning_count / (trade_count - winning_count))

        # punish if below goal
        if win_loss_ratio_loss > 0.0:
            return UNDESIRED_SOLUTION

        # Sharpe Ratio
        expected_returns_mean = total_profit.sum() / days_period
        up_stdev = np.std(total_profit)
        if up_stdev != 0:
            # calculate Sharpe ratio, but scale down to match other parameters
            sharp_ratio_loss = 0.01 - (expected_returns_mean / up_stdev * np.sqrt(365)) / 100.0
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

        # define weights
        weight_num_trades = 0.4
        weight_duration = 4.0
        weight_abs_profit = 1.0
        weight_exp_profit = 0.2
        weight_day_profit = 0.2
        weight_expectancy = 0.5
        weight_win_loss_ratio = 1.0
        weight_sharp_ratio = 1.0
        weight_sortino_ratio = 0.25

        if config['exchange']['name']:
            if (config['exchange']['name'] == 'kucoin'):
                # kucoin is extremely volatile, with v.high profits in backtesting (but not in real markets)
                # so, reduce influence of absolute profit and no. of trades (and sharpe/sortino)
                # the goal is reduce the number of losing and highly risky trades (the cost is some loss of profits)
                weight_num_trades = 0.1
                weight_duration = 2.0
                weight_abs_profit = 0.1
                weight_exp_profit = 0.2
                weight_day_profit = 0.2
                weight_expectancy = 1.0
                weight_win_loss_ratio = 1.0
                weight_sharp_ratio = 0.25
                weight_sortino_ratio = 0.05

        # weight the results (values are based on trial & error). Goal is for anything -ve to be a decent  solution
        num_trades_loss     = weight_num_trades * num_trades_loss
        duration_loss       = weight_duration * duration_loss
        abs_profit_loss     = weight_abs_profit * abs_profit_loss
        exp_profit_loss     = weight_exp_profit * exp_profit_loss
        day_profit_loss     = weight_day_profit * day_profit_loss
        expectancy_loss     = weight_expectancy * expectancy_loss
        win_loss_ratio_loss = weight_win_loss_ratio * win_loss_ratio_loss
        sharp_ratio_loss    = weight_sharp_ratio * sharp_ratio_loss
        sortino_ratio_loss  = weight_sortino_ratio * sortino_ratio_loss


        result = abs_profit_loss + num_trades_loss + duration_loss + exp_profit_loss + day_profit_loss + \
                 win_loss_ratio_loss + expectancy_loss + sharp_ratio_loss + sortino_ratio_loss

        if (result < 0.0):
            print(" prof:{:.3f} n:{:.3f} dur:{:.3f} w/l:{:.3f} " \
                  "expy:{:.3f}  sharpe:{:.3f} sortino:{:.3f} " \
                  " Total:{:.3f}"\
                  .format(abs_profit_loss, num_trades_loss, duration_loss,  win_loss_ratio_loss, \
                          expectancy_loss, sharp_ratio_loss, sortino_ratio_loss, \
                          result))


        return result
