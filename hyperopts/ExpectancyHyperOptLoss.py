"""
ExpectancyHyperOptLoss

This module is a custom HyperoptLoss class

The goal is to use Expectancy as a metric, but also filters out bad scenarios (losing, not enough tradees etc)
For details on Expectancy, refere to: https://www.freqtrade.io/en/stable/edge/

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 2                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 8    # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)



class ExpectancyHyperOptLoss(IHyperOptLoss):


    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        debug_level = 0 # displays (more) messages if higher

        days_period = (max_date - min_date).days
        # target_trades = days_period*EXPECTED_TRADES_PER_DAY
        if config['max_open_trades']:
            target_trades = days_period * config['max_open_trades']
        else:
            target_trades = days_period * EXPECTED_TRADES_PER_DAY

        # Calculate trade loss metric first, because this is used elsewhere
        # Several other metrics are misleading if there are not enough trades

        # trade loss
        if trade_count > MIN_TRADES_PER_DAY * days_period:
            num_trades_loss = (target_trades - trade_count) / target_trades
        else:
            # just return a large number if insufficient trades. Makes other calculations easier/safer
            if debug_level > 1:
                print(" \tTrade count too low:{:.0f}".format(trade_count))
            return UNDESIRED_SOLUTION

        total_profit = results["profit_abs"]

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit > 0.0001, 'upside_returns'] = 1.0

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            winning_count = results['upside_returns'].sum()

        # Losing trades
        results['downside_returns'] = 0
        results.loc[total_profit < 0, 'downside_returns'] = 1.0


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
        e = r * w - l

        return -e
