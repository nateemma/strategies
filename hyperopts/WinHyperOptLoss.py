"""
WinHyperOptLoss

This module is a custom HyperoptLoss class based on Profit and Win/Loss ratio

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict


# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 1                         # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 8    # used to filter out scenarios where there are not enough trades
UNDESIRED_SOLUTION = 2.0             # indicates that we don't want this solution (so hyperopt will avoid)


class WinHyperOptLoss(IHyperOptLoss):


    """
    Defines a custom loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        debug_level = 1 # displays (more) messages if higher

        # Winning trades

        total_profit = results["profit_abs"]

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            results['upside_returns'] = 0
            results.loc[total_profit > 0.0001, 'upside_returns'] = 1.0
            winning_count = results['upside_returns'].sum()

        # calculate win ratio loss. Scale so that 0.0 equates to 50% win/loss ratio
        # win_ratio_loss = 10.0 * (0.5 - winning_count / trade_count)
        win_ratio_loss = 10.0 * (winning_count / trade_count)

        # use profit and drawdown as a tie-breaker

        # profitable?
        profit_loss = -backtest_stats['profit_total']
        if profit_loss > 0:
            # print("profit: {:.2f}".format(profit_loss))
            return abs(profit_loss)

        drawdown_loss = 0.0
        if 'max_drawdown' in backtest_stats:
            drawdown_loss = (backtest_stats['max_drawdown'] - 1.0)

        result = win_ratio_loss + drawdown_loss + profit_loss

        return result

