"""
ExpectancyHyperOptLoss

This module is a custom HyperoptLoss class

The goal is to use Expectancy as a metric, but also filters out bad scenarios (losing, not enough tradees etc)
For details on Expectancy, refere to: https://www.freqtrade.io/en/stable/edge/

To deploy this, copy the file to the <freqtrade>/user_data/hyperopts directory
"""
from math import exp

from pandas import DataFrame

from freqtrade.data.metrics import calculate_calmar
from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import numpy as np
from typing import Any, Dict

# Contstants to allow evaluation in cases where thre is insufficient (or nonexistent) info in the configuration
EXPECTED_TRADES_PER_DAY = 2                       # used to set target goals
MIN_TRADES_PER_DAY = EXPECTED_TRADES_PER_DAY / 3  # used to filter out scenarios where there are not enough trades
EXPECTED_AVE_PROFIT = 0.004                       # used to assess actual profit vs desired profit. Typical is 0.4% (0.004)
UNDESIRED_SOLUTION = 3.0                          # indicates that we don't want this solution (so hyperopt will avoid)


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

        # if debug_level >= 2:
        #     profit_cols = [col for col in results.columns if 'profit' in col]
        #     print("Profit columns:")
        #     print(profit_cols)

        stake = backtest_stats['stake_amount']
        total_profit_pct = results["profit_abs"] / stake

        # Winning trades
        results['upside_returns'] = 0
        results.loc[total_profit_pct > 0.0001, 'upside_returns'] = 1.0

        if backtest_stats['wins']:
            winning_count = backtest_stats['wins']
        else:
            winning_count = results['upside_returns'].sum()

        # Losing trades
        results['downside_returns'] = 0
        results.loc[total_profit_pct < 0, 'downside_returns'] = 1.0

        # Expectancy (refer to freqtrade edge page for info)
        w = winning_count / trade_count
        l = 1.0 - w
        results['net_gain'] = total_profit_pct * results['upside_returns']
        results['net_loss'] = total_profit_pct * results['downside_returns']
        ave_profit = results['net_gain'].sum() / trade_count
        ave_loss = results['net_loss'].sum() / trade_count

        if abs(ave_loss) < 0.01:
            ave_loss = 0.01  # set min loss = 1%, otherwise results can be wildly skewed
        r = ave_profit / abs(ave_loss)
        e = r * w - l

        # expectancy_loss = 1.0 - e  # goal is <1.0
        expectancy_loss = -e
        drawdown_loss = 0.0
        abs_profit_loss = 0.0

        # use Calmar and profit as a tie-breaker
        starting_balance = config['dry_run_wallet']
        calmar_loss = -calculate_calmar(results, min_date, max_date, starting_balance) / 100.0
        if (debug_level > 1):
                print(f"calmar_loss:{calmar_loss:.3f}")

        # Daily/Average profit
        ave_profit_loss = 0.0
        days_period = (max_date - min_date).days
        if 'profit_total_abs' in backtest_stats:
            profit_sum = backtest_stats['profit_total_abs']
        elif "profit_abs" in results:
            profit_sum = results["profit_abs"].sum()
        else:
            profit_sum = 0.0

        if 'profit_mean' in backtest_stats:
            ave_profit_loss = EXPECTED_AVE_PROFIT - backtest_stats['profit_mean']
        else:
            ave_profit_loss = EXPECTED_AVE_PROFIT - ((profit_sum / days_period) / 100.0)
        ave_profit_loss = -ave_profit_loss * 100.0

        if profit_sum < 0.0:
            if (debug_level > 1):
                print(f"-ve profit:{profit_sum:.3f}")
            ave_profit_loss = UNDESIRED_SOLUTION + abs(ave_profit_loss)


        calmar_loss = 1.0 * calmar_loss
        ave_profit_loss = 1.0 * ave_profit_loss

        calmar_loss = max(-1.0, calmar_loss) # limit contribution
        ave_profit_loss = max(-1.0, ave_profit_loss)

        result = expectancy_loss + calmar_loss + ave_profit_loss

        if abs_profit_loss < -100.0:
            result = UNDESIRED_SOLUTION
        elif abs_profit_loss > 0.0:
            result = result + 2.0 # penalise -ve profit

        if ((debug_level == 1) & (result < 0.0)) | (debug_level > 1):
            print(f" exp:{expectancy_loss:.2f} calmar:{calmar_loss:.2f} profit:{ave_profit_loss:.2f} Total:{result:.2f}")

        return result
