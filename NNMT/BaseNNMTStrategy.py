# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
BaseNNMTStrategy - shared scaffolding for Neural Network Multi-Task strategies.

Sits between BaseNNStrategy (single-task defaults + shared pipeline) and the
concrete NNMTStrategy. Multi-task class attributes, target calculators, and
overridden pipeline methods belong here so a second multi-task strategy can
inherit them without duplicating NNMTStrategy.
"""

import sys
from pathlib import Path

# Match NNMTStrategy's sys.path setup so sibling-module imports resolve
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter


class BaseNNMTStrategy(BaseNNStrategy):
    """
    Multi-task neural network strategy base.

    Empty in this commit; subsequent phases move attributes and methods up from
    NNMTStrategy. NNMTStrategy still inherits the full multi-task surface area
    via this class — behavior is unchanged.
    """

    profit_conflict_to_neutral = True
    PROFIT_EMA_SPAN = 5
    PROFIT_ATR_SCALE = 1.0

    # -----------
    # Hyperopt parameters
    # -----------

    # Consecutive signal filter (Note: increasing causes delay in real-time detection)
    min_consecutive_buys = IntParameter(
        1, 2, default=1, space="buy", optimize=True, load=True
    )

    # prediction

    optimize_bias = False

    bias_trading_sell = DecimalParameter(
        0.01,
        0.06,
        default=0.03,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_trading_buy = DecimalParameter(
        0.01,
        0.06,
        default=0.05,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_profit_low = DecimalParameter(
        0.05,
        0.18,
        default=0.09,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )
    bias_profit_high = DecimalParameter(
        0.05,
        0.18,
        default=0.08,
        decimals=2,
        space="buy",
        optimize=optimize_bias,
        load=True,
    )

    apply_task_filters = BooleanParameter(
        default=False,
        space="buy",
        optimize=True,
        load=True,
    )
    # -----------
    # Class level parameters
    # -----------

    augment_training_data = True  # signal augmentation; GAN augmentation gates on gan_augment

    filter_signals = False  # don't double filter

    regime_lookback = 20  # Periods for regime detection
    volatility_lookback = 10  # Periods to calculate volatility
    risk_threshold = 0.02  # Risk threshold for binary classification

    PROFIT_TAKE_THRESHOLD = 0.02
    PROFIT_STOP_LOSS_THRESHOLD = 0.015

    task_thresholds = {
        "momentum": {"low": -0.5, "high": 0.6},
        "flow": {"low": -5.0, "high": 5.0},
        "profit": {"low": -0.006, "high": 0.006},
    }
