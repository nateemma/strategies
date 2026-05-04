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


class BaseNNMTStrategy(BaseNNStrategy):
    """
    Multi-task neural network strategy base.

    Empty in this commit; subsequent phases move attributes and methods up from
    NNMTStrategy. NNMTStrategy still inherits the full multi-task surface area
    via this class — behavior is unchanged.
    """
    pass
