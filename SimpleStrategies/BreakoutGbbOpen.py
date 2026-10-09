# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""BreakoutGbbOpen — the `breakout_gbb` TRIGGER, traded with NO model.

Board: EV (fillable) +0.000692, 95% CI vs null [+0.001284, +0.002675], verdict
`selection_value`, base rate 12.49%. The conjunction of the two triggers that have each
now been traded alone (`breakout`, -? and `gbb`, +2.64pp), which makes it the one arm
whose result is partly PREDICTED by the others: if the two components' returns do not
compose, the conjunction is doing something neither does.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class BreakoutGbbOpen(TriggerOnlyStrategy):
    """`breakout_gbb` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "breakout_gbb"
