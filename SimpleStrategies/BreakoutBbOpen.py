# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""BreakoutBbOpen — the `breakout_bb` TRIGGER, traded with NO model.

Board: EV (fillable) +0.001546, 95% CI vs null [+0.002032, +0.003650], verdict
`selection_value`, base rate 6.08%. THE STRONGEST non-diagnostic row on the board by
EV and by interval, so if the economic-value column predicts a return anywhere it should
predict one here. It also has the lowest gapfill share of the family (27.1%), which the
report flags as a partial confound of that column.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class BreakoutBbOpen(TriggerOnlyStrategy):
    """`breakout_bb` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "breakout_bb"
