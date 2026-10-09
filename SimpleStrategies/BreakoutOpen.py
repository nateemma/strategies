# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""BreakoutOpen — the plain `breakout` TRIGGER, traded with NO model.

Board: EV (fillable) +0.000807, 95% CI vs null [+0.001361, +0.002802], verdict
`selection_value`, base rate 6.45%. One of four `selection_value` rows, all breakout
family, and the one that isolates the bare prior-high breakout from the volume,
Bollinger and gbb variants -- so it says whether the family's score is the breakout
itself or the condition bolted onto it.

This trigger is also where the signal-bar leak was largest: paired at bar `i` it scored
+0.009076, paired correctly at `i+1` it scores +0.000807. 91% of its apparent value was
its own signal bar.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class BreakoutOpen(TriggerOnlyStrategy):
    """`breakout` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "breakout"
