# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""No gate, 4096-bar startup: is the longer warm-up itself neutral? (H1 control) (feature 026)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_RegOn4096(SimpleTrigger_FamCombined):
    ATTR_OVERRIDES = {"startup_candle_count": 4096}
