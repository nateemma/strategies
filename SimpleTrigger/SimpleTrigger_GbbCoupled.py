# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`gbb` in COUPLED exit mode with targets chosen to MAP ONTO the decoupled exit (023 SC-007).

The take-profit and max hold are derived from the label's targets instead of set directly.
`min_gain` is the gross move that nets ARM_D_OPEN's 0.022 take-profit after fees, and `horizon`
is 960 bars = ARM_D_OPEN's 10-day max hold. `gbb`'s buy label has no loss barrier, so the stop is
unchanged ("none-in-label"). Its framework_sha therefore equals SimpleTrigger_Gbb's, and its trades
must too: this is the identity that proves coupled mode is a pure parameter mapping. Real coupling
experiments set the targets the label was actually built with.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from Framework.TrainingSignals import LabelMethod
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy
from utils.ExitPolicy import price_ratio_for_net


class SimpleTrigger_GbbCoupled(SimpleTriggerStrategy):
    TRIGGER_NAME = "gbb"
    EXIT_MODE = "coupled"
    LABEL_METHOD = LabelMethod.gbb
    FEE_PER_SIDE = 0.001
    LABEL_TARGETS = {"min_gain": price_ratio_for_net(0.022, 0.001) - 1.0, "horizon": 960}
