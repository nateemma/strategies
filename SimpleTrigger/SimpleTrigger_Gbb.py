# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""`gbb` through arm D's framework, no model -- arm D's twin (feature 023 US3).

Default ARM_D_OPEN preset, so its framework_sha equals NNNC_A_Raw_Open's (asserted in
tests/test_presets.py). Paired against arm D it isolates the model's contribution; paired
against NNNC_TriggerGate (arm D's real NNNC path with the trigger as the entry) it proves the
framework is shared (FR-008)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_Gbb(SimpleTriggerStrategy):
    TRIGGER_NAME = "gbb"
