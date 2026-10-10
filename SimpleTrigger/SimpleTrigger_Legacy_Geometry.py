# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GeometryOpen (recorded arm `wf_geo`) re-expressed on the SimpleTrigger family -- feature 023 SC-001.

LEGACY_TRIGGERONLY preset: TriggerOnly's framework verbatim, INCLUDING its static -5% stop.
That is not arm D's framework (023 research R1); this class exists only to prove the new
family reproduces the recorded trades exactly. Use the ARM_D_OPEN default for new work."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_Legacy_Geometry(SimpleTriggerStrategy):
    TRIGGER_NAME = "geometry"
    PRESET = "LEGACY_TRIGGERONLY"
