# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""029 BASELINE PROBE (T005): arm D's model entries with the stop invariants watching.

The twin's machinery (borrowed per-fold NNNC_ARO weights, never training, StopInvariantMixin),
but the entry is arm D's own -- the model's prediction against the plan's per-fold threshold --
not the gbb trigger. Run with FT_STOP_INVARIANTS=report: the mixin returns the parent's stop
unchanged and only records violations, so this probe must make exactly arm D's trades. Its
violation count decides whether the 029 merge may put the halting check on every NNNC arm.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_A_Raw_Open import NNNC_A_Raw_Open
from NNNC_TriggerGate import NNNC_TriggerGate


class NNNC_SIProbe(NNNC_TriggerGate):
    get_entry_conditions = NNNC_A_Raw_Open.get_entry_conditions
