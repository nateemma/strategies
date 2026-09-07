# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Oracle_Path — oracle-gate screen for the PATH-AWARE label. A CEILING.

`regime/ORACLE_GATE.md` is explicit: before training a model on any new label,
run that label through the guards-off oracle. PASS requires W1 return > 0 AND
at least 200 trades. A FAIL records progress but does NOT license training.

This arm exists to apply that gate to `LabelMethod.gbb_path` — the path-aware
buy/sell pair that drops labels whose barrier fires before the promised move.
It needs no training, because an oracle's entry signal IS the label.

Held FIXED against `NNMT_Oracle_Lbl02` so the path constraint is the only
difference: same label thresholds (0.02), same guards, same exit configuration,
same aliased weights. The path labels are a strict SUBSET of the incumbent's
(verified), so any change is attributable to the constraint alone.

NOTHING MEASURED WITH THIS CLASS IS PROMOTABLE.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_Oracle_Lbl02 import NNMT_Oracle_Lbl02


class NNMT_Oracle_Path(NNMT_Oracle_Lbl02):

    # LabelMethod.gbb_path (27). The incumbent is gbb (17).
    TRAINING_TYPE = 27
