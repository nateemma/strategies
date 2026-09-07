# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Path_s7 — PATH-AWARE label arm, seed 7.

Identical to NNMT_Lbl02_s7 in every respect except the labeler:
LabelMethod.gbb_path (27) instead of gbb (17). Same 0.02 thresholds, same
profit-head pin, same control exit configuration in its .json, and its own
model directory via the inherited class-name-keyed get_model_path.

So the comparison NNMT_Path_s7 vs NNMT_Lbl02_s7 isolates the PATH
CONSTRAINT alone (Constitution VIII), and the path labels are a verified strict
SUBSET of the incumbent's.

Licensed by the oracle gate: gbb_path scored +23.991% guards-off on W1 against
the incumbent's +17.791% (regime/PATH_LABEL_GATE.md). The gate says the label
COULD pay; this arm measures whether the model learns it. The prior from this
family is poor -- a 45.9pp ceiling move previously bought -0.157pp realised.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_Lbl02_s7 import NNMT_Lbl02_s7


class NNMT_Path_s7(NNMT_Lbl02_s7):

    # LabelMethod.gbb_path — the only difference from the parent.
    TRAINING_TYPE = 27
