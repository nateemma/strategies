# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_Open_Path_s7 — PATH-AWARE label at the OPEN gate, seed 7.

Identical to NNNC_MLX_Open_s7 in every respect except the labeler:
LabelMethod.gbb_path (27) instead of gbb (17). Same 0.02 thresholds, same
guards-open / threshold-0.80 operating point pinned in its .json, and its own
weights (get_model_path is class-name keyed and deliberately not overridden,
because this arm must train at its own seed).

WHY THE OPEN GATE, when the same label was rejected on NNNC's production gate:
its ceiling advantage is entirely gate-dependent.

    gbb_path advantage over gbb, W1 ceiling
        CLOSED gate (production)   -0.338pp
        OPEN gate                  +3.112pp

and that is also where the headroom is:

    W1 headroom (perfect entry minus real model)
        CLOSED gate   +3.488pp   (real +10.446%, ceiling +13.934%)
        OPEN gate    +17.824pp   (real  +8.720%, ceiling +26.544%)

The open gate is the only configuration measured in this repository with a large
headroom for a model to work in. NNMT has -4.558pp and could never pay; NNNC
closed has +3.488pp. This arm asks whether a better-specified label converts any
of that 17.8pp — the question the closed-gate screen could not reach.

The prior is still cautious: on NNMT a +6.2pp ceiling gain produced -0.091pp
realised. The difference here is that there is measured room to capture.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Open_s7 import NNNC_MLX_Open_s7


class NNNC_Open_Path_s7(NNNC_MLX_Open_s7):

    # LabelMethod.gbb_path — the only difference from the parent.
    TRAINING_TYPE = 27
