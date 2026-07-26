# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_MLX_tronly — trading-head-only variant (all aux heads disabled).

Weights the classifier loss entirely onto the trading head; the aux heads
(regime/risk/momentum/flow/profit) get zero loss weight, so the shared
backbone trains on trading alone. In the head-weight study this was the
best NNMT config (~+0.2pp paired over the 6-head baseline) — the aux heads
provide no positive transfer, so concentrating capacity on trading wins.
Compare to the default NNMT_MLX (trading 0.57, learnable aux kept).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX import NNMT_MLX


class NNMT_MLX_tronly(NNMT_MLX):

    # buy_params = { **NNMT_MLX.buy_params,
    #     "entry_enable_guards": False,
    #     }

    _CLASSIFIER_TASK_WEIGHTS = {
        "trading": 1.0,
        "regime": 0.0,
        "risk": 0.0,
        "momentum": 0.0,
        "flow": 0.0,
        "profit": 0.0,
    }
