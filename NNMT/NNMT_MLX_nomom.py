# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_MLX_nomom — task-ablation arm for the NNMT aux-head study (2026-09-01).

Drops 'momentum' (NMI 0.028). Leave-one-out arm.

Loss weights are renormalised to sum to 1.0 in
`MLXClassifierMultiTask._task_weights`, so a naive leave-one-out would also
RAISE trading's share (4/7 -> 4/6) and confound "removed a task" with "gave
trading more weight". Every arm therefore holds trading at 4.0 and the aux
budget at 3.0, redistributing that budget among the remaining aux tasks —
trading's share stays 0.571 across the whole study.

Only training is affected: `apply_task_filters` defaults False, so aux heads
do not gate entries and the labels are identical across arms.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX_Conv import NNMT_MLX_Conv


class NNMT_MLX_nomom(NNMT_MLX_Conv):

    # The model must participate in EXITS or the arms are
    # indistinguishable: with this False, 67% of exits are a
    # trailing stop and the network has no say at all.
    # MERGE from the parent -- a fresh dict silently drops inherited keys.
    sell_params = {
        **NNMT_MLX_Conv.sell_params,
        "enable_exit_signal": True,
    }


    _CLASSIFIER_TASK_WEIGHTS = {'trading': 4.0, 'regime': 0.0, 'risk': 1.5, 'momentum': 0.0, 'flow': 1.5, 'profit': 0.0}
