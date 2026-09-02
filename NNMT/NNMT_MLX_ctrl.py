# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_MLX_ctrl — guards-off CONTROL arm for the NNMT aux-head study.

Identical task weights to NNMT_MLX; the only difference is that the entry
guard chain is disabled. Its model is a byte copy of the NNMT_MLX model
trained in the guards-on pass, so it is the same network — the guards-on and
guards-off arms differ only in entry gating, never in training.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX import NNMT_MLX


class NNMT_MLX_ctrl(NNMT_MLX):

    # The model must participate in EXITS or the arms are
    # indistinguishable: with this False, 67% of exits are a
    # trailing stop and the network has no say at all.
    # MERGE from the parent -- a fresh dict silently drops inherited keys.
    sell_params = {
        **NNMT_MLX.sell_params,
        "enable_exit_signal": True,
    }

    # MERGE from the parent — a fresh dict silently drops inherited keys.
