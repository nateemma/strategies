# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_MLX_trprofit — trading + profit only.

The two highest-signal tasks and the two hardest to predict: `trading` is the
objective itself, and `profit` has by far the highest normalised mutual
information with it (0.130, ~4x the best of the currently-trained heads) while
being near-unlearnable on its own (MCC ~0.23, which is why it was switched off).

The bet is that a hard, highly-relevant auxiliary transfers more than three
easy, weakly-related ones. Every other arm in the study keeps the low-relevance
heads; this one drops all of them.

Trading share is held at 4/7 = 0.571 like every other arm (weights are
renormalised to sum to 1.0 in MLXClassifierMultiTask._task_weights), so the
whole aux budget of 3.0 goes to `profit` alone.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_MLX_Conv import NNMT_MLX_Conv


class NNMT_MLX_trprofit(NNMT_MLX_Conv):

    # The model must participate in EXITS or the arms are
    # indistinguishable: with this False, 67% of exits are a
    # trailing stop and the network has no say at all.
    # MERGE from the parent -- a fresh dict silently drops inherited keys.
    sell_params = {
        **NNMT_MLX_Conv.sell_params,
        "enable_exit_signal": True,
    }


    _CLASSIFIER_TASK_WEIGHTS = {
        'trading': 4.0, 'regime': 0.0, 'risk': 0.0,
        'momentum': 0.0, 'flow': 0.0, 'profit': 3.0,
    }
