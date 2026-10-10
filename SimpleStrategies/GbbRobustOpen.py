# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GbbRobustOpen — the `gbb` trigger with ONE STATIC guard config, chosen leakage-free.

WHY THIS ARM EXISTS. Feature 022's gate returned `indeterminate` for per-fold tuning and the
two-state scheme degenerated to a constant, so the only remaining guard question is WHICH
CONSTANT. That is not an adaptive scheme, so Principle VII's gate does not apply to it — and
the values live here in the class rather than in `run_folds.py`'s `guards` plan key, which is
why `_require_trackable_verdict` is not involved. The gate blocks adaptivity; a fixed
configuration is an ordinary static arm like A and B.

HOW THE VALUES WERE CHOSEN, and the mistake that preceded them. The grid's maximin
("the safe middle range") was first computed over TEST-window scores, which is a HINDSIGHT
selection — the same flaw that made arm B's +18.76pp 97.8% in-sample. Re-selected over TRAIN
windows only, both maximin AND max-mean pick:

    entry_atr_pct 0.013   entry_adx_threshold 20.0   entry_rvol_threshold 1.9

The other two quality thresholds stay OPEN, because the grid never varied them; this arm
therefore differs from arm B (`GbbGuardedOpen`), which also tuned `guard_metric` 0.1,
`close_norm` 0.6 and `bb_width` 0.094.

WHAT THE PROXY PREDICTS, recorded before the run so the comparison is honest. Out of sample
this configuration scored: measurable in 10 of 15 folds, mean proxy_score +0.00467, **worst
fold -0.02457**. The current all-open default scored 15/15, mean +0.00046, worst -0.00225. So
the proxy expects HIGHER MEAN and a ~10x WORSE WORST FOLD, with trading in a third fewer folds.
Those are proxy scores and not returns (FR-018); this arm is what turns them into a return.

THE QUESTION IT SETTLES. Whether `BaseStrategy`'s guard defaults should change. Grounds would
require beating arm A (`GbbOpen`, +2.64pp, 7/15) by more than the harness's 3-4pp/yr floor with
a paired interval excluding zero. A higher mean bought with a 10x worse worst fold is not
grounds, and the 2026-09-06 decision to open the guards was taken deliberately "for headroom
and measurability".

EXIT: the Open operating point, inherited unchanged from TriggerOnlyStrategy (FR-009), so this
arm is comparable exit-for-exit to A, B and D by `exit_config_hash` equality.
NO MODEL is fitted (FR-020).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class GbbRobustOpen(TriggerOnlyStrategy):
    """`gbb` + the TRAIN-selected static guard configuration, no model."""

    TRIGGER_NAME = "gbb"

    #: Selected on training windows only — maximin and max-mean agree. `entry_enable_guards`
    #: stays True because it is what applies the quality thresholds at all.
    buy_params = {
        **TriggerOnlyStrategy.buy_params,
        "entry_enable_guards": True,
        "entry_atr_pct": 0.013,
        "entry_adx_threshold": 20.0,
        "entry_rvol_threshold": 1.9,
    }
