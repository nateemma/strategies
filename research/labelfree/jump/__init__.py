"""Online sparse jump-model regime gate (feature 016, brief 08).

This package is the study's LAST brief and an explicit NEGATIVE CONTROL. `regime/VERDICT.md`
already tested eight regime instruments and returned FAIL; a jump model differs mechanically
from all of them only in that it penalises SWITCHING, which is the one property Phase C never
sampled.

Two facts shape every module here.

1. THE PACKAGE'S DEFAULT PREDICTOR IS THE LEAK. `jumpmodels.jump.JumpModel.predict()` defaults
   to `use_viterbi=True` -- a backward traceback, so a bar's state depends on bars after it.
   Only `predict_online()` does the forward-only `value_mx.argmin(axis=1)`. Audited 2026-09-15
   at site-packages/jumpmodels/jump.py:135-139 (traceback) and :561 (online). Reaching for the
   obvious method would have reintroduced exactly the leak this feature exists to measure.

2. THE FIT IS STOCHASTIC EVEN THOUGH THE BACKTESTS ARE NOT. Gated arms reuse the ungated fold
   weights and train nothing, so the backtests are deterministic -- and that is precisely what
   makes the gap easy to miss. `fit()` starts from a k-means++ initialisation, so every blocked
   bar and every drawdown delta inherits that randomness. Hence `fit_ensemble` (FR-022) and the
   cross-seed agreement floor (FR-023). Recorded precedent: multi-task head weights once moved
   entries LESS than a random seed did, which is what closed that line.
"""
from __future__ import annotations

import sys
from pathlib import Path

JUMPMODELS_VERSION = "0.1.1"          # pinned; cross-check only, never the primary verdict

OUT = Path(__file__).resolve().parent / "out"

# Thresholds are pre-registered in spec.md. They live here so no module can quietly hold a
# different one, and so a reader can see every gate trigger in one place.
LEAK_FLOOR_PP = 7.633          # FR-009  evaluation.bootstrap.BASELINE_HALF_WIDTH_PP
AGREEMENT_FLOOR = 0.95         # FR-016  in-repo vs jumpmodels
SEED_AGREEMENT_FLOOR = 0.95    # FR-023  across seeds
NOVEL_BLOCK_FLOOR = 0.20       # FR-024  vs the incumbent's rvol guard
RETURN_FLOOR_PP_YR = 3.350     # SC-011  "unchanged within the floor"
MIN_ACTIVE_FOLDS = 10          # SC-020  of 15
SEEDS = (0, 1, 2, 3, 4)        # FR-022  >= 5

K_STATES = 2                   # pre-registered primary; K=3 is a labelled sensitivity only
MIN_STATE_SHARE = 0.10         # degeneracy guard on lam -- see model.calibrate_lam


def assert_interpreter() -> None:
    """FR-017 -- the pyenv shim is a DIFFERENT 3.12.13 carrying no pandas and no scipy.

    Recorded failure: `cma` was installed into the pyenv base by mistake and the import
    succeeded from one interpreter while every data module failed from the other. Verified
    2026-09-15: the pyenv shim reports `No module named 'pandas'`.

    NOT `Path(sys.executable).resolve()`: the venv's `bin/python` is a SYMLINK to the pyenv
    binary, so resolving it lands outside the venv and the check fails on the correct
    interpreter. `sys.prefix` is the venv root and is not resolved away, while `sys.base_prefix`
    stays the pyenv install -- their inequality is what "inside a venv" actually means.
    """
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            f"not running inside a virtualenv: {sys.executable}\n"
            f"(prefix {sys.prefix})\n"
            f"use /Users/philprice95/projects/freqtrade/.venv/bin/python")
    if "freqtrade/.venv" not in sys.prefix:
        raise SystemExit(
            f"wrong virtualenv: {sys.prefix}\n"
            f"use /Users/philprice95/projects/freqtrade/.venv/bin/python")


assert_interpreter()
