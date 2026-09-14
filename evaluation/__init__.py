"""Offline evaluation tooling — how a change to a strategy gets JUDGED.

Distinct from `Framework/`, which is strategy-agnostic RUNTIME code that strategies inherit
from. Nothing here runs inside a backtest; it all runs around one.

  bootstrap.py       the paired bootstrap, one statistic for single-window and pooled arms
  foldplan.py        anchored walk-forward fold geometry, with purge/embargo
  run_folds.py       run an arm across every fold, one strategy per process
  pool.py            pool per-fold trades into one set
  compare_pooled.py  arm vs control on the pooled set, with the admissibility gate
  power.py           what the harness can and cannot resolve

Entry point: README.md. Decision rule, verdict vocabulary and the three traps live there.

PROMOTED from `evaluation/` on 2026-09-14 (docs/WALKFORWARD_TODO.md item 1). It had been
parked inside a finished study's scratch space while becoming the standard way every arm in
this repository gets judged.

THE PINNED CONSTANTS BELOW WERE MEASURED, NOT CHOSEN. Recomputing them silently rebases every
verdict already recorded against them.
"""

BASELINE_HALF_WIDTH_PP = 7.633      # 95% CI half-width, single window, unpaired
BASELINE_PAIRED_PP = 6.1            # half-width when arms are paired on their symmetric difference
BASELINE_N_TRADES = 99
BASELINE_RETURN_PCT = 9.831
BASELINE_CI = (2.383, 17.649)
BASELINE_SD_PP = 3.893

BOOTSTRAP_SEED = 0                  # must match evaluation/bootstrap.py (FR-022)
N_RESAMPLES = 20000

# Honest ceiling (research R2): pooling every fold the data allows, at double the universe, moves
# the floor to ~2.5-2.8pp — a ~2.7x improvement, NOT 10x. Effects of 3-7pp become newly
# measurable; 1-2pp parameter effects remain unresolvable AFTER this ships.
TARGET_HALF_WIDTH_PP = 4.0          # SC-001a — a TARGET, missable; missing it is a RESULT
TARGET_POOLED_TRADES = 350          # SC-002
