"""RL in a pooled trading environment (feature 012, brief 04).

One shared policy over all whitelist pairs, trained by PPO against the realised economics of
brief 1's simulator. The question is narrow: do path-dependent decisions -- discrete stops,
grace logic, "exit now vs wait one more bar" -- do better when SAMPLED than when SMOOTHED?

THE INVARIANT (research R2). One index symbol, feature-wide:

    the action taken on the observation ENDING AT BAR i
    earns the return between open[i+1] and open[i+2]

An episode of T decisions therefore consumes T+2 bars. Feature 009 lost five bugs to axis
ambiguity, each producing plausible shapes and wrong numbers with no error.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree.sim import LIQUIDITY_FRACTION       # noqa: E402,F401
# ^ imported, never re-declared: a second copy would drift from the simulator the equivalence
#   gate compares against, and the drift would be invisible.

SCREEN_FOLD = "f07"           # brief 04 experiment 2; the only fold "the screening fold" denotes
EXEC_LAG_BARS = 2             # R2: decide at i, fill at i+1, measure to i+2
MIN_EXPOSURE = 0.05           # participation floor, matching research/labelfree/arms.py
# Both values below were set by the learning-curve diagnostic (rl/curve.py, 2026-09-13),
# from TRAINING-RANGE behaviour only -- a budget selection, never a configuration and never a
# test quantity (SC-012).
#
#   reward IQM by budget: -0.1950 (150k) / -0.0951 (500k) / -0.1919 (1M) / -0.1905 (2M)
#                         / -0.1892 (3M)   -> flat from 1M, moving 0.0013 per doubling
#   turnover spread across 3 seeds:        60.8-139.5 / 4.8-102.0 / 58.7-115.9
#                                          / 39.7-150.0 / 38.6-162.5  -> never narrows
#
# So budget is not the binding constraint past 1M and SEEDS are: three seeds disagree by
# 2-4x at every checkpoint and move in OPPOSITE directions (seed 1 rising, seeds 2-3 falling).
# The protocol floor is >=5; 10 is chosen because the diagnostic showed 5 cannot resolve an
# action-space difference against that spread.
N_SEEDS = 10
STEPS_PER_FOLD = 1_000_000    # the curve flattens here; more steps buy 0.0013 per doubling
STEPS_RATIONALE = "rl/curve.py 2026-09-13: reward IQM flat from 1M; seed spread dominates"
EPISODE_DAYS = (30, 90)       # brief 04: episodes of 30-90 days
BARS_PER_DAY = 96             # 15-minute bars

# SC-018 screen bar, PINNED BEFORE THE SCREEN RUNS (analyze H1). "Distinguishable" without a
# threshold is vacuous: two policies differ in some digit almost surely, so a literal reading
# always passes and the gate stops gating.
DISTINGUISHABLE_MIN_BAR_FRACTION = 0.05   # exposure paths must differ on >5% of decision bars

# SC-011 reward-hacking control. A multiple, pinned before the probe runs (analyze M1):
# "turnover rises measurably" passes on a 1% rise.
PROBE_MIN_TURNOVER_RATIO = 2.0

# Drawdown termination, following FreqAI's max_training_drawdown_pct.
MAX_EPISODE_DRAWDOWN = 0.40

BOOTSTRAP_SEED = 0
