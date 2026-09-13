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
N_SEEDS = 5                   # protocol section 4: >=5 seeds, IQM, never a best seed
STEPS_PER_FOLD = 1_500_000    # brief 04 "1-2M steps per fold"
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
