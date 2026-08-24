"""MomentumRegimeBasket15mFast — faster-rotation variant of MomentumRegimeBasket15m.

Identical mechanics (BTC>SMA100 daily regime, per-coin SMA50 trend filter, TOP_N=3,
hourly rebalance on 15m data, accumulating liquidity-capped fills) — the ONLY change
is the momentum ranking window: 14 days instead of 90. A 2-week ranking rotates the
basket much faster, so it trades at a higher pace when risk-on.

Vectorized lookback sweep (2026-08, daily close-to-close, 20bps/turn, keeping SMA100):
  lb=90 (base): +27% total, DD -69%, 151 rebalances
  lb=14 (this): +338% total, DD -65%, 252 rebalances
BUT that sweep has NO fill/liquidity model, and a 14d ranking is much LESS sticky than
90d — the accumulation mechanic (fill a little each candle while a coin stays top-N) has
less runway, which is exactly what made the naive fast strategy fail (the "-17% fill
artifact"). So lb=14 is the most fill-optimistic cell; this freqtrade run — with the
per-candle liquidity cap and next-candle fills — is the honest execution test.

*** OPEN DECISION (2026-08-23) -- CONSIDER MOVING TO lb=21. REVISIT THIS. ***
Cross-regime sweep on history back to 2021 (EXIT_RANK_N=9, FILL_VOLUME_LAG=0),
total return % with rank in brackets:

    lb     P1 2021-05..2022-12   P2 2023-01..2024-08   P3 2024-09..2026-08
     7        +21.7 (6)             +73.9 (6)            +243.7 (4)
    14        +35.6 (5)            +166.2 (2)            +834.5 (1)   <- this class
    21        +95.8 (1)            +186.0 (1)            +345.7 (2)
    30        +76.3 (2)            +163.7 (3)            +189.2 (5)
    60        +48.1 (4)             +87.9 (5)             +94.4 (6)
    90        +67.2 (3)            +118.6 (4)            +263.8 (3)

lb=14 ranks FIFTH OF SIX in P1 -- its dominance is a P3 phenomenon (2.4x the
runner-up there). lb=21 is 1st/1st/2nd, the only value strong in every regime.

The "persistence validated" claim for lb=14 came from splitting 2024-05..2026-08
in half; with the longer history that whole range is ONE regime, so both halves
shared it and the study could not have detected this. lb=14 is NOT validated
cross-regime.

NOT changed yet because it is a genuine trade-off, not a bug: lb=14 has far more
recent-regime upside (P3 ret/DD 26.2 vs lb=21's 7.8), so switching buys
cross-regime consistency at the cost of P3 performance. Deciding requires a view
on whether the current regime persists.

TO ACT ON IT: set MOM_LOOKBACK_DAYS = 21 below, re-run the three windows, and
re-check the EXIT_RANK_N default -- the two interact (the sweep above fixed
EXIT_RANK_N=9, which was itself chosen with lb=14). Tune one at a time.

*** SECOND OPEN DECISION: EXIT_RANK_N 9 vs 15. ***
Re-swept under capped exits (now the default). At this class's lb=14, worst-of-3
windows: N=9 gives +38.8%, N=15 gives +112.5% -- ~3x better, and positive in all
three windows. It replicates under free exits (+112.3%), so it is not an artifact
of the execution model. For N=9: better in the deployment regime (P3 +401.7 vs
+313.6), trades more (102 vs 85 in P3, so less sample noise), and per-window
hyperopt picked 9/11/7 with nothing near 15. Same shape of trade-off as the
lookback below -- recent-regime return vs cross-regime robustness. See the base
class docstring for the full table. NOT changed.

Config: config/config_mom_15m.json (same as the base).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from MomentumRegimeBasket15m import MomentumRegimeBasket15m


class MomentumRegimeBasket15mFast(MomentumRegimeBasket15m):
    MOM_LOOKBACK_DAYS = 14   # 90 -> 14: rotate the basket on 2-week momentum
