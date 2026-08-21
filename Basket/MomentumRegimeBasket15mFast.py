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

Config: config/config_mom_15m.json (same as the base).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from MomentumRegimeBasket15m import MomentumRegimeBasket15m


class MomentumRegimeBasket15mFast(MomentumRegimeBasket15m):
    MOM_LOOKBACK_DAYS = 14   # 90 -> 14: rotate the basket on 2-week momentum
