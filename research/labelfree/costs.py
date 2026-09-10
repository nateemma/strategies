"""Cost model: fee, estimated half-spread, volatility-scaled slippage (T006).

FR-004   fee per side at the real tier + half-spread + vol-scaled slippage, inside the
         training objective; training cost scale gamma is swept, evaluation is full cost
FR-004a  the data is OHLCV only (no bid/ask), so half-spread is ESTIMATED per bar from
         high/low with a conservative floor. A flat constant is forbidden: it under-costs
         illiquid pairs by the widest margin, exactly where this repo has twice found
         phantom edges (research.md R2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4
BINANCEUS_TAKER = 10.0 * BPS      # ~10 bp per side
SPREAD_FLOOR_BPS = 1.0            # per side; estimate may not collapse to zero
SLIPPAGE_K = 0.10                 # slippage = K * bar volatility


@dataclass(frozen=True)
class CostModel:
    fee_per_side: float = BINANCEUS_TAKER
    spread_floor_bps: float = SPREAD_FLOOR_BPS
    slippage_k: float = SLIPPAGE_K
    estimator: str = "abdi_ranaldo"     # or "corwin_schultz" (sensitivity path)
    gamma: float = 1.0                  # training cost scale; evaluation pins 1.0+

    def half_spread(self, df: pd.DataFrame) -> pd.Series:
        s = (abdi_ranaldo(df) if self.estimator == "abdi_ranaldo"
             else corwin_schultz(df))
        return (s / 2.0).clip(lower=self.spread_floor_bps * BPS)

    def slippage(self, df: pd.DataFrame) -> pd.Series:
        rng = (np.log(df["high"]) - np.log(df["low"])).clip(lower=0.0)
        return (self.slippage_k * rng).fillna(0.0)

    def per_unit(self, df: pd.DataFrame, *, evaluation: bool = False) -> pd.Series:
        """Total cost charged per unit of |executed exposure change|."""
        g = 1.0 if evaluation else self.gamma
        return g * (self.fee_per_side + self.half_spread(df) + self.slippage(df))

    def describe(self) -> dict:
        return {
            "fee_per_side_bps": self.fee_per_side / BPS,
            "estimator": self.estimator,
            "spread_floor_bps": self.spread_floor_bps,
            "slippage_k": self.slippage_k,
            "gamma": self.gamma,
        }


def abdi_ranaldo(df: pd.DataFrame, window: int = 96) -> pd.Series:
    """Abdi & Ranaldo (2017) close-high-low spread estimator, as a fraction of price.

    s = 2 * sqrt( E[ (c_t - eta_t) * (c_t - eta_{t+1}) ] ), eta = midpoint of log high/low.
    Negative expectations are truncated to zero (the estimator is not sign-constrained);
    the caller's floor then applies.
    """
    c = np.log(df["close"])
    eta = (np.log(df["high"]) + np.log(df["low"])) / 2.0
    x = (c - eta) * (c - eta.shift(-1))
    m = x.rolling(window, min_periods=max(8, window // 8)).mean()
    return (2.0 * np.sqrt(m.clip(lower=0.0))).fillna(0.0)


def corwin_schultz(df: pd.DataFrame) -> pd.Series:
    """Corwin & Schultz (2012) two-period high-low estimator. Sensitivity path only.

    Frequently returns negative values, which are truncated; that truncation is a
    material part of its behaviour and is why Abdi-Ranaldo is the primary (research R2).
    """
    h, l = np.log(df["high"]), np.log(df["low"])
    beta = ((h - l) ** 2).rolling(2).sum()
    h2 = np.log(df["high"].rolling(2).max())
    l2 = np.log(df["low"].rolling(2).min())
    gamma = (h2 - l2) ** 2
    k = 3 - 2 * np.sqrt(2.0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0.0)
    return (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).fillna(0.0)
