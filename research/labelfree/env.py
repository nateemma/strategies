"""Pooled trading environment (feature 012, brief 04). Gymnasium-shaped, numpy economics.

WHY NUMPY AND NOT sim.net_returns DIRECTLY (research R1). FR-001 says reuse brief 1's
simulator as the step function, but `sim.net_returns` is vectorised over a COMPLETE exposure
path and built on MLX, while this env is called once per bar ~10^8 times inside SB3's CPU
rollout loop across worker processes. The economics separate cleanly per bar, so the numpy
step below is a faithful restatement -- and "faithful" is exactly the claim this study keeps
getting burned by, so `rl/equivalence.py` proves it BIT-FOR-BIT and is a blocking gate.

THE INVARIANT (research R2), one index symbol feature-wide:

    the action taken on the observation ENDING AT BAR i
    earns the return between open[i+1] and open[i+2]

so an episode of T decisions consumes T+2 bars, and the last two bars of a window are
consumable as outcome but never as decision points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gymnasium as gym                                      # noqa: E402
from gymnasium import spaces                                 # noqa: E402

from research.labelfree import rl as RL                      # noqa: E402
from research.labelfree.rl.rewards import RewardSpec, realised_reward  # noqa: E402


def step_economics(w_i: float, w_prev: float, bar_return: float,
                   cost: float, quote_volume: float | None) -> tuple[float, float, float]:
    """One bar, indexed by i. Returns (reward, requested, executed).

    Mirrors sim.net_returns exactly:
        requested = |w_i - w_{i-1}|            (w_{-1} = 0)
        executed  = min(requested, LIQUIDITY_FRACTION * quote_volume_i)   [hard cap]
        reward    = w_i * bar_return - cost_i * executed

    Costs are charged on EXECUTED, not requested: the liquidity cap is not a free brake.
    """
    requested = abs(w_i - w_prev)
    if quote_volume is None:
        executed = requested
    else:
        executed = min(requested, RL.LIQUIDITY_FRACTION * quote_volume)
    return w_i * bar_return - cost * executed, requested, executed


def bar_returns_from_open(open_px: np.ndarray) -> np.ndarray:
    """log(open[i+2]) - log(open[i+1]) — the R2 invariant, matching sim.bar_returns."""
    o = np.asarray(open_px, dtype=float)
    return np.log(o[2:]) - np.log(o[1:-1])


class PooledTradingEnv(gym.Env):
    """One (pair, window) episode. A shared policy sees many of these across all pairs."""

    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, open_px: np.ndarray, cost: np.ndarray,
                 quote_volume: np.ndarray, *, reward_spec: RewardSpec | None = None,
                 max_drawdown: float = RL.MAX_EPISODE_DRAWDOWN,
                 continuous: bool = False, band: float = 0.0):
        super().__init__()
        n = len(open_px) - RL.EXEC_LAG_BARS          # decisions available under the invariant
        if n <= 0:
            raise ValueError(
                f"window of {len(open_px)} bars yields no decisions: the reward at bar i needs "
                f"open[i+1] and open[i+2], so a window must exceed {RL.EXEC_LAG_BARS} bars")
        self.n = n
        self.features = np.asarray(features, dtype=np.float32)[:n]
        self.r = bar_returns_from_open(open_px)[:n]
        self.cost = np.asarray(cost, dtype=float)[:n]
        self.vol = np.asarray(quote_volume, dtype=float)[:n]
        self.reward_spec = reward_spec or realised_reward()
        self.max_drawdown = max_drawdown
        self.continuous = continuous
        self.band = band

        n_feat = self.features.shape[1] if self.features.ndim == 2 else 1
        # Observation = features + the position state the brief says FreqAI cannot backtest.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_feat + 3,),
                                            dtype=np.float32)
        self.action_space = (spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
                             if continuous else spaces.Discrete(2))
        self._reset_state()

    def _reset_state(self):
        self.i = 0
        self.w = 0.0
        self.bars_in_trade = 0
        self.entry_ref = 0.0
        self.equity = 0.0
        self.peak = 0.0
        self.turnover = 0.0

    def _obs(self) -> np.ndarray:
        f = self.features[self.i]
        f = np.atleast_1d(f).astype(np.float32)
        unreal = self.equity - self.entry_ref if self.w > 0 else 0.0
        return np.concatenate([f, np.array([self.w, self.bars_in_trade / 100.0, unreal],
                                           dtype=np.float32)])

    def action_masks(self) -> np.ndarray:
        """MaskablePPO hook. Long-only is enforced by MASKING, never by penalising (FR-004):
        a penalty is a reward-shaping term and FR-003 forbids those."""
        # index 0 = flat, index 1 = long. Both always legal for a long-only target exposure;
        # "sell" is not a separate action, it is a move to flat, and is invalid when flat.
        return np.array([True, True]) if self.w > 0 else np.array([True, True])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        w_prev = self.w
        if self.continuous:
            target = float(np.clip(np.asarray(action).ravel()[0], 0.0, 1.0))
            # FIXED no-trade band (research R4): hold inside it. Brief 02's bimodality came
            # from a LEARNED band finding one of two fixed points; a fixed band has no such
            # interaction and no attractor to fall into.
            w_i = w_prev if abs(target - w_prev) < self.band else target
        else:
            w_i = float(int(np.asarray(action).ravel()[0]))

        reward, requested, executed = step_economics(
            w_i, w_prev, float(self.r[self.i]), float(self.cost[self.i]),
            float(self.vol[self.i]))
        reward = self.reward_spec.transform(
            reward, w_i=w_i, w_prev=w_prev, executed=executed,
            bar_return=float(self.r[self.i]), bars_in_trade=self.bars_in_trade)

        if not np.isfinite(reward):
            raise FloatingPointError(                      # FR-007
                f"non-finite reward at bar {self.i}: w={w_i} r={self.r[self.i]} "
                f"cost={self.cost[self.i]} executed={executed}")

        self.turnover += executed
        self.equity += reward
        self.peak = max(self.peak, self.equity)
        if w_prev == 0.0 and w_i > 0.0:
            self.entry_ref, self.bars_in_trade = self.equity, 0
        elif w_i > 0.0:
            self.bars_in_trade += 1
        else:
            self.bars_in_trade = 0
        self.w = w_i
        self.i += 1

        # terminated (drawdown) and truncated (window exhausted) are DISTINCT: PPO bootstraps
        # a truncated episode's value and must not bootstrap a terminated one's (research R6).
        terminated = bool(self.peak - self.equity > self.max_drawdown)
        truncated = bool(self.i >= self.n) and not terminated
        obs = self._obs() if self.i < self.n else np.zeros(self.observation_space.shape,
                                                           dtype=np.float32)
        return obs, float(reward), terminated, truncated, {
            "w": w_i, "executed": executed, "turnover": self.turnover, "equity": self.equity}
