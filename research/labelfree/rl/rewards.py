"""Reward specifications (T009, research R5).

TWO SPECS, and only one may ever produce a result.

  realised       the brief's reward: realised net-of-cost economics, nothing else.
  freqai_shaped  FreqAI's shipped shape (+25/entry, -1/idle bar, pnl*100 on exit) -- the
                 POSITIVE CONTROL. It exists to be hacked.

Why the control is a gate and not an experiment (SC-011): an environment whose reward cannot
be SHOWN to be hackable has not been shown to be unhackable. This session has twice produced
a control that passed vacuously -- feature 010's liveness negative control severed nothing on
its first version, and feature 011's planted leak needed a separation check before its
detection threshold meant anything.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSpec:
    name: str
    probe: bool = False          # probe rewards can never produce a reported figure (T010)
    entry_bonus: float = 0.0
    idle_penalty: float = 0.0
    exit_multiplier: float = 1.0

    def transform(self, base: float, *, w_i: float, w_prev: float, executed: float,
                  bar_return: float, bars_in_trade: int) -> float:
        """`base` is the realised economics. `realised` returns it untouched."""
        r = base
        if self.entry_bonus and w_prev == 0.0 and w_i > 0.0:
            r += self.entry_bonus
        if self.idle_penalty and w_i == 0.0:
            r -= self.idle_penalty
        if self.exit_multiplier != 1.0 and w_prev > 0.0 and w_i == 0.0:
            r *= self.exit_multiplier
        return r


def realised_reward() -> RewardSpec:
    """The only spec that may produce a result. No entry bonus, no idle penalty, no
    take-profit multiplier (FR-003)."""
    return RewardSpec(name="realised")


def freqai_shaped_reward() -> RewardSpec:
    """The positive control. Pays for churn exactly as FreqAI's showcase reward does --
    the docs themselves say agents 'cheat' with it."""
    return RewardSpec(name="freqai_shaped", probe=True,
                      entry_bonus=25.0, idle_penalty=1.0, exit_multiplier=100.0)


SPECS = {"realised": realised_reward, "freqai_shaped": freqai_shaped_reward}
