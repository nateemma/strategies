"""Cost-budget stake cap (feature 036 US3; contracts/cost_budget_stake.md).

OFF by default (`STAKE_COST_BUDGET_BPS = None`). In that case nothing changes:
`custom_stake_amount`, `confirm_trade_entry` and `_fillable_stake` are exactly the parent's.

When ON, the 1/10-of-candle-volume rule is replaced by the largest stake whose estimated entry
cost, from the committed cost model (research/live_pnl/cost_model.json), stays within the budget
in bps. That stake is S_max = max_participation(budget) × rolling 24 h quote volume (or the
per-pair stake curve's inverse).

It works by making BaseStrategy's own headroom equal S_max, so that every intermediate override
(e.g. FamilyCoexistenceMixin's pair cap) still runs unchanged:
- `_fillable_stake` returns (quote_volume, S_max), so `custom_stake_amount` gives
  min(proposed, S_max);
- `confirm_trade_entry` runs the parent with QUOTE_VOLUME_HEADROOM_MULT = qv / S_max for that call,
  so its check `qv >= max(MIN_QUOTE_VOLUME, mult × size)` becomes size <= S_max. The
  MIN_QUOTE_VOLUME floor is kept.

Pairs the cost model excludes keep the original 1/10 rule; re-statement drops them.
"""

from __future__ import annotations

import json
from pathlib import Path


class CostBudgetStakeMixin:
    STAKE_COST_BUDGET_BPS: float | None = None
    STAKE_COST_MODEL: str | None = None
    STAKE_COST_MODEL_SHA: str | None = None
    MODEL_KEY_IGNORED = ("STAKE_COST_BUDGET_BPS", "STAKE_COST_MODEL", "STAKE_COST_MODEL_SHA")

    _cb_model = None

    def _cb_on(self) -> bool:
        return self.STAKE_COST_BUDGET_BPS is not None

    def _cb(self) -> dict:
        if self._cb_model is None:
            import hashlib

            if not self.STAKE_COST_MODEL or not Path(self.STAKE_COST_MODEL).is_file():
                raise ValueError(f"{type(self).__name__}: cost model {self.STAKE_COST_MODEL!r} missing")
            blob = Path(self.STAKE_COST_MODEL).read_bytes()
            sha = hashlib.sha256(blob).hexdigest()[:16]
            if self.STAKE_COST_MODEL_SHA and sha != self.STAKE_COST_MODEL_SHA:
                raise ValueError(f"{type(self).__name__}: cost model sha {sha} != declared "
                                 f"{self.STAKE_COST_MODEL_SHA}")
            type(self)._cb_model = json.loads(blob)
        return self._cb_model

    def _cb_smax(self, pair: str, dataframe) -> float | None:
        """S_max for the last analysed candle, or None for a pair the model excludes."""
        from research.live_pnl import depth as D

        m = self._cb()
        if pair in m["excluded_pairs"]:
            return None
        if pair not in m["tiers"]:
            raise ValueError(f"{type(self).__name__}: no cost tier for {pair} (no per-pair default)")
        budget = float(self.STAKE_COST_BUDGET_BPS)
        if m["choice"] == "participation":
            qv = (dataframe["close"] * dataframe["volume"]).tail(96)
            if len(qv) < 96:
                return 0.0
            curve = D.MonotoneCurve.from_json(m["curves"][f"{m['tiers'][pair]}|buy"])
            return curve.max_participation(budget) * float(qv.sum())
        curve = D.MonotoneCurve.from_json(m["per_pair"][f"{pair}|buy"])
        return curve.max_participation(budget)  # x = stake for per-pair curves

    def _fillable_stake(self, pair: str) -> tuple:
        qv, fillable = super()._fillable_stake(pair)
        if not self._cb_on():
            return qv, fillable
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        smax = self._cb_smax(pair, dataframe)
        return (qv, fillable) if smax is None else (qv, smax)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, *args, **kwargs):
        if not self._cb_on() or self.dp.runmode.value in ("plot", "other"):
            return super().custom_stake_amount(pair, current_time, current_rate, proposed_stake,
                                               *args, **kwargs)
        _, smax = self._fillable_stake(pair)
        return min(proposed_stake, smax)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag, side, **kwargs):
        if not self._cb_on() or self.dp.runmode.value in ("plot", "other"):
            return super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force,
                                               current_time, entry_tag, side, **kwargs)
        qv, smax = self._fillable_stake(pair)
        had = "QUOTE_VOLUME_HEADROOM_MULT" in self.__dict__
        old = self.__dict__.get("QUOTE_VOLUME_HEADROOM_MULT")
        self.QUOTE_VOLUME_HEADROOM_MULT = float("inf") if smax <= 0 else qv / smax
        try:
            return super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force,
                                               current_time, entry_tag, side, **kwargs)
        finally:
            if had:
                self.QUOTE_VOLUME_HEADROOM_MULT = old
            else:
                del self.QUOTE_VOLUME_HEADROOM_MULT
