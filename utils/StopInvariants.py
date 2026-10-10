"""Runtime stop invariants (feature 023, research R9, FR-017).

The stop path has shipped real defects that every run "passed": the post-grace rescale returned a
POSITIVE ratio, which freqtrade's `abs()` turned into a far LOOSER stop (fixed 09db623; a -4% stop
became -46%). Those were found indirectly, through P&L. This mixin makes the defect classes halt
the run instead.

Mix it in AHEAD of the strategy base so it wraps the real `custom_stoploss`:

    class MyStrategy(StopInvariantMixin, BaseStrategy): ...

It returns the parent's value unchanged when every check passes, so it cannot alter a trade.

CHECKS (a violation raises StopInvariantError):
  (a) sign      -- a returned ratio in (0, 1) is the sign-flip defect. 1.0 (and above) is the
                   framework's deliberate "no change" sentinel; None is freqtrade's.
  (b) price     -- |ratio| <= 1: freqtrade places the stop at current*(1-|ratio|), so a larger
                   magnitude is a negative stop price.
  (c) bound     -- once grace has expired (or when no grace is configured), the implied stop
                   PRICE must be >= open_rate*(1+floor), floor = min(atr_stoploss_floor, stoploss)
                   when the ATR stop is adaptive (it falls back to `stoploss` without ATR), else
                   `stoploss`. Checked in PRICE space, never on the ratio:
                   the ratio is relative to current_rate, so a correct entry-anchored stop on a
                   winning trade has a ratio more negative than the floor.
  (d) direction -- the first stop placed after grace must not sit BELOW the last stop placed
                   during grace while the price is still above that earlier stop.

EXEMPT: `STOP_AT_MARKET_RATIO`, the deliberate "fire now" clamp for a trade already through its
post-grace stop -- its implied price is the market, which is below the bound by construction.

Strategy-agnostic: no family names. Per-trade state (the last stop price) is kept on the
instance, keyed by trade id.
"""

from __future__ import annotations

from utils.ExitPolicy import STOP_AT_MARKET_RATIO


_TOL = 1e-9


class StopInvariantError(RuntimeError):
    """A stop value that violates a documented invariant. Halts the run (FR-017)."""


def _num(v, default=0.0) -> float:
    v = getattr(v, "value", v)  # hyperopt Parameter or plain float
    return float(default if v is None else v)


class StopInvariantMixin:
    def custom_stoploss(
        self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs
    ):
        value = super().custom_stoploss(
            pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs
        )
        self._check_stop(pair, trade, current_time, current_rate, value)
        return value

    def _check_stop(self, pair, trade, current_time, current_rate, value) -> None:
        if value is None or value >= 1.0 or value == STOP_AT_MARKET_RATIO:
            return
        tid = getattr(trade, "id", id(trade))
        where = f"{pair} trade {tid} at {current_time}"
        if value > 0.0:
            raise StopInvariantError(
                f"stop sign: custom_stoploss returned {value!r} > 0 for {where}. freqtrade "
                f"applies abs(), so this LOOSENS the stop (the 09db623 defect class)."
            )
        if abs(value) > 1.0 + _TOL:
            raise StopInvariantError(
                f"stop price: |{value!r}| > 1 implies a negative stop price for {where}."
            )
        stop_price = current_rate * (1.0 - abs(value))

        grace_hours = _num(getattr(self, "stoploss_grace_hours", 0.0))
        age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        in_grace = grace_hours > 0.0 and age_h < grace_hours

        last = getattr(self, "_si_last_stop", None)
        if last is None:
            last = self._si_last_stop = {}

        if not in_grace:
            adaptive = bool(getattr(self, "use_atr_adaptive_stoploss", False))
            static = _num(getattr(self, "stoploss", -1.0), default=-1.0)
            # Adaptive: the ATR stop is clamped to [floor, cap], but `_resolve_normal_stop`
            # FALLS BACK to the static stoploss when ATR is unavailable -- so the documented
            # bound is the looser of the two, not the floor alone.
            floor = (
                min(_num(getattr(self, "atr_stoploss_floor", static), default=static), static)
                if adaptive
                else static
            )
            bound = trade.open_rate * (1.0 + floor)
            if stop_price < bound * (1.0 - _TOL):
                raise StopInvariantError(
                    f"stop bound: implied stop price {stop_price:.8g} is below "
                    f"open_rate*(1+{floor}) = {bound:.8g} for {where} (ratio {value!r}, "
                    f"current {current_rate})."
                )
            prev = last.get(tid)
            if (
                prev is not None
                and prev[1]
                and current_rate > prev[0]
                and stop_price < prev[0] * (1.0 - _TOL)
            ):
                raise StopInvariantError(
                    f"stop direction: first post-grace stop {stop_price:.8g} is below the "
                    f"in-grace stop {prev[0]:.8g} while price {current_rate} is above it, "
                    f"for {where}."
                )
        last[tid] = (stop_price, in_grace)
