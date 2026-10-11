"""The post-grace stop sign defect (FIXED 2026-10-08) and its live consequence.

THE DEFECT. `BaseStrategy.custom_stoploss`'s post-grace branch returned
`(open_rate * (1 + normal) / current_rate) - 1`, rescaling the intended entry-anchored
stop into the ratio-relative-to-current_rate that freqtrade's return convention wants.
That rescale is sign-valid only while `current_rate >= open_rate * (1 + normal)`. Below
it the expression is POSITIVE, and `Trade.adjust_stop_loss` applies
`new_loss = current_price * (1 - abs(stoploss / leverage))` (`trade_model.py:862`) -- so
the `abs()` turned the overshoot into a LOOSER stop rather than a tighter one, scaled by
how far underwater the trade was.

WHY THE EXISTING FIXTURE MISSED IT. `test_exit_policy_characterisation.py` samples 1,440
real-data cases across ages 0/3/6.4/6.6/48h, and every one of them passed both before and
after this fix -- none of its sampled bars happened to be post-grace AND more than
|normal| underwater. A characterisation fixture pins the behaviour it happens to visit;
it is not a substitute for a test aimed at a branch.

HOW IT WAS FOUND. Feature 021's reconciliation, as 28 of the 42 remaining table/arm
reason disagreements: the offline table applied the -4% post-grace stop as specified and
exited at ~29-31 bars (7.3-7.8h, just past the 6.5h grace window), while the live arm
loosened and ran on.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.ExitPolicy import STOP_AT_MARKET_RATIO  # noqa: E402

ENTRY = 100.0
NORMAL = -0.04          # the post-grace stop this stub resolves to
GRACE_HOURS = 6.5
GRACE_LEVEL = -0.27


class _Stub:
    """Only what `custom_stoploss`'s grace branch touches."""

    def __init__(self):
        self.stoploss_grace_hours = GRACE_HOURS
        self.stoploss_grace_level = GRACE_LEVEL
        self.stoploss = NORMAL
        self.use_atr_adaptive_stoploss = False
        self._resolve_normal_stop = types.MethodType(lambda self, pair, trade=None: NORMAL, self)


class _Trade:
    def __init__(self, age_hours: float):
        self.open_rate = ENTRY
        self.open_date_utc = datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(
            hours=age_hours
        )
        self.pair = "SOL/USDT"


def _call(current_rate: float, age_hours: float = 8.0) -> float:
    return BaseStrategy.custom_stoploss(
        _Stub(), pair="SOL/USDT", trade=_Trade(age_hours),
        current_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        current_rate=current_rate, current_profit=current_rate / ENTRY - 1.0,
        after_fill=False,
    )


def _freqtrade_stop_price(current_price: float, ratio: float) -> float:
    """`Trade.adjust_stop_loss`'s arithmetic for a long, leverage 1 (`trade_model.py:862`),
    written out so this test measures the stop PRICE the strategy actually causes rather
    than the ratio it returns -- the ratio alone is what made the defect invisible."""
    return current_price * (1.0 - abs(ratio))


# ---------------------------------------------------------------------------------------
# Unchanged: while the market is at or above the intended stop, the rescale is sign-valid.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("current_rate", [97.0, 100.0, 105.0, 150.0])
def test_above_the_stop_the_rescale_still_yields_exactly_the_entry_anchored_stop(current_rate):
    """The whole point of the rescale: an entry-anchored, NON-trailing stop. A winner at
    150 must still stop at 96, not at 144."""
    ratio = _call(current_rate)
    assert _freqtrade_stop_price(current_rate, ratio) == pytest.approx(
        ENTRY * (1.0 + NORMAL), rel=1e-12
    )


def test_exactly_at_the_stop_level_takes_the_at_market_clamp():
    """The boundary, and a second quiet defect it exposes. At `current_rate` exactly equal
    to the intended stop the rescale yields exactly 0.0 -- and `interface.py` discards a
    falsy custom stoploss (`if stop_loss_value_custom and ...`), so pre-fix this case left
    the stop UNTOUCHED at the grace level rather than tightening to it. freqtrade cannot
    represent a stop at exactly the market either (it would need `abs(ratio) == 0`), so
    the at-market clamp is the only representable outcome, and the correct one."""
    ratio = _call(ENTRY * (1.0 + NORMAL))
    assert ratio == STOP_AT_MARKET_RATIO
    assert _freqtrade_stop_price(ENTRY * (1.0 + NORMAL), ratio) < ENTRY * (1.0 + NORMAL)


# ---------------------------------------------------------------------------------------
# FIXED: below the intended stop, the stop must TIGHTEN to the market, never loosen.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "current_rate,buggy_stop_pct",
    [(95.0, -0.06), (90.0, -0.16), (80.0, -0.36), (75.0, -0.46)],
)
def test_below_the_stop_the_stop_no_longer_LOOSENS(current_rate, buggy_stop_pct):
    """The regression test proper. `buggy_stop_pct` is what the pre-fix code produced,
    measured: a trade 10% underwater got -16% and one 25% underwater got -46%, against
    the -4% the policy asks for. The fix must not reproduce those."""
    ratio = _call(current_rate)
    stop_price = _freqtrade_stop_price(current_rate, ratio)
    buggy_price = ENTRY * (1.0 + buggy_stop_pct)
    assert stop_price > buggy_price, (
        f"stop {stop_price / ENTRY - 1:+.2%} is no tighter than the pre-fix "
        f"{buggy_stop_pct:+.0%}"
    )
    # ... and it sits essentially AT the market, so the stop fires on this bar.
    assert stop_price == pytest.approx(current_rate, rel=1e-5)
    assert stop_price <= current_rate, "freqtrade cannot hold a stop above the market"


def test_the_returned_ratio_is_never_positive_post_grace():
    """A positive return value is the defect's signature, because `abs()` silently
    reinterprets it as a loosening. Swept across and through the stop level."""
    for rate in [150.0, 105.0, 100.0, 96.1, 96.0, 95.9, 90.0, 75.0, 50.0, 1.0]:
        assert _call(rate) <= 0.0, f"positive ratio returned at current_rate={rate}"


def test_at_market_clamp_is_truthy_so_freqtrade_does_not_discard_it():
    """`interface.py` ignores a custom stoploss that is falsy
    (`if stop_loss_value_custom and not (isnan or isinf)`), so a clamp of exactly 0.0
    would be silently dropped and the stop left untouched -- the defect again, quietly."""
    assert STOP_AT_MARKET_RATIO
    assert STOP_AT_MARKET_RATIO < 0.0


# ---------------------------------------------------------------------------------------
# The grace window itself is untouched by this fix.
# ---------------------------------------------------------------------------------------

def test_inside_the_grace_window_nothing_changed():
    """During grace the branch returns `grace_level` on the after-fill call and 1.0 (no
    change) otherwise, whatever the price -- the fix must not reach into it."""
    assert _call(75.0, age_hours=1.0) == 1.0
    assert _call(150.0, age_hours=1.0) == 1.0
    out = BaseStrategy.custom_stoploss(
        _Stub(), pair="SOL/USDT", trade=_Trade(1.0),
        current_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        current_rate=75.0, current_profit=-0.25, after_fill=True,
    )
    assert out == GRACE_LEVEL
