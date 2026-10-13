# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T011 (specs/021-label-exit-coordination) — FR-019, "the exit is held fixed", made
checkable.

FR-019 is otherwise a prose header: nothing in the spec or the code raises if the
exit a future outcome table/leaderboard describes quietly drifts from the exit the
strategy actually runs. This test pins the two live sources of truth for the Open
operating point (`NNNC/NNNC_MLX_Open.json`, the pinned JSON the strategy loads,
and `Framework.BaseStrategy`'s class attributes) against the concrete
numbers quoted in FR-019 / contracts/exit-policy.md, and against `ExitParams`'
defaults where those defaults are themselves meant to BE the Open operating point
(see `utils/ExitPolicy.py`'s field comments: `take_profit` and `max_hold_days`
default to the Open values on purpose). A drift in any of these would otherwise
pass silently and every outcome computed from `ExitParams()` would describe an
exit the strategy does not run.

Fields read off their live source rather than hardcoded, per the task:
`atr_stoploss_multiplier` / `atr_stoploss_cap` / `atr_stoploss_floor` come off the
`BaseStrategy` class itself (plain class attributes, not freqtrade Parameters, so
reading the class is reading the live value -- no hyperopt JSON can override them).
The grace / take-profit / hold-cap values are NOT plain class attributes -- they are
freqtrade hyperopt `Parameter`s whose class-level defaults
(`BaseStrategy.sell_params`) are deliberately NOT the Open operating point's values
(e.g. base `cexit_max_days` default is 21, `stoploss_grace_hours` default is 2.6);
the Open point overrides them via `NNNC_MLX_Open.json`, which is the file read here.

`fee_per_side` has no single live source to read: freqtrade takes the fee from the
exchange/pair's trading-fee tier at runtime, not from strategy code, so there is
nothing in `Framework.BaseStrategy` or the JSON to assert it against. Documented
below rather than silently skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.ExitPolicy import ExitParams  # noqa: E402

OPEN_JSON = Path(__file__).resolve().parents[2] / "NNNC" / "NNNC_MLX_Open.json"


def _open_sell_params() -> dict:
    payload = json.loads(OPEN_JSON.read_text())
    return payload["params"]["sell"]


# --------------------------------------------------------------------------------
# atr_stoploss_* -- plain class attributes on BaseStrategy, not hyperopt Parameters,
# so no strategy JSON can override them and reading the class IS reading the live
# value for every family built on BaseStrategy/BaseNNStrategy.
# --------------------------------------------------------------------------------

def test_atr_stoploss_multiplier_matches_live_class_attribute():
    assert BaseStrategy.atr_stoploss_multiplier == 2.5
    assert ExitParams().atr_stoploss_multiplier == BaseStrategy.atr_stoploss_multiplier


def test_atr_stoploss_cap_matches_live_class_attribute():
    assert BaseStrategy.atr_stoploss_cap == -0.02
    assert ExitParams().atr_stoploss_cap == BaseStrategy.atr_stoploss_cap


def test_atr_stoploss_floor_matches_live_class_attribute():
    assert BaseStrategy.atr_stoploss_floor == -0.04
    assert ExitParams().atr_stoploss_floor == BaseStrategy.atr_stoploss_floor


# --------------------------------------------------------------------------------
# grace / take-profit / hold-cap -- the Open operating point's pinned JSON, NOT
# BaseStrategy's class-level hyperopt defaults (which are deliberately different
# values -- see module docstring).
# --------------------------------------------------------------------------------

def test_open_json_exists_and_is_the_pinned_operating_point():
    assert OPEN_JSON.exists(), (
        f"{OPEN_JSON} missing -- this test pins FR-019 against the Open operating "
        "point's own saved params, not a hardcoded guess at them."
    )
    payload = json.loads(OPEN_JSON.read_text())
    assert payload["strategy_name"] == "NNNC_MLX_Open"


def test_take_profit_matches_open_operating_point():
    sell = _open_sell_params()
    assert sell["cexit_take_profit"] == 0.022
    # `ExitParams.take_profit` defaults to the Open value on purpose (see
    # utils/ExitPolicy.py field comment), so the default must still agree with the
    # pinned JSON rather than having quietly drifted from it.
    assert ExitParams().take_profit == sell["cexit_take_profit"]


def test_max_hold_days_matches_open_operating_point():
    sell = _open_sell_params()
    assert sell["cexit_max_days"] == 10
    assert ExitParams().max_hold_days == sell["cexit_max_days"]


def test_grace_hours_matches_open_operating_point():
    sell = _open_sell_params()
    assert sell["stoploss_grace_hours"] == 24.0  # 6.5 before 2026-10-13 (research/nnnc_framework)
    # NOT asserted against ExitParams()'s default (0.0): the dataclass default is
    # BaseStrategy's framework-wide "grace disabled" default, not the Open point's.
    # A table built for the Open point must construct ExitParams with this value
    # explicitly -- which is exactly what this test exists to pin down.
    overridden = ExitParams(stoploss_grace_hours=sell["stoploss_grace_hours"])
    assert overridden.stoploss_grace_hours == 24.0


def test_grace_level_matches_open_operating_point():
    sell = _open_sell_params()
    assert sell["stoploss_grace_level"] == -0.27
    overridden = ExitParams(stoploss_grace_level=sell["stoploss_grace_level"])
    assert overridden.stoploss_grace_level == -0.27


def test_static_stoploss_matches_open_operating_point():
    """The Open JSON also pins the plain (non-ATR) stoploss at -0.05, matching
    `BaseStrategy.stoploss` and `ExitParams.static_stoploss`'s default."""
    payload = json.loads(OPEN_JSON.read_text())
    assert payload["params"]["stoploss"]["stoploss"] == -0.05
    assert BaseStrategy.stoploss == -0.05
    assert ExitParams().static_stoploss == -0.05


def test_open_operating_point_params_construct_a_valid_exitparams():
    """Assembling every FR-019 field pinned above into one ExitParams must not
    raise -- i.e. the pinned operating point is internally consistent under the
    same validation `ExitParams.__post_init__` enforces everywhere else."""
    sell = _open_sell_params()
    ExitParams(
        atr_stoploss_multiplier=BaseStrategy.atr_stoploss_multiplier,
        atr_stoploss_cap=BaseStrategy.atr_stoploss_cap,
        atr_stoploss_floor=BaseStrategy.atr_stoploss_floor,
        static_stoploss=BaseStrategy.stoploss,
        stoploss_grace_hours=sell["stoploss_grace_hours"],
        stoploss_grace_level=sell["stoploss_grace_level"],
        take_profit=sell["cexit_take_profit"],
        max_hold_days=sell["cexit_max_days"],
    )


# --------------------------------------------------------------------------------
# fee_per_side -- no single live source. Documented, not silently skipped.
# --------------------------------------------------------------------------------

def test_fee_per_side_has_no_live_strategy_source_documented():
    """freqtrade reads the trading fee from the exchange/pair at runtime (ccxt's
    fee tiers), never from strategy code -- there is no `BaseStrategy` attribute or
    strategy JSON field to assert this against, unlike every other field above.
    This test asserts only that `ExitParams.fee_per_side` is the documented
    assumption (0.001, i.e. 10bps/side) used elsewhere in this codebase's offline
    simulation (`utils/ExitPolicy.py`'s dataclass default), not that it is "the"
    live value -- there isn't one single live value to pin.
    """
    assert ExitParams().fee_per_side == 0.001
