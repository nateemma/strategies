"""Search-space invariants (T007)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import space as SP          # noqa: E402


def test_round_trip():
    assert SP._check(2000) == 0


def test_ten_dimensions_including_the_two_added_in_planning():
    assert len(SP.DIMS) == 10
    assert "cexit_take_profit" in SP.NAMES, "51.6% of exits; the brief's list omits it"
    assert "hold_window_hours" in SP.NAMES, "minimum holding time, named at spec.md:14"


def test_incumbent_is_read_from_the_strategys_own_files():
    """Not hard-coded, so it cannot drift from what production runs."""
    assert SP.INCUMBENT["stoploss_grace_hours"] == pytest.approx(24.0)  # 6.5 before 2026-10-13
    assert SP.INCUMBENT["cexit_take_profit"] == pytest.approx(0.024)
    assert SP.INCUMBENT["entry_rvol_threshold"] == pytest.approx(2.0)
    assert SP.INCUMBENT["atr_stoploss_multiplier"] == pytest.approx(2.5)
    assert SP.INCUMBENT["cexit_max_days"] == 10


def test_every_incumbent_lies_inside_its_bounds():
    for d in SP.DIMS:
        assert d.lo <= d.incumbent <= d.hi, f"{d.name} incumbent outside its own bounds"


def test_floor_cap_ordering_is_rejected_not_clipped():
    """Clipping would map distinct candidates onto one point and corrupt the trial count."""
    bad = dict(SP.INCUMBENT, atr_stoploss_floor=-0.01, atr_stoploss_cap=-0.03)
    with pytest.raises(SP.Inadmissible):
        SP.validate(bad)
    SP.validate(dict(SP.INCUMBENT))                   # the incumbent itself must be valid


def test_integer_rounding_collides_candidate_ids():
    """Two z-vectors rounding to the same cexit_max_days are ONE evaluation for scoring."""
    a = dict(SP.INCUMBENT, cexit_max_days=10)
    b = dict(SP.INCUMBENT, cexit_max_days=10.0)
    assert SP.candidate_id(a) == SP.candidate_id(b)
    c = dict(SP.INCUMBENT, cexit_max_days=11)
    assert SP.candidate_id(a) != SP.candidate_id(c)


def test_on_bound_fires_at_each_bound():
    for d in SP.DIMS:
        if d.one_sided:
            continue
        assert d.name in SP.on_bound(dict(SP.INCUMBENT, **{d.name: d.lo}))
        assert d.name in SP.on_bound(dict(SP.INCUMBENT, **{d.name: d.hi}))


def test_one_sided_dimension_is_exempt_from_the_bound_finding():
    """hold_window_hours is at its floor by construction — the mechanism can only widen.

    Reporting it as "a space drawn too narrowly" would be a false FR-019 finding on every
    single run.
    """
    assert "hold_window_hours" not in SP.on_bound(dict(SP.INCUMBENT))
    assert SP.INCUMBENT["hold_window_hours"] == 12.0


def test_incumbent_is_at_zero_distance_from_itself():
    assert SP.distance_from_incumbent(dict(SP.INCUMBENT)) == pytest.approx(0.0)
    far = {d.name: (d.hi if abs(d.hi - d.incumbent) > abs(d.lo - d.incumbent) else d.lo)
           for d in SP.DIMS}
    assert SP.distance_from_incumbent(far) > 0.4
