"""037 T012a: the harvest -- raw signals are the class's own pre-filter output, aliases, defects,
the cache sha, the 24 ARO inputs and the gbb candidate flag."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.indicator_screen import harvest as H


@pytest.fixture(scope="module")
def harvested(loaded, sol):
    s = loaded["strats"]
    strats = [s["CCI"], s["Combo"], s["BBBreakout"], s["Ehlers_BB"]]
    return H.harvest_frame(sol, strats, loaded["aro"], "SOL/USDT")


def test_raw_cci_equals_its_own_signal_and_ignores_every_filter(loaded, sol, base, harvested):
    frame, info = harvested
    cci = loaded["strats"]["CCI"]
    d = base.copy()
    expect = (np.asarray(cci.get_entry_signals(d), dtype=float) > 0).astype(np.int8)
    assert np.array_equal(frame["raw_buy:CCI"].to_numpy(), expect)
    d2 = base.copy()                                # filter columns flipped: no effect
    d2["atr_ok"] = False
    d2["market_state"] = 2
    d2["trend_mode"] = 1 - d2["trend_mode"].fillna(0)
    assert np.array_equal((np.asarray(cci.get_entry_signals(d2), dtype=float) > 0).astype(np.int8), expect)
    assert info["classes"]["CCI"]["status"] == "harvested" and "CCI:cci" in frame.columns


def test_family_filter_would_have_removed_fires(loaded, base, harvested):
    """The family filter (get_entry_conditions) is NOT applied: raw fires >= filtered fires."""
    frame, _ = harvested
    cci = loaded["strats"]["CCI"]
    d = base.copy()
    filtered = np.asarray(cci.get_entry_conditions(d), dtype=bool)
    assert frame["raw_buy:CCI"].sum() > filtered.sum()


def test_all_24_aro_inputs_present_and_equal_to_the_control(loaded, sol, harvested):
    frame, info = harvested
    aro = loaded["aro"]
    assert len(info["aro_include_list"]) == 24
    ad = H.populate_path(aro, sol, "SOL/USDT")
    for c in aro.include_list:
        col = f"aro:{c}"
        src = info["aliases"].get(col, col)
        assert src in frame.columns, col
        assert np.allclose(frame[src].to_numpy(np.float64), ad[c].to_numpy(np.float64).astype(np.float32),
                           equal_nan=True, rtol=1e-6), col


def test_gbb_flag_equals_the_rule_on_the_control_frame(loaded, sol, harvested):
    from research.outcomes.builtin_triggers import _gbb

    frame, _ = harvested
    ad = H.populate_path(loaded["aro"], sol, "SOL/USDT")
    expect = _gbb(ad).to_numpy().astype(np.int8)
    assert frame["base:gbb_candidate"].sum() == expect.sum() > 0
    assert np.array_equal(frame["base:gbb_candidate"].to_numpy(), expect)


def test_aliases_and_defects():
    assert H._defect(np.array([1.0, 1.0, np.nan])) == "constant"
    assert H._defect(np.array([np.nan, np.nan])) == "all_missing"
    assert H._defect(np.array([1.0, 2.0])) is None
    a = np.array([1.0, np.nan, 2.0])
    assert H._key(a) == H._key(a.copy()) != H._key(np.array([1.0, np.nan, 2.5]))


def test_constant_signal_is_excluded_with_a_reason(loaded, sol):
    class Never:                                    # a class whose signal never fires
        def get_entry_signals(self, d):
            d["never_ind"] = np.arange(len(d), dtype=float)
            return np.zeros(len(d))

        def get_exit_signals(self, d):
            return np.zeros(len(d))

    frame, info = H.harvest_frame(sol.iloc[:3000].reset_index(drop=True),
                                  [loaded["strats"]["CCI"], Never()], None, "SOL/USDT")
    assert info["excluded_columns"]["raw_buy:Never"] == "constant"
    assert "Never:never_ind" in frame.columns


def test_sidecar_sha_mismatch_forces_a_rebuild(tmp_path):
    f = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=3, freq="15min", tz="UTC"),
                      **{c: [1.0, 2.0, 3.0] for c in H.OHLCV}, "base:x": [1.0, 2.0, 3.0]})
    H.write_cache("SOL/USDT", f, {"classes": {}}, "abc", cache=tmp_path)
    assert H.is_fresh("SOL/USDT", "abc", cache=tmp_path)
    assert not H.is_fresh("SOL/USDT", "def", cache=tmp_path)
    assert list(H.read_cache("SOL/USDT", ["base:x", "nope"], cache=tmp_path).columns) == ["date", "base:x"]
    assert json.loads(H.sidecar_path("SOL/USDT", tmp_path).read_text())["rows"] == 3


def test_data_flags_mark_filled_bars_and_their_warmup():
    d = pd.Series(pd.date_range("2021-01-01", periods=3000, freq="15min", tz="UTC"))
    raw = d.drop(index=range(1200, 1500))           # a 300-bar filled gap
    has, ok = H.data_flags(d, raw)
    assert has[1199] == 1 and has[1200] == 0 and has[1500] == 1
    assert not ok[:999].any() and ok[999] == 1      # warm-up
    assert not ok[1200:1500].any()                  # filled bars
    assert not ok[1500] and ok[2800]                # trailing window still >5% filled, then clean
