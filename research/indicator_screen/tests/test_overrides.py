"""037 T011b: each override changes the signal vs the as-built code and reaches the class's own path
(Principle IV: a null arm must be shown to be live)."""
from __future__ import annotations

import numpy as np

from research.indicator_screen import overrides as O


def _asbuilt_entry(strat, base):
    d = base.copy()
    d["atr_ok"] = True
    return np.asarray(strat.get_entry_signals(d), dtype=float) > 0


def test_override_kinds(loaded):
    s = loaded["strats"]
    assert O.kind(s["Ehlers_BB"]) == "ehlers_triggers" and O.kind(s["HTDCP"]) == "ehlers_triggers"
    assert O.kind(s["BBBreakout"]) == "force_trend_mode"
    assert O.kind(s["Ichimoku"]) == "ichimoku_switches" and O.kind(s["CCI"]) == "plain"


def test_ehlers_raw_triggers_differ_from_the_voted_signal(loaded, base):
    s = loaded["strats"]["Ehlers_BB"]
    built = _asbuilt_entry(s, base)
    d = base.copy()
    d["atr_ok"] = True
    raw = O.raw_signals(s, d)
    union = (raw["buy:cycle"] | raw["buy:trend"]).astype(bool)
    assert union.sum() > built.sum()                # the vote/strict regions removed fires
    assert not (built & ~union).any()               # every as-built fire is a raw trigger fire
    assert "ehler_buy" not in d.columns


def test_bbbreakout_forced_trend_mode_reaches_its_own_code(loaded, base):
    s = loaded["strats"]["BBBreakout"]
    built = _asbuilt_entry(s, base)
    d = base.copy()
    d["atr_ok"] = True
    raw = O.raw_signals(s, d)
    tm = base["trend_mode"].to_numpy()
    recombined = np.where(tm == 1, raw["buy:tm1"], raw["buy:tm0"]).astype(bool)
    assert np.array_equal(recombined, built)        # tm1/tm0 are exactly its two branches
    assert (raw["buy:tm1"] | raw["buy:tm0"]).sum() > built.sum()
    assert np.array_equal(d["trend_mode"].to_numpy(), tm)   # value columns see the real mode


def test_ichimoku_switches_off_differs_where_the_cloud_filter_binds(loaded, base):
    s = loaded["strats"]["Ichimoku"]
    s.entry_use_cloud_thickness.value = True
    built = _asbuilt_entry(s, base)
    d = base.copy()
    d["atr_ok"] = True
    raw = O.raw_signals(s, d)
    assert raw["buy"].sum() > built.sum() and not (built & ~raw["buy"].astype(bool)).any()
