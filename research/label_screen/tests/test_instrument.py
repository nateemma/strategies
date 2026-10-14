"""034 T012: instrument pure parts on synthetic inputs."""

from __future__ import annotations

from research.label_screen import instrument as I


def test_four_dp():
    assert I.four_dp(0.1234) and not I.four_dp(0.12345) and not I.four_dp(None)


def test_keys_distinct():
    keys = {"CONTROL": {"f00": "a", "f01": "b"}, "X": {"f00": "c", "f01": "d"},
            "Y": {"f00": "c", "f01": "e"}}
    r = I.keys_distinct(keys)
    assert r["CONTROL"] and not r["X"] and not r["Y"]


def _s0(**kw):
    row = {"weights_missing": [], "key_distinct": True, "sz_thresholds_missing": [],
           "sz_thresholds_4dp": True, "classes_4dp": True}
    return row | kw


def test_s0_pass_conditions():
    assert I.s0_pass(_s0())
    assert not I.s0_pass(_s0(weights_missing=["f03"]))
    assert not I.s0_pass(_s0(key_distinct=False))
    assert not I.s0_pass(_s0(sz_thresholds_missing=["f00"]))


def _sz(eq=True, n=10, cap=1.0, viol=0, mism=0, thr=True):
    return {"n_trades": n, "trades_equal_width_book": eq, "entry_violations": viol,
            "tag_mismatches": mism, "threshold_loaded_ok": thr,
            "stakes": {"capital_ratio_vs_width": cap}}


def test_sz_summary_identity_capital_and_zero_trades():
    ok = {f"f0{i}": _sz() for i in range(8)}
    assert I.sz_arm_summary(ok)["S0_pass"]
    two = ok | {"f01": _sz(eq=False), "f02": _sz(eq=False)}
    assert I.sz_arm_summary(two)["S0_pass"]
    three = two | {"f03": _sz(eq=False)}
    assert not I.sz_arm_summary(three)["S0_pass"]
    rich = {k: _sz(cap=1.2) for k in ok}
    assert not I.sz_arm_summary(rich)["S0_pass"]
    assert I.sz_arm_summary(ok | {"f04": _sz(n=0)})["S0_inconclusive"]


def test_es_summary():
    f = {"n_trades": 5, "non_model_entries": 0, "threshold_loaded_ok": True, "control_trades": 6}
    assert I.es_arm_summary({"f00": f})["S0_pass"]
    assert not I.es_arm_summary({"f00": f | {"non_model_entries": 1}})["S0_pass"]
