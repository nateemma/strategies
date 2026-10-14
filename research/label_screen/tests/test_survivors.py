"""034 T011: the survivor rule -- gates, cap at 3, tie-break by AUC, none-survive."""

from __future__ import annotations

from research.label_screen import survivors as V


REFS = {"pre_registered": 0.072, "control_at_share": 0.06}


def _inputs(s1: dict, audit=None, s0=None):
    tags = list(s1)
    audit = audit or {t: "PASS" for t in tags}
    s0 = s0 or {t: True for t in tags}
    return ({"candidates": {t: {"status": audit[t]} for t in tags}},
            {"candidates": {t: {"S0_pass": s0[t]} for t in tags}},
            {"references": REFS, "candidates": s1})


def _s1(gap, auc=0.6, ok=True):
    return {"gap": gap, "auc": auc, "S1_pass": ok, "degenerate_folds": []}


def test_top_three_by_gap_with_auc_tie_break():
    s1 = {"A": _s1(0.10), "B": _s1(0.09, 0.55), "C": _s1(0.09, 0.65), "D": _s1(0.08),
          "E": _s1(0.05, ok=False)}
    r = V.select(*_inputs(s1))
    assert r["survivors"] == ["A", "C", "B"]
    assert not r["candidates"]["D"]["survives"] and "top 3" in r["candidates"]["D"]["reason"]
    assert not r["candidates"]["E"]["survives"]


def test_gates_before_ranking():
    s1 = {"A": _s1(0.20), "B": _s1(0.15), "C": {"status": "INCONCLUSIVE", "reason": "x"}}
    r = V.select(*_inputs(s1, audit={"A": "INCONCLUSIVE", "B": "PASS", "C": "PASS"},
                          s0={"A": True, "B": False, "C": True}))
    assert r["survivors"] == []
    assert "S0a" in r["candidates"]["A"]["reason"] and "S0" in r["candidates"]["B"]["reason"]


def test_reference_rows_can_survive():
    r = V.select(*_inputs({"GFM": _s1(0.1)}, audit={"GFM": "REFERENCE"}))
    assert r["survivors"] == ["GFM"]
