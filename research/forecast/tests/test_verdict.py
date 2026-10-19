"""038 verdict rule (T018): research R9 as committed in preregistration.json."""
from __future__ import annotations

import json

import pytest

from research.forecast import common as C
from research.forecast import verdict as V

PR = json.loads(C.PREREG.read_text())
KEY = f"mean_net_{float(PR['costs']['verdict_bps'])}"


def cell(n=300, net=0.001, edge=0.0005, lb=0.0002, fail=0.0, npy=1.0):
    return {"n_trades": n, KEY: net, "edge_vs_rand": edge, "net_lb95": lb, "fail_share": fail,
            "net_per_pair_year": npy}


def halves(e=None, l=None):
    return {"early": e or cell(), "late": l or cell()}


def test_pass():
    v, _ = V.decide(halves(), PR, learned=False)
    assert v == "PASS (PROVISIONAL)"


@pytest.mark.parametrize("bad", [{"net": -0.0001}, {"net": 0.0}, {"edge": -0.0001}])
@pytest.mark.parametrize("half", ["early", "late"])
def test_fail_on_net_or_edge_in_either_half(bad, half):
    h = halves()
    h[half] = cell(**bad)
    assert V.decide(h, PR, learned=False)[0] == "FAIL"


def test_too_few_trades_is_inconclusive():
    assert V.decide(halves(e=cell(n=199)), PR, learned=False)[0] == "INCONCLUSIVE"


def test_zero_trades_is_inconclusive_not_fail():
    z = {"n_trades": 0, KEY: None, "edge_vs_rand": None, "net_lb95": None, "fail_share": 0.0}
    assert V.decide({"early": z, "late": z}, PR, learned=False)[0] == "INCONCLUSIVE"


def test_late_lower_bound_must_clear_zero():
    assert V.decide(halves(l=cell(lb=-0.0001)), PR, learned=False)[0] == "INCONCLUSIVE"


def test_fail_share_over_limit_is_inconclusive():
    assert V.decide(halves(e=cell(fail=0.06), l=cell(fail=0.06)), PR, learned=False)[0] == "INCONCLUSIVE"


def test_learned_pass_needs_seed2_pass():
    assert V.decide(halves(), PR, learned=True, seed2=None)[0] == "INCONCLUSIVE"
    assert V.decide(halves(), PR, learned=True, seed2=halves(l=cell(lb=-1)))[0] == "INCONCLUSIVE"
    assert V.decide(halves(), PR, learned=True, seed2=halves())[0] == "PASS (PROVISIONAL)"


def _scores(null_pass: bool):
    good = {str(n): halves() for n in C.HORIZONS}
    bad = {str(n): halves(e=cell(net=-0.001)) for n in C.HORIZONS}
    return {"lgb_L": {"n": good}, "lgb_L_s2": {"n": good}, "lgb_null": {"n": good if null_pass else bad},
            "lgb_null_s2": {"n": good}, "holt": {"n": good}}


def test_null_pass_downgrades_learned_tiers_only():
    out = V.judge(_scores(null_pass=True), PR)
    assert out["null_passes"]
    assert all(c["verdict"] == "INCONCLUSIVE" for c in out["cells"]["lgb_L"].values())
    assert all(c["verdict"].startswith("PASS") for c in out["cells"]["holt"].values())
    out = V.judge(_scores(null_pass=False), PR)
    assert not out["null_passes"]
    assert all(c["verdict"].startswith("PASS") for c in out["cells"]["lgb_L"].values())
    assert "lgb_L_s2" not in out["cells"]  # reruns inform (e); they get no verdict of their own


def test_most_viable_and_break_even():
    s = {"holt": {"n": {str(n): halves(l=cell(npy=float(n == 16))) for n in C.HORIZONS}},
         "ar": {"n": {str(n): halves(e=cell(net=-0.001 if n < 8 else 0.001, npy=0.0), l=cell(npy=0.0))
                      for n in C.HORIZONS}}}
    s["holt"]["n"]["1"] = halves(e=cell(net=-0.001))
    s["holt"]["n"]["2"] = halves(e=cell(net=-0.001))
    s["holt"]["n"]["4"] = halves(e=cell(net=-0.001))
    out = V.judge(s, PR)
    assert out["most_viable_n"] == 16
    assert out["break_even_n"] == 8


def test_refuses_without_committed_preregistration(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "PREREG", tmp_path / "missing.json")
    assert C.refuse_unless_registered([]) == 2


def test_ridge_is_learned_and_deterministic():
    good = {str(n): halves() for n in C.HORIZONS}
    bad = {str(n): halves(e=cell(net=-0.001)) for n in C.HORIZONS}
    out = V.judge({"ridge_L": {"n": good}, "lgb_null": {"n": bad}}, PR)
    assert all(c["verdict"].startswith("PASS") for c in out["cells"]["ridge_L"].values())
    out = V.judge({"ridge_L": {"n": good}, "lgb_null": {"n": good}, "lgb_null_s2": {"n": good}}, PR)
    assert all(c["verdict"] == "INCONCLUSIVE" for c in out["cells"]["ridge_L"].values())  # null downgrade
