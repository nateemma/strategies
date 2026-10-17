"""Composite sizer: normalisation rows stop at each pair's tail start; the class reads its own tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]

from research.cmp_sizer import signals as SG  # noqa: E402


def test_composite_is_signed_mean_of_z_on_norm_rows():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, np.nan]])
    rows = np.array([True, True, False])
    c = SG.composite(X, np.array([1.0, -1.0]), rows)
    assert np.allclose(c, [(-1 + 1) / 2, (1 - 1) / 2, (3 + 0) / 2])


def test_class_points_at_cmp_tables():
    import NNNC_CS_CMP as M
    assert M.NNNC_CS_CMP.HGB_SIGNALS.name == "cmp_sizer"


def test_pick_passes_the_cmp_plan(monkeypatch):
    from research.cmp_sizer import pick as P

    seen = {}
    monkeypatch.setattr(P.HP, "pick", lambda fid, plan: (seen.setdefault("pick", plan), (0.1, {"tail_leak_pairs": []}))[1])
    monkeypatch.setattr(P.HP, "verify", lambda fid, plan: seen.setdefault("verify", plan) is not None)
    monkeypatch.setattr(P.HP.CP, "store", lambda plan, fid, t, extra: seen.setdefault("store", plan) and t)
    assert P.main(["--fold", "f00"]) == 0 and P.main(["--fold", "f00", "--verify"]) == 0
    assert {p.name for p in seen.values()} == {"plan_cs_cmp.json"}
