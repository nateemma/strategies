"""Mixed sizer: percentile maps, the mixed score, and the class reading both scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]

from research.mix_sizer import knots as K  # noqa: E402


def test_pct_and_mixed():
    k = {"aro": [0.1, 0.2, 0.3, 0.4], "cmp": [-1.0, 0.0, 1.0, 2.0]}
    assert np.allclose(K.pct([0.05, 0.25, 0.9], k["aro"]), [0.0, 0.5, 1.0])
    m = K.mixed([0.25, np.nan], [1.5, 0.0], k)
    assert np.isclose(m[0], 0.5 * (0.5 + 0.75)) and np.isnan(m[1])


def test_class_mixes_aro_and_composite(tmp_path):
    import NNNC_CS_MIX as M

    d = pd.date_range("2023-01-01", periods=3, freq="15min", tz="UTC")
    pd.DataFrame({"date": d[:2], "pair": ["SOL/USDT"] * 2, "p_buy": [-1.0, 2.0]}).to_parquet(tmp_path / "HGB_f03.parquet")
    (tmp_path / "MIX_f03.json").write_text(json.dumps({"aro": [0.1, 0.2, 0.3, 0.4], "cmp": [-1.0, 0.0, 1.0, 2.0]}))
    cls = type("NNNC_CS_MIX_f03", (M.NNNC_CS_MIX,), {"HGB_SIGNALS": tmp_path, "MIX_KNOTS": tmp_path})
    s = cls.__new__(cls)
    s.curr_pair = "SOL/USDT"
    df = pd.DataFrame({"date": d, "%predict_prob_buy": [0.4, 0.15, 0.3]})
    assert np.allclose(s.sizer_score(df), [0.5 * (1.0 + 0.25), 0.5 * (0.25 + 1.0), 0.0])


def test_pick_passes_the_mix_plan(monkeypatch):
    from research.mix_sizer import pick as P

    seen = {}
    monkeypatch.setattr(P.HP, "pick", lambda fid, plan: (seen.setdefault("pick", plan), (0.1, {"tail_leak_pairs": []}))[1])
    monkeypatch.setattr(P.HP, "verify", lambda fid, plan: seen.setdefault("verify", plan) is not None)
    monkeypatch.setattr(P.HP.CP, "store", lambda plan, fid, t, extra: seen.setdefault("store", plan) and t)
    assert P.main(["--fold", "f00"]) == 0 and P.main(["--fold", "f00", "--verify"]) == 0
    assert {p.name for p in seen.values()} == {"plan_cs_mix.json"}
