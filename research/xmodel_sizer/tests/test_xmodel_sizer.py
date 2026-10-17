"""Cross-model sizers: the mix mixin's score, class wiring, and the pick passing each model's plan."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]


def test_mixin_scores_model_and_composite(tmp_path):
    from _mix_sizer import MixSizerMixin

    d = pd.date_range("2023-01-01", periods=3, freq="15min", tz="UTC")
    pd.DataFrame({"date": d[:2], "pair": ["SOL/USDT"] * 2, "p_buy": [-1.0, 2.0]}).to_parquet(tmp_path / "HGB_f03.parquet")
    (tmp_path / "MIX_f03.json").write_text(json.dumps({"model": [0.1, 0.2, 0.3, 0.4], "cmp": [-1.0, 0.0, 1.0, 2.0]}))
    cls = type("X_f03", (MixSizerMixin,), {"CMP_TABLES": tmp_path, "MIX_KNOTS": tmp_path})
    s = cls()
    s.curr_pair = "SOL/USDT"
    df = pd.DataFrame({"date": d, "%predict_prob_buy": [0.4, 0.15, 0.3]})
    assert np.allclose(s.sizer_score(df), [0.5 * (1.0 + 0.25), 0.5 * (0.25 + 1.0), 0.0])


def test_classes_put_the_mix_score_first():
    import NNNC_CS_GBR_MIX as G
    import NNNC_CS_MLX_MIX as M
    from _mix_sizer import MixSizerMixin

    for c, m in ((G.NNNC_CS_GBR_MIX, "gbr"), (M.NNNC_CS_MLX_MIX, "mlx")):
        assert c.sizer_score is MixSizerMixin.sizer_score
        assert c.MIX_KNOTS.name == f"mix_sizer_{m}"


def test_pick_passes_each_models_plan(monkeypatch):
    from research.xmodel_sizer import pick as P

    seen = []
    monkeypatch.setattr(P.HP, "pick", lambda fid, plan: (seen.append(plan.name), (0.1, {"tail_leak_pairs": []}))[1])
    monkeypatch.setattr(P.HP, "verify", lambda fid, plan: seen.append(plan.name) or True)
    monkeypatch.setattr(P.HP.CP, "store", lambda plan, fid, t, extra: seen.append(plan.name) or t)
    for m in ("GBR", "MLX"):
        assert P.main(["--model", m, "--fold", "f00"]) == 0 and P.main(["--model", m, "--fold", "f00", "--verify"]) == 0
    assert seen == ["plan_cs_gbr_mix.json"] * 3 + ["plan_cs_mlx_mix.json"] * 3
