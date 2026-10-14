"""033 T022: S1 helpers -- the WR agree rule, the split statistics, and the late-half gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.width_relative import separation as S


def test_wr_agree_needs_argmax_and_threshold():
    t = pd.DataFrame({"buy_argmax": [True, True, False, True], "p_buy": [0.6, 0.5, 0.9, 0.5001]})
    assert S.wr_agree(t, 0.5).tolist() == [True, False, False, True]


def test_split_stats_gap_and_capital():
    r = S.split_stats(np.array([0.2, 0.0, -0.1, 0.1]), np.array([True, False, False, True]))
    assert r["agree"] == {"n": 2, "pp_per_trade": 0.15}
    assert r["disagree"] == {"n": 2, "pp_per_trade": -0.05}
    assert r["gap"] == 0.2
    assert r["agree_0.5_pp_per_capital"] == round((0.2 + 0.1 - 0.05) / 0.75, 3)


def test_late_half_only_after_s3_is_judged(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)
    assert not S.s3_judged("ARO")
    (tmp_path / "verdicts.json").write_text(json.dumps(
        {"models": {"ARO": {"exit_off": {"S3": "not_reached"}}}}))
    assert not S.s3_judged("ARO")
    (tmp_path / "verdicts.json").write_text(json.dumps(
        {"models": {"ARO": {"exit_off": {"S3": {"pass": False}}}}}))
    assert S.s3_judged("ARO") and not S.s3_judged("GBR")
