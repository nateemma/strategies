"""HGB sizer study: the sizer hook, the HGB score lookup and the fit/tail split."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]

from _width_sizer import WidthSizerMixin  # noqa: E402

from research.hgb_sizer import signals as SG  # noqa: E402
from research.indicator_screen import windows as W  # noqa: E402


def test_default_sizer_score_is_the_model_p_buy():
    df = pd.DataFrame({"%predict_prob_buy": [0.1, 0.9]})
    assert np.array_equal(WidthSizerMixin().sizer_score(df), [0.1, 0.9])


def test_hgb_score_maps_by_pair_and_date_and_missing_is_zero(tmp_path, monkeypatch):
    import NNNC_CS_HGB as M

    d = pd.date_range("2023-01-01", periods=3, freq="15min", tz="UTC")
    pd.DataFrame({"date": d[:2], "pair": ["SOL/USDT", "SOL/USDT"], "p_buy": [0.2, 0.7]}).to_parquet(
        tmp_path / "HGB_f03.parquet")
    cls = type("NNNC_CS_HGB_f03", (M.NNNC_CS_HGB,), {"HGB_SIGNALS": tmp_path})
    s = cls.__new__(cls)
    s.curr_pair = "SOL/USDT"
    df = pd.DataFrame({"date": d})
    assert np.allclose(s.sizer_score(df), [0.2, 0.7, 0.0])
    assert np.isnan(df["sizer_score"].iloc[2])
    s.curr_pair = "ZEC/USDT"
    assert np.array_equal(s.sizer_score(pd.DataFrame({"date": d})), [0.0, 0.0, 0.0])


def test_fit_rows_end_a_purge_before_each_pairs_tail():
    d = pd.date_range("2021-01-01", periods=1000, freq="15min", tz="UTC")
    panel = SimpleNamespace(pairs=["A", "B"], date=pd.DatetimeIndex(np.concatenate([d, d[200:]])),
                            pair=np.r_[np.zeros(1000, int), np.ones(800, int)])
    panel.data_ok = np.ones(1800, bool)
    panel.candidate = np.ones(1800, bool)
    fold = SimpleNamespace(train_start=d[0], start=d[-1] + pd.Timedelta("15min"), end=d[-1] + pd.Timedelta("1D"))
    tails = SG.tail_starts(panel, fold)
    assert tails[0] == d[800] and tails[1] == d[200 + 640]
    rows = SG.fit_rows(panel, np.ones(1800), fold, tails)
    purge = W.purge(SG.TARGET) * W.BAR
    for k in (0, 1):
        r = rows[panel.pair[rows] == k]
        assert panel.date[r].max() < tails[k] - purge + pd.Timedelta("1s")
        assert panel.date[r].max() >= tails[k] - purge - W.BAR
    sc = SG.score_rows(panel, fold, tails)
    assert all(panel.date[sc[panel.pair[sc] == k]].min() == tails[k] for k in (0, 1))
