"""The in-strategy transition stand-in (NNNC/_transition_standin.py) and its history store
(utils/CloseHistoryStore.py) reproduce the screen's definitions; the live pieces fail safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNNC")]

import _transition_standin as TS  # noqa: E402
from research.nnmt import transition as T  # noqa: E402
from utils.CloseHistoryStore import CloseHistoryStore, _grid  # noqa: E402


def _ohlcv(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = c * np.exp(rng.normal(0, 0.001, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    v = rng.gamma(2.0, 50.0, n) * (rng.uniform(size=n) > 0.2)
    d = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": v})


def test_constants_match_the_pre_registration_and_the_harvest():
    pr = json.loads((STRAT / "research/nnmt/preregistration_transition.json").read_text())
    s, d = pr["stand_in"], pr["definition"]
    assert list(TS.INDICATORS) == s["indicators"] and list(TS.RAW8) == s["raw8"]
    assert list(TS.STATE_LENGTHS) == s["state_lengths"]
    assert (TS.SIGMA_WINDOW, TS.SIGMA_MIN) == (d["sigma_window"], d["sigma_min_periods"])
    from research.clfstudy import data as CD
    from research.indicator_screen import harvest as H

    assert TS.RAW8_WINDOW == CD._ROLL_WINDOW
    assert (TS.WARMUP_BARS, TS.DATA_OK_SHARE) == (H.W.WARMUP_BARS, H.DATA_OK_SHARE)


def test_feature_order_is_the_screens():
    pr = T._pr()
    own_btc = T.stand_in_columns(pr)
    z = [n for L in pr["stand_in"]["state_lengths"] for n in (f"z_pair_{L}", f"z_btc_{L}")]
    assert TS.feature_names() == own_btc + z


def test_zscore_raw8_and_data_ok_equal_the_research_code():
    from research.clfstudy.data import raw8 as cd_raw8
    from research.indicator_screen import harvest as H

    df = _ohlcv()
    pr = T._pr()
    d = pr["definition"]
    for L in TS.STATE_LENGTHS:
        a = TS.zscore(df["close"], L).to_numpy()
        b = T.zscore(df["close"], L, d["sigma_window"], d["sigma_min_periods"]).to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0)
    r_new, r_ref = TS.raw8(df), cd_raw8(df)
    for c in TS.RAW8:
        assert np.array_equal(r_new[c].to_numpy(), r_ref[c].to_numpy()), c
    raw_dates = df["date"][np.random.default_rng(1).uniform(size=len(df)) > 0.03]
    has = df["date"].isin(raw_dates).astype(np.int8)
    _, ok_ref = H.data_flags(df["date"], raw_dates)
    assert np.array_equal(TS.data_ok(has).to_numpy(), ok_ref.astype(bool))


def test_store_grid_fill_and_presence():
    raw = _ohlcv(10)[["date", "close"]].drop(index=[3, 4]).reset_index(drop=True)
    g = _grid(raw, 15)
    assert len(g) == 10 and g["has"].tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    assert g["close"].iloc[3] == g["close"].iloc[2] == g["close"].iloc[4]


def test_store_appends_live_candles_marks_a_seam_hole_and_checks_coverage(tmp_path):
    full = _ohlcv(50)
    s = CloseHistoryStore(tmp_path, persist_dir=tmp_path / "persist")
    s._load_raw = lambda pair: full.iloc[:30][["date", "close"]]  # local download ends at bar 29
    s.update("X/USDT", full.iloc[25:40])  # live buffer overlaps the download: contiguous
    g = s.series("X/USDT", full["date"].iloc[25])
    assert len(g) == 40 and g["has"].tolist() == [1] * 40
    s.update("X/USDT", full.iloc[45:50])  # a hole (bars 40-44 never seen)
    g = s.series("X/USDT", full["date"].iloc[25])
    assert len(g) == 50 and g["has"].iloc[40:45].tolist() == [0] * 5 and g["has"].iloc[45:].tolist() == [1] * 5
    assert (tmp_path / "persist" / "X_USDT.parquet").is_file()
    assert s.series("X/USDT", full["date"].iloc[0] - pd.Timedelta(minutes=15)) is None  # history does not reach


def test_a_missing_artifact_raises_and_the_default_source_is_tables(tmp_path):
    with pytest.raises(FileNotFoundError):
        TS.StandIn("19990101", root=tmp_path)
    from _transition_signal import TransitionSignalMixin

    assert TransitionSignalMixin.TR_SOURCE == "tables"


def test_mainline_cutoff_is_the_newest_fit_on_or_before_the_first_candle(tmp_path, monkeypatch):
    from _transition_signal import TransitionSignalMixin

    for d in ("20260101", "20260601", "20261020"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "meta.json").write_text("{}")
    (tmp_path / "20260301").mkdir()  # no meta.json: not a finished fit
    monkeypatch.setattr(TS, "ARTIFACTS", tmp_path)
    m = TransitionSignalMixin()
    assert m._tr_cutoff(pd.Timestamp("2026-07-15", tz="UTC")) == "20260601"
    assert m._tr_cutoff(pd.Timestamp("2026-10-20", tz="UTC")) == "20261020"
    with pytest.raises(FileNotFoundError):
        m._tr_cutoff(pd.Timestamp("2025-12-31", tz="UTC"))
    m.MODEL_TRAIN_TIMERANGE = "20210101-20230101"
    assert m._tr_cutoff(pd.Timestamp("2026-07-15", tz="UTC")) == "20230101"
