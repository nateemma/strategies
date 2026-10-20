"""040 T006: the judge's identity check and model-decided share."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.nnmt import judge as J  # noqa: E402

FOLDS = [f"f{i:02d}" for i in range(15)]
OPEN = {"entry_adx_threshold": -1.0, "entry_atr_pct": -1.0, "entry_bb_width_threshold": -1.0,
        "entry_close_norm_threshold": 5.0, "entry_enable_guards": True, "entry_guard_threshold": 5.0,
        "entry_rvol_threshold": -1.0}


def _recs(monkeypatch, tmp_path, hashes: dict) -> None:
    from research.nnnc_switches import judge as NJ

    for tag, hs in hashes.items():
        for f, h in zip(FOLDS, hs):
            (tmp_path / f"{tag}_{f}_{f.upper()}.json").write_text(json.dumps({"trades_hash": h}))
    monkeypatch.setattr(NJ, "_rec_path", lambda tag, fid: tmp_path / f"{tag}_{fid}_{fid.upper()}.json")


def test_identity_passes_on_equal_hashes(monkeypatch, tmp_path):
    _recs(monkeypatch, tmp_path, {"a": [str(i) for i in range(15)], "b": [str(i) for i in range(15)]})
    r = J.identity({"folds": FOLDS}, "a", "b")
    assert r["pass"] and r["folds_differing"] == [] and r["folds"] == 15


def test_identity_fails_on_one_differing_fold(monkeypatch, tmp_path):
    hs = [str(i) for i in range(15)]
    _recs(monkeypatch, tmp_path, {"a": hs, "b": hs[:7] + ["x"] + hs[8:]})
    r = J.identity({"folds": FOLDS}, "a", "b")
    assert not r["pass"] and r["folds_differing"] == ["f07"]


def test_identity_fails_on_a_missing_fold(monkeypatch, tmp_path):
    hs = [str(i) for i in range(15)]
    _recs(monkeypatch, tmp_path, {"a": hs, "b": hs[:14]})
    r = J.identity({"folds": FOLDS}, "a", "b")
    assert not r["pass"] and r["folds_missing"] == ["f14"]


def _table(n=100, entries=10):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"volume": np.ones(n), "rvol": rng.uniform(0.5, 2, n), "atr_pct_roll": rng.uniform(0, 0.02, n),
                       "guard_metric": rng.uniform(0, 1, n), "close_norm": rng.uniform(-1, 1, n),
                       "adx": rng.uniform(10, 50, n), "bb_width": rng.uniform(0, 0.1, n),
                       "enter_long": 0.0})
    df.loc[: entries - 1, "enter_long"] = 1.0
    return df


def test_model_share_is_one_minus_entries_over_guard_passing_bars():
    df = _table()
    r = J.model_share(df, OPEN)  # guards open: every bar with volume passes
    assert r["guard_pass_bars"] == 100 and r["entries"] == 10
    assert r["model_share"] == 0.9


def test_guard_pass_mirrors_the_strategy_chain():
    df = _table()
    df.loc[0, "volume"] = 0.0
    buy = {**OPEN, "entry_adx_threshold": 30.0, "entry_rvol_threshold": 1.0}
    g = J.guard_pass(df, buy)
    want = (df["volume"] > 0) & (df["rvol"] > 1.0) & (df["adx"] > 30.0)
    assert (g == want.to_numpy()).all()
    off = J.guard_pass(df, {**buy, "entry_enable_guards": False})  # rvol still applies, adx does not
    assert (off == ((df["volume"] > 0) & (df["rvol"] > 1.0)).to_numpy()).all()
