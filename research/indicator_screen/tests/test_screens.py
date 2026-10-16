"""037 T020 / T024: the screens on a synthetic 3-pair cache -- ranking, the instrument check, the
late lock, byte-identical reruns, train-span MI edges, the E0 rule, pairs and models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W

N = 14000
PAIRS = ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
T0 = pd.Timestamp("2021-01-01", tz="UTC")


def _t(bar):
    return T0 + bar * W.BAR


WINS = [W.Window("E0", "era", "e0", _t(2000), _t(4000), None, None)] + [
    W.Window(f"f0{i}", "test", "early" if i < 3 else "late", _t(4000 + 2000 * i), _t(6000 + 2000 * i),
             _t(0), _t(4000 + 2000 * i)) for i in range(5)]


def windows_fn(halves=("e0", "early", "late")):
    return [w for w in WINS if w.half in halves]


PR = {"min_rows": 50, "min_fires": 5, "max_fire_rate": 0.95, "mi_bins": 10, "residual_grid": 3,
      "sample_rows": 5000, "seed": 0, "residual_min_strength": 0.02, "redundancy_cut": 0.8,
      "instrument": {"top_k": 5, "columns": ["base:guard_metric", "base:bb_width"],
                     "target": "label:buy", "population": "all"},
      "instrument_gate_halves": ["e0", "early"],
      "pairs": {"min_windows": 2, "uplift": 1.10, "min_lift": 1.20, "top_n": 5},
      "models": {"tree": {"max_depth": 3, "min_samples_leaf": 50, "class_weight": "balanced", "random_state": 0},
                 "hgb": {"max_iter": 30, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                         "l2_regularization": 1.0, "early_stopping": False, "random_state": 0},
                 "restricted_inputs": ["base:bb_width", "base:guard_metric"]}}


def _pair_frame(seed, shuffle_guard=False):
    rng = np.random.default_rng(seed)
    date = pd.date_range(T0, periods=N, freq="15min")
    guard = rng.random(N)
    width = rng.random(N) * 0.1
    cand = (guard < 0.3) & (width > 0.035)
    a = rng.random(N) < 0.3                      # raw signal A
    b = rng.random(N) < 0.3                      # raw signal B: A AND B carries the label
    p = 0.02 + 0.6 * cand + 0.25 * (a & b)
    buy = (rng.random(N) < p).astype(float)
    sell = (rng.random(N) < 0.1).astype(float)
    bar = np.arange(N)
    e0 = (bar >= 2000) & (bar < 4000)
    late = bar >= 10000
    f = {"date": date, "open": 1.0, "high": 1.0, "low": 1.0, "close": 100 + np.cumsum(rng.normal(0, 1, N)),
         "volume": 1.0,
         "meta:data_ok": (bar >= W.WARMUP_BARS).astype(np.int8),
         "base:gbb_candidate": cand.astype(np.int8),
         "label:buy": buy, "label:sell": sell,
         "base:bb_width": width, "base:guard_metric": rng.permutation(guard) if shuffle_guard else guard,
         "X:planted": buy + rng.normal(0, 0.6, N),
         "X:flip": np.where(e0, -1, 1) * buy + rng.normal(0, 0.6, N),
         "X:shift": buy + rng.normal(0, 0.6, N) + 100 * late,
         "raw_buy:A": a.astype(np.int8), "raw_buy:B": b.astype(np.int8),
         "raw_sell:S": (rng.random(N) < 0.2).astype(np.int8)}
    for k in range(5):
        f[f"N:noise{k}"] = rng.random(N)
    return pd.DataFrame(f)


COLS = ["base:bb_width", "base:guard_metric", "X:planted", "X:flip", "X:shift", "raw_buy:A",
        "raw_buy:B", "raw_sell:S"] + [f"N:noise{k}" for k in range(5)]


def _cache(path, shuffle_guard=False):
    for i, p in enumerate(PAIRS):
        H.write_cache(p, _pair_frame(i, shuffle_guard), {"classes": {}}, "t", cache=path)
    return path


@pytest.fixture
def synth(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "late_lock", lambda *a, **k: "testlock")
    return _cache(tmp_path / "cache")


def _run(cache, tmp, halves, **kw):
    from research.indicator_screen import screen_single as SS
    from research.indicator_screen.panel import Panel

    panel = Panel(PAIRS, cache=cache)
    out = tmp / "out"
    summ = tmp / kw.pop("name", "single.json")
    rc = SS.run("labels", halves, panel=panel, out_dir=out, summary_path=summ, cols=COLS,
                windows_fn=windows_fn, pr=PR, red_path=tmp / "red.json", **kw)
    return rc, summ


def test_planted_column_ranks_first_and_instrument_passes(synth, tmp_path):
    rc, summ = _run(synth, tmp_path, ["e0", "early"])
    s = json.loads(summ.read_text())
    assert rc == 0 and s["instrument"]["verdict"] == "PASS"
    assert s["instrument"]["top"][0]["column"] in ("X:planted", "X:shift")
    assert s["prereg_lock"] is None                  # no late window, no lock needed


def test_instrument_fails_when_guard_metric_is_shuffled(tmp_path, monkeypatch):
    cache = _cache(tmp_path / "bad", shuffle_guard=True)
    rc, summ = _run(cache, tmp_path, ["e0", "early"])
    s = json.loads(summ.read_text())
    assert rc == 1 and s["instrument"] == "FAIL"
    assert s["detail"]["ranks"]["base:guard_metric"] > 5


def test_late_is_refused_without_the_lock(tmp_path, monkeypatch):
    cache = _cache(tmp_path / "c")

    def refuse(*a, **k):
        raise W.LateLockError("not committed")

    monkeypatch.setattr(W, "late_lock", refuse)
    with pytest.raises(W.LateLockError):
        _run(cache, tmp_path, ["late"])


def test_rerun_is_byte_identical(synth, tmp_path):
    _run(synth, tmp_path, ["e0", "early", "late"], name="a.json")
    _run(synth, tmp_path, ["e0", "early", "late"], name="b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_mi_edges_come_from_the_training_span(synth, tmp_path):
    _run(synth, tmp_path, ["e0", "early", "late"])
    df = pd.read_feather(tmp_path / "out" / "single_labels_late.feather")
    g = df[(df["column"] == "X:shift") & (df["target"] == "label:buy") & (df["population"] == "all")
           & (df["window"] == "f03")]
    # f03 is the first shifted (+100) window and its training span has none: with train-span edges
    # every f03 value lands in the top bin (f04's anchored span already contains f03)
    assert (g["mi_edges_from"] == "train").all() and (g["mi_norm"] < 0.01).all()
    e = pd.read_feather(tmp_path / "out" / "single_labels_e0.feather")
    e = e[(e["column"] == "X:shift") & (e["target"] == "label:buy") & (e["population"] == "all")]
    assert (e["mi_edges_from"] == "self").all() and (e["mi_norm"] > 0.05).all()


def test_e0_flip_is_not_a_candidate(synth, tmp_path):
    _, summ = _run(synth, tmp_path, ["e0", "early", "late"])
    res = {(r["column"], r["target"], r["population"]): r for r in json.loads(summ.read_text())["results"]}
    flip = res[("X:flip", "label:buy", "candidate")]
    planted = res[("X:planted", "label:buy", "candidate")]
    assert flip["halves"]["early"]["median_auc"] > 0.5 and flip["halves"]["e0"]["median_auc"] < 0.5
    assert flip["candidate_flags"]["e0_agrees"] is False
    assert planted["candidate_flags"]["e0_agrees"] is True
    assert planted["candidate_flags"]["beats_width_both_halves"] is True
    assert planted["candidate_flags"]["residual_both_halves"] is True
    w = res[("base:bb_width", "label:buy", "candidate")]
    assert abs(w["median_strat_auc"] - 0.5) < 0.05      # width carries nothing within its own cells
    assert planted["redundancy_group"] != res[("N:noise0", "label:buy", "candidate")]["redundancy_group"]
