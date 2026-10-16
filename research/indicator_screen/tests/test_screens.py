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
      "instrument": {"rule": "direction_strength", "target": "label:buy", "population": "all",
                     "expected_direction": {"base:guard_metric": -1, "base:bb_width": 1}, "min_strength": 0.1},
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
    c = rng.random(N) < 0.3                      # C AND D carries the label except in E0
    dd = rng.random(N) < 0.3
    bar = np.arange(N)
    e0 = (bar >= 2000) & (bar < 4000)
    p = 0.02 + 0.6 * cand + 0.25 * (a & b) + np.where(e0, -0.3, 0.25) * (c & dd)
    buy = (rng.random(N) < np.clip(p, 0, 1)).astype(float)
    sell = (rng.random(N) < 0.1).astype(float)
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
         "raw_buy:C": c.astype(np.int8), "raw_buy:D": dd.astype(np.int8),
         "raw_sell:S": (rng.random(N) < 0.2).astype(np.int8)}
    for k in range(5):
        f[f"N:noise{k}"] = rng.random(N)
    return pd.DataFrame(f)


COLS = ["base:bb_width", "base:guard_metric", "X:planted", "X:flip", "X:shift", "raw_buy:A",
        "raw_buy:B", "raw_buy:C", "raw_buy:D", "raw_sell:S"] + [f"N:noise{k}" for k in range(5)]


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
    assert s["instrument"]["top"][0]["column"] in ("X:planted", "X:shift", "X:flip")
    assert all(v["pass"] for v in s["instrument"]["columns"].values())
    assert s["prereg_lock"] is None                  # no late window, no lock needed


def test_instrument_fails_when_guard_metric_is_shuffled(tmp_path, monkeypatch):
    cache = _cache(tmp_path / "bad", shuffle_guard=True)
    rc, summ = _run(cache, tmp_path, ["e0", "early"])
    s = json.loads(summ.read_text())
    assert rc == 1 and s["instrument"] == "FAIL"
    assert s["detail"]["columns"]["base:guard_metric"]["pass"] is False
    assert s["detail"]["columns"]["base:bb_width"]["pass"] is True


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


# ---------------------------------------------------------------------------- combinations (T024)


def _panel(cache):
    from research.indicator_screen.panel import Panel

    return Panel(PAIRS, cache=cache)


def test_planted_pair_is_selected_and_confirmed_and_select_reads_no_late_window(synth):
    from research.indicator_screen import screen_pairs as SP
    from research.indicator_screen.screen_single import LabelTargets

    panel = _panel(synth)
    t = LabelTargets(panel)
    sel = SP.select(panel, t, "label:buy", windows_fn, COLS, PR)
    assert sel["windows_read"] == ["E0", "f00", "f01", "f02"]
    surv = {(s["a"], s["b"]): s for s in sel["populations"]["candidate"]["survivors"]}
    assert ("raw_buy:A", "raw_buy:B") in surv and ("raw_buy:C", "raw_buy:D") in surv
    assert "late" not in surv[("raw_buy:A", "raw_buy:B")]
    conf = SP.confirm(panel, t, sel, windows_fn, PR)
    surv = {(s["a"], s["b"]): s for s in conf["populations"]["candidate"]["survivors"]}
    assert surv[("raw_buy:A", "raw_buy:B")]["confirmed"] == "yes"
    assert surv[("raw_buy:C", "raw_buy:D")]["confirmed"] == "early+late only"   # fails in E0


def test_sell_pairs_use_raw_sells():
    from research.indicator_screen import screen_pairs as SP

    assert SP.signal_columns("label:sell", COLS) == ["raw_sell:S"]
    assert "raw_sell:S" not in SP.signal_columns("B-pair:48", COLS)
    assert SP.populations("label:buy") == ("candidate",) and SP.populations("C-btc:288") == ("all", "candidate")


def test_window_lifts_match_direct_computation():
    from research.indicator_screen import screen_pairs as SP

    rng = np.random.default_rng(1)
    S = (rng.random((5000, 3)) < 0.3).astype(float)
    y = (rng.random(5000) < 0.2).astype(float)
    single, pair = SP.window_lifts(S, y, 5, 0.95)
    both = (S[:, 0] > 0) & (S[:, 2] > 0)
    assert abs(pair[0, 2] - y[both].mean() / y.mean()) < 1e-12
    assert abs(single[1] - y[S[:, 1] > 0].mean() / y.mean()) < 1e-12


class _Regime:
    """A fake regime provider: target defined everywhere (B) or only inside test windows (A2)."""
    kind = "regime"

    def __init__(self, panel, target):
        self.p, self.ids = panel, (target,)
        rng = np.random.default_rng(3)
        self.v = (rng.random(len(panel)) < 0.5).astype(float)
        if target.startswith("A2:"):
            inside = np.zeros(len(panel), bool)
            for w in WINS:
                if w.role == "test":
                    inside |= w.mask(panel.date)
            self.v[~inside] = np.nan

    def y(self, t, w):
        return self.v


def test_models_purge_subsample_restricted_rules_and_determinism(synth):
    from research.indicator_screen import screen_models as SM

    panel = _panel(synth)
    folds = [w for w in WINS if w.role == "test"]
    t = _Regime(panel, "B-pair:48")
    f = folds[1]
    tr = SM.train_rows(panel, t, "B-pair:48", "all", f, folds)
    assert panel.date[tr].max() < f.start - 48 * W.BAR
    assert (panel.date[tr].minute == 0).all()                     # hourly subsample
    tr_c = SM.train_rows(panel, t, "B-pair:48", "candidate", f, folds)
    assert not (panel.date[tr_c].minute == 0).all()               # candidate rows not subsampled
    a = SM.fit_fold(panel, t, "B-pair:48", f, folds, COLS, PR)
    b = SM.fit_fold(panel, t, "B-pair:48", f, folds, COLS, PR)
    assert json.dumps(a, default=str) == json.dumps(b, default=str)
    for r in a:
        if r["model"] == "tree":
            assert all(rule.count(" AND ") <= 2 for rule in r["rules"])        # <= 3 conditions
            if r["inputs"] == "restricted":
                used = {c for rule in r["rules"] for c in PR["models"]["restricted_inputs"] if c in rule}
                assert set(r["top_splits"]) <= set(PR["models"]["restricted_inputs"]) and used


def test_label_model_beats_restricted_when_a_planted_column_exists(synth):
    from research.indicator_screen import screen_models as SM
    from research.indicator_screen.screen_single import LabelTargets

    panel = _panel(synth)
    folds = [w for w in WINS if w.role == "test"]
    recs = SM.fit_fold(panel, LabelTargets(panel), "label:buy", folds[2], folds, COLS, PR)
    auc = {(r["model"], r["inputs"]): r["auc"] for r in recs}
    assert auc[("hgb", "all")] > auc[("hgb", "restricted")]
    assert all(r["population"] == "candidate" and r["subsample"] == 1 for r in recs)


def test_a2_model_on_the_first_fold_is_not_computable(synth):
    from research.indicator_screen import screen_models as SM

    panel = _panel(synth)
    folds = [w for w in WINS if w.role == "test"]
    t = _Regime(panel, "A2:48")
    r0 = SM.fit_fold(panel, t, "A2:48", folds[0], folds, COLS, PR)
    assert all(not r["computable"] and r["reason"] == "A2: no prior test windows" for r in r0)
    tr = SM.train_rows(panel, t, "A2:48", "all", folds[2], folds)
    assert len(tr) and panel.date[tr].max() < folds[2].start                   # only earlier test windows
    assert (folds[0].mask(panel.date[tr]) | folds[1].mask(panel.date[tr])).all()
