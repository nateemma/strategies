"""Feature 033: the width-relative pooled relabel (contract C1).

* `TrainingSignals.path_outcome` -- the close-based gbb_path / gbb_path_sell trade outcome.
* `Framework.PooledLabel.relabel` -- per-side width buckets fitted on the training split, top
  share BUY / bottom share SELL, `%wr_keep` for the row filter.
* The `TrainingEngine._train_keyed` hook -- absent spec: the labels reaching `train_model` equal the
  pre-033 capture (`fixtures/train_keyed_labels_pre033.npz`, taken BEFORE the change).
* Model keys -- `fixtures/model_keys_pre033.json`, captured before any Framework/NNNC edit.

To regenerate the captures (ONLY from the pre-change code):
    REGEN_033_FIXTURE=1 ../../.venv/bin/python -m pytest Framework/tests/test_pooled_label.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "Framework", STRAT / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


FIX = Path(__file__).resolve().parent / "fixtures"
TKL = FIX / "train_keyed_labels_pre033.npz"
KEYS = FIX / "model_keys_pre033.json"
REGEN = os.environ.get("REGEN_033_FIXTURE") == "1"
N_ROWS, PAIRS = 200, 2
KEY_CLASSES = ("NNNC_A_Raw_Open", "NNNC_GbbRaw_Open", "NNNC_FS_GBR_GR", "NNNC_CO_ARO",
               "NNNC_CO_GBR", "NNNC_ARO_f00", "NNNC_FS_GBR_GR_f00")


# --- fixtures (captured from the pre-033 code) ----------------------------------------------


def _pair_frame(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N_ROWS)))
    df = pd.DataFrame({"close": close, "f0": rng.standard_normal(N_ROWS),
                       "guard_metric": rng.uniform(-1, 1, N_ROWS),
                       "bb_width": rng.uniform(0.0, 0.08, N_ROWS)})
    return df, rng.randint(0, 3, size=N_ROWS).astype(np.int64)


def _inputs():
    frames, labels = zip(*(_pair_frame(300 + i) for i in range(PAIRS)))
    return list(frames), list(labels)


class _Recorder:
    """Stands in for TrainingEngine: `_train_keyed` with a recording `train_model`."""

    def __init__(self, **attrs):
        from Framework.TrainingEngine import TrainingEngine

        self._train_keyed = TrainingEngine._train_keyed.__get__(self)
        self.TRAIN_DATA_SPLIT = 0.8
        self.classifier = None
        self.config = {}
        self.seen = None
        for k, v in attrs.items():
            setattr(self, k, v)

    def train_model(self, frames, labels, classifier, pair_names=None):
        self.seen = ([f.copy() for f in frames], [np.asarray(x).copy() for x in labels])


def _keyed_labels(**attrs):
    r = _Recorder(**attrs)
    frames, labels = _inputs()
    r._train_keyed(frames, labels, None, pair_names=[f"P{i}/USDT" for i in range(PAIRS)])
    return r


@pytest.mark.skipif(not REGEN, reason="capture only on demand, from the pre-change code")
def test_capture_pre033_train_keyed_labels():
    r = _keyed_labels()
    np.savez(TKL, **{f"p{i}": lab for i, lab in enumerate(r.seen[1])})


def test_no_spec_labels_match_pre033_capture():
    r = _keyed_labels()
    cap = np.load(TKL)
    assert len(r.seen[1]) == PAIRS
    for i, lab in enumerate(r.seen[1]):
        np.testing.assert_array_equal(lab, cap[f"p{i}"])
    assert all("%wr_keep" not in f.columns for f in r.seen[0])


def _key(name: str) -> str:
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_wf.json")
    return MK.key_hash(MK.resolve(load_strategy_class(name), cfg))


@pytest.mark.skipif(not REGEN, reason="capture only on demand, from the pre-change code")
def test_capture_pre033_model_keys():
    KEYS.write_text(json.dumps({n: _key(n) for n in KEY_CLASSES}, indent=1) + "\n")


@pytest.mark.parametrize("name", KEY_CLASSES)
def test_existing_model_keys_unchanged(name):
    assert _key(name) == json.loads(KEYS.read_text())[name]


# --- C1: path_outcome -----------------------------------------------------------------------


def _closes(xs):
    return pd.DataFrame({"close": np.asarray(xs, dtype=float)})


def test_path_outcome_buy_barriers_and_horizon():
    from Framework.TrainingSignals import path_outcome

    # bar 0: +2.1% at h=2 -> target; bar 1: 95 / 100.5 = -5.47% at h=2 -> stop first
    df = _closes([100, 100.5, 102.1, 95.0, 96.0, 96.5, 96.2])
    out, bars = path_outcome(df, target=0.02, stop=-0.05, horizon=3, side="buy")
    assert out[0] == 0.02 and bars[0] == 2
    assert out[1] == -0.05 and bars[1] == 2          # 95 / 100.5 - 1 = -5.47%
    assert out[3] == pytest.approx(96.2 / 95.0 - 1) and bars[3] == 3   # horizon close
    assert np.isnan(out[6]) and bars[6] == -1         # last bar: no future
    assert np.isnan(out[5]) and bars[5] == -1         # 1 bar left, no barrier hit


def test_path_outcome_near_end_barrier_counts():
    from Framework.TrainingSignals import path_outcome

    df = _closes([100, 100, 103])  # bar 0 hits +2% at h=2 though horizon 48 runs past the end
    out, bars = path_outcome(df, target=0.02, stop=-0.05, horizon=48, side="buy")
    assert out[0] == 0.02 and bars[0] == 2
    assert out[1] == 0.02 and bars[1] == 1


def test_path_outcome_sell_mirror():
    from Framework.TrainingSignals import path_outcome

    df = _closes([100, 99.0, 97.9, 103.0, 101.0])
    out, bars = path_outcome(df, target=-0.02, stop=0.022, horizon=3, side="sell")
    assert out[0] == -0.02 and bars[0] == 2           # 97.9 = -2.1% first
    assert out[2] == 0.022 and bars[2] == 1           # 103 / 97.9 = +5.2% first: take-profit


@pytest.mark.parametrize("side,target,stop", [("buy", -0.02, -0.05), ("buy", 0.02, 0.05),
                                              ("sell", 0.02, 0.022), ("sell", -0.02, -0.01),
                                              ("up", 0.02, -0.05)])
def test_path_outcome_bad_signs_raise(side, target, stop):
    from Framework.TrainingSignals import path_outcome

    with pytest.raises(ValueError):
        path_outcome(_closes([1, 2, 3]), target=target, stop=stop, horizon=2, side=side)


def _real_frame():
    p = STRAT.parent / "data" / "binanceus" / "ETH_USDT-15m.feather"
    if not p.is_file():
        pytest.skip("no local ETH 15m data")
    d = pd.read_feather(p)
    return d.iloc[-3000:].reset_index(drop=True)  # about a month


@pytest.mark.parametrize("real", [False, True])
def test_path_outcome_matches_gbb_path_labels(real):
    """outcome == target exactly where labels_gbb_path (rule gates off) is 1, on every bar."""
    from Framework.TrainingSignals import labels_gbb_path, labels_gbb_path_sell, path_outcome

    if real:
        df = _real_frame()
    else:
        rng = np.random.RandomState(7)
        df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 1500)))})
    before = df.copy()
    buy = labels_gbb_path(df, guard_threshold=1e9, bb_width_threshold=-1e9, min_gain=0.02,
                          horizon=48, stop_frac=-0.05).to_numpy()
    out, _ = path_outcome(df, target=0.02, stop=-0.05, horizon=48, side="buy")
    np.testing.assert_array_equal(buy == 1, out == 0.02)
    sell = labels_gbb_path_sell(df, guard_threshold=-1e9, bb_width_threshold=-1e9, min_loss=0.02,
                                horizon=48, take_profit=0.022).to_numpy()
    out_s, _ = path_outcome(df, target=-0.02, stop=0.022, horizon=48, side="sell")
    np.testing.assert_array_equal(sell == 1, out_s == -0.02)
    assert (buy == 1).sum() > 10 and (sell == 1).sum() > 10
    pd.testing.assert_frame_equal(df, before)


# --- C1: relabel ----------------------------------------------------------------------------


def _spec(**over):
    s = {"target": 0.02, "stop": -0.05, "horizon": 48, "reference": "bb_width",
         "eligible": (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035)),
         "exit_eligible": (("guard_metric", "gt", 0.3), ("bb_width", "gt", 0.035)),
         "exit_loss": 0.02, "exit_take_profit": 0.022, "buckets": 10, "share": 0.5,
         "min_bucket": 20}
    s.update(over)
    return s


def test_assign_hand_computed_buckets_ties_and_cut():
    from Framework.PooledLabel import assign

    # two buckets by ref (0.1 x4 | 0.9 x4); in bucket A outcomes tie at the target (0.02) and
    # the tie-break (-bars) decides; bucket B is plain. share 0.5 -> 2 of 4 per bucket.
    o = np.array([0.02, 0.02, 0.02, -0.05, 0.01, 0.005, -0.01, 0.003])
    s2 = np.array([-5, -2, -9, 3, 0, 0, 0, 0], dtype=float)
    ref = np.array([0.1] * 4 + [0.9] * 4)
    pos = np.arange(8)
    sel, rep = assign(o, s2, ref, pos, np.ones(8, bool), 0.5, 2, 1, "buy")
    assert sel.tolist() == [True, True, False, False, True, True, False, False]
    assert [b["pos_share"] for b in rep] == [0.5, 0.5]
    sel_s, _ = assign(o, s2, ref, pos, np.ones(8, bool), 0.5, 2, 1, "sell")
    assert sel_s.tolist() == [False, False, True, True, False, False, True, True]


def test_assign_fit_rows_set_the_cut_and_validation_follows():
    from Framework.PooledLabel import assign

    rng = np.random.RandomState(1)
    n = 400
    o, ref = rng.normal(0, 0.01, n), rng.uniform(0, 1, n)
    s2, pos = np.zeros(n), np.arange(n)
    fit = np.arange(n) < 300
    sel, rep = assign(o, s2, ref, pos, fit, 0.5, 4, 1, "buy")
    o2 = o.copy()
    o2[~fit] = rng.normal(0, 0.01, (~fit).sum())  # change only validation outcomes
    sel2, rep2 = assign(o2, s2, ref, pos, fit, 0.5, 4, 1, "buy")
    assert [b["cut"] for b in rep] == [b["cut"] for b in rep2]
    assert [b["lo"] for b in rep] == [b["lo"] for b in rep2]
    np.testing.assert_array_equal(sel[fit], sel2[fit])
    for b in rep:  # validation labelled by the cut
        assert b["val_n"] > 0 and 0.2 < b["val_pos_share"] < 0.8


def test_assign_merges_small_buckets():
    from Framework.PooledLabel import assign

    ref = np.r_[np.zeros(50), np.ones(5)]  # quantile edges leave a 5-bar bucket
    o = np.linspace(-0.01, 0.01, 55)
    sel, rep = assign(o, np.zeros(55), ref, np.arange(55), np.ones(55, bool), 0.5, 4, 20, "buy")
    assert all(b["n"] >= 20 for b in rep) and rep[0]["merged"]


def _frames(n=1500, pairs=2, width_corr=True):
    out = []
    for i in range(pairs):
        rng = np.random.RandomState(40 + i)
        bw = rng.uniform(0.02, 0.12, n)
        # wider bands -> bigger moves: the outcome is width-correlated by construction (R3 case)
        ret = rng.normal(0, 1, n) * (0.002 + (0.08 * bw if width_corr else 0.004))
        out.append(pd.DataFrame({"close": 100 * np.exp(np.cumsum(ret)),
                                 "guard_metric": rng.uniform(-1, 1, n), "bb_width": bw}))
    return out


def test_relabel_shares_spearman_keep_and_classes():
    from Framework.PooledLabel import KEEP_COL, relabel

    frames = _frames()
    labels = [np.full(len(f), 1, dtype=np.int64) for f in frames]
    out, rep = relabel(frames, labels, _spec(), fit_ends=[1200, 1200])
    for b in rep["buckets"]:
        assert abs(b["pos_share"] - 0.5) <= 1 / b["n"] + 1e-9
    assert abs(rep["spearman_buy"]) <= 0.05 and abs(rep["spearman_sell"]) <= 0.05
    assert {b["side"] for b in rep["buckets"]} == {"buy", "sell"}
    for f, lab in zip(frames, out):
        k = f[KEEP_COL].to_numpy()
        buy_el = (f.guard_metric < 0.3) & (f.bb_width > 0.035)
        sell_el = (f.guard_metric > 0.3) & (f.bb_width > 0.035)
        assert set(lab[(k == 1) & buy_el]) <= {1, 2} and set(lab[(k == 1) & sell_el]) <= {0, 1}
        assert np.all(lab[~(buy_el | sell_el)] == 1)          # untouched (all HOLD on input)
        assert np.all(np.isnan(k[-1:]))                         # no future at the last bar
        assert np.all(np.isnan(k[~(buy_el | sell_el).to_numpy()]))


def test_relabel_keep_column_drives_the_row_filter():
    from Framework.PooledLabel import KEEP_COL, relabel
    from Framework.RowFilter import row_filter_mask

    frames = _frames(600)
    relabel(frames, [np.ones(len(f), int) for f in frames], _spec(min_bucket=5))
    for f in frames:
        m = row_filter_mask(f, ((KEEP_COL, "gt", 0.5),))
        np.testing.assert_array_equal(m, f[KEEP_COL].to_numpy() == 1.0)


def test_relabel_deterministic_and_only_keep_column_added():
    from Framework.PooledLabel import KEEP_COL, relabel

    a, b = _frames(600), _frames(600)
    before = [f.copy() for f in a]
    la, _ = relabel(a, [np.ones(len(f), int) for f in a], _spec(min_bucket=5))
    lb, _ = relabel(b, [np.ones(len(f), int) for f in b], _spec(min_bucket=5))
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(x, y)
    for f, g in zip(a, before):
        pd.testing.assert_frame_equal(f.drop(columns=[KEEP_COL]), g)


def test_relabel_overlap_and_bad_spec_and_missing_column():
    from Framework.PooledLabel import relabel

    fr = _frames(300)
    ones = [np.ones(len(f), int) for f in fr]
    with pytest.raises(ValueError, match="overlap"):
        relabel(fr, ones, _spec(exit_eligible=(("bb_width", "gt", 0.035),), min_bucket=5))
    for bad in ({"share": 1.0}, {"buckets": 1}, {"target": -0.1}, {"reference": 3}):
        with pytest.raises(ValueError):
            relabel(_frames(300), ones, _spec(**bad))
    with pytest.raises(KeyError):
        relabel(_frames(300), ones, _spec(reference="nope"))
    with pytest.raises(ValueError):
        relabel(_frames(300), ones, {"target": 0.02})


# --- C1: the TrainingEngine hook ------------------------------------------------------------


def test_hook_relabels_with_training_fit_ends():
    from Framework.PooledLabel import KEEP_COL, relabel

    spec = _spec(min_bucket=5, horizon=12)
    r = _keyed_labels(POOLED_LABEL=spec, seq_len=16)
    frames, labels = _inputs()
    fit_ends = [int(0.8 * len(f)) - 15 for f in frames]
    want, rep = relabel(frames, labels, spec, fit_ends)
    for got, exp in zip(r.seen[1], want):
        np.testing.assert_array_equal(got, exp)
    assert all(KEEP_COL in f.columns for f in r.seen[0])
    assert r._pooled_label_report["kept"] == rep["kept"]


def test_hook_writes_width_relative_json_beside_fresh_weights(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from Framework import ModelKey as MK

    monkeypatch.setattr(MK, "training_lock", lambda loc: nullcontext())
    monkeypatch.setattr(MK, "record_training", lambda *a, **k: None)
    loc = SimpleNamespace(kind="absent", prefix="p", hash="h", path=str(tmp_path / "m" / "w.bin"))
    (tmp_path / "m").mkdir()

    class _Clf:
        def model_exists(self):
            return True

    r = _keyed_labels(POOLED_LABEL=_spec(min_bucket=5, horizon=12), seq_len=16,
                      _model_location=loc, classifier=_Clf())
    out = json.loads((tmp_path / "m" / "width_relative.json").read_text())
    assert out["kept"] == r._pooled_label_report["kept"] and out["buckets"]
