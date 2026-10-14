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
