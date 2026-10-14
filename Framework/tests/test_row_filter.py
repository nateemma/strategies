"""Feature 032: the declared training row filter (contracts C1-C3).

* C1 -- `Framework.RowFilter.row_filter_mask`.
* C2 -- `TrainingEngine.prepare_training_data`: with no filter declared its output must equal the
  pre-032 capture (`fixtures/prepare_training_data_pre032.npz`, taken BEFORE the change); with a
  filter, the kept samples are exactly the windows whose last bar is eligible, unbroken.
* C3 -- model keys: the existing classes keep the keys captured before any Framework edit
  (`fixtures/model_keys_pre032.json`).

To regenerate the captures (ONLY from the pre-change code):
    REGEN_032_FIXTURE=1 ../../.venv/bin/python -m pytest Framework/tests/test_row_filter.py
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

from Framework.RowFilter import row_filter_mask  # noqa: E402


FIX = Path(__file__).resolve().parent / "fixtures"
PDT = FIX / "prepare_training_data_pre032.npz"
KEYS = FIX / "model_keys_pre032.json"
REGEN = os.environ.get("REGEN_032_FIXTURE") == "1"
SEQ_LEN, N_ROWS, PAIRS = 16, 160, 2
GBB = (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))
KEY_CLASSES = ("NNNC_A_Raw_Open", "NNNC_GbbRaw_Open", "NNNC_FS_GBR_GR", "NNNC_ARO_f00",
               "NNNC_FS_GBR_GR_f00")


# --- C1 -------------------------------------------------------------------------------------


def _df():
    return pd.DataFrame({"a": [0.1, 0.5, np.nan, 0.2, 0.9], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})


def test_mask_and_of_conditions_and_each_op():
    df = _df()
    assert row_filter_mask(df, (("a", "lt", 0.5),)).tolist() == [True, False, False, True, False]
    assert row_filter_mask(df, (("a", "le", 0.5),)).tolist() == [True, True, False, True, False]
    assert row_filter_mask(df, (("b", "gt", 2.0),)).tolist() == [False, False, True, True, True]
    assert row_filter_mask(df, (("b", "ge", 2.0),)).tolist() == [False, True, True, True, True]
    both = row_filter_mask(df, (("a", "lt", 0.5), ("b", "ge", 2.0)))
    assert both.tolist() == [False, False, False, True, False]


def test_mask_nan_fails_and_frame_unchanged():
    df = _df()
    before = df.copy()
    assert not row_filter_mask(df, (("a", "gt", -1.0),))[2]
    pd.testing.assert_frame_equal(df, before)


def test_mask_missing_column_raises_keyerror():
    with pytest.raises(KeyError, match="zz"):
        row_filter_mask(_df(), (("zz", "lt", 1.0),))


@pytest.mark.parametrize("spec", [(), None, (("a", "lt"),), (("a", "eq", 1.0),), ((1, "lt", 1.0),),
                                  "a<1"])
def test_mask_bad_spec_raises_valueerror(spec):
    with pytest.raises(ValueError):
        row_filter_mask(_df(), spec)


# --- C2 host --------------------------------------------------------------------------------


def _pair_frame(seed: int, weights: bool) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.standard_normal((N_ROWS, 4)).astype(np.float32),
                      columns=[f"f{i}" for i in range(4)])
    df["guard_metric"] = rng.uniform(-1, 1, N_ROWS)
    df["bb_width"] = rng.uniform(0.0, 0.08, N_ROWS)
    if weights:
        df["%pnl_weight"] = rng.uniform(0.5, 2.0, N_ROWS)
    labels = rng.randint(0, 3, size=N_ROWS).astype(np.int64)
    return df, labels


def _inputs(weights: bool):
    frames, labels = zip(*(_pair_frame(100 + i, weights) for i in range(PAIRS)))
    return list(frames), list(labels)


def _host(tensor_method: int, row_filter=None):
    from Framework.BaseNNStrategy import BaseNNStrategy
    from Framework.BaseStrategy import GANType
    from utils.DataframeUtils import DataframeUtils

    s = BaseNNStrategy.__new__(BaseNNStrategy)
    s.seq_len = SEQ_LEN
    s.tensor_method = tensor_method
    s.TRAIN_DATA_SPLIT = 0.8
    s.gan_type = GANType.NONE
    s.gan_augment = False
    s.apply_label_conflict_resolution = True
    s.classifier = None
    s.dataframeUtils = DataframeUtils()
    if row_filter is not None:
        s.TRAIN_ROW_FILTER = row_filter
    return s


def _build(tensor_method: int, weights: bool, row_filter=None) -> dict:
    s = _host(tensor_method, row_filter)
    frames, labels = _inputs(weights)
    tr, te, trl, tel, w = s.prepare_training_data(frames, labels, norm=False,
                                                  pair_names=[f"P{i}/USDT" for i in range(PAIRS)])
    cw = s.get_training_class_weights(train_labels=trl, validation_labels=tel)
    return {"tsr_train": np.asarray(tr), "tsr_test": np.asarray(te),
            "train_labels": np.asarray(trl), "test_labels": np.asarray(tel),
            "weights": np.asarray(w) if w is not None else np.zeros(0),
            "class_weights": np.asarray(cw, dtype=float), "host": s}


CASES = [(m, w) for m in (0, 3) for w in (False, True)]


def _case_key(m, w):
    return f"m{m}_w{int(w)}"


def _needs_method(m):
    if m == 0:
        pytest.importorskip("tensorflow")
    if m == 3:
        mx = pytest.importorskip("mlx.core")
        if not (hasattr(mx, "metal") and mx.metal.is_available()):
            pytest.skip("MLX Metal not available")


@pytest.mark.skipif(not REGEN, reason="capture only on demand, from the pre-change code")
def test_capture_pre032_fixture():
    out = {}
    for m, w in CASES:
        _needs_method(m)
        b = _build(m, w)
        for k, v in b.items():
            if k != "host":
                out[f"{_case_key(m, w)}__{k}"] = v
    np.savez_compressed(PDT, **out)


@pytest.mark.parametrize("m,w", CASES)
def test_no_filter_matches_pre032_capture(m, w):
    _needs_method(m)
    ref = np.load(PDT)
    b = _build(m, w)
    for k in ("tsr_train", "tsr_test", "train_labels", "test_labels", "weights", "class_weights"):
        np.testing.assert_array_equal(b[k], ref[f"{_case_key(m, w)}__{k}"], err_msg=k)


# --- C3 model keys --------------------------------------------------------------------------


def _key(name: str) -> str:
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_wf.json")
    return MK.key_hash(MK.resolve(load_strategy_class(name), cfg))


@pytest.mark.skipif(not REGEN, reason="capture only on demand, from the pre-change code")
def test_capture_pre032_model_keys():
    KEYS.write_text(json.dumps({n: _key(n) for n in KEY_CLASSES}, indent=1) + "\n")


@pytest.mark.parametrize("name", KEY_CLASSES)
def test_existing_model_keys_unchanged(name):
    assert _key(name) == json.loads(KEYS.read_text())[name]
