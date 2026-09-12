"""Tests for the meta-labelling feature (011, brief 03).

T003 asserts the archive still matches the constants the whole feature is calibrated to.
Every other test pins a defect the analyze pass or an earlier feature actually found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import metalabel as M          # noqa: E402


@pytest.fixture(scope="module")
def archive():
    return json.loads((ROOT / M.ARCHIVE_PATH).read_text())


# --- T003: the archive still matches the pinned constants ------------------

def test_archive_shape_matches_pinned_constants(archive):
    """If any of these has moved, the FLOOR and BASE_WIN_RATE must be recomputed
    before anything else runs (quickstart S1). They are measurements, not choices."""
    assert archive["n_trades_pooled"] == M.N_ARCHIVE_TRADES
    assert archive["folds_complete"] == 15


def test_evaluation_span_matches(archive):
    ev = [t for t in archive["trades"] if t["fold_id"] in M.EVAL_FOLDS]
    assert len(ev) == M.N_EVAL_TRADES
    warm = [t for t in archive["trades"] if t["fold_id"] in M.WARMUP_FOLDS]
    assert len(warm) == M.N_WARMUP_TRADES


def test_base_win_rate_matches(archive):
    """73.2% is why accuracy is never reported: a model accepting everything scores it."""
    ev = [t for t in archive["trades"] if t["fold_id"] in M.EVAL_FOLDS]
    rate = sum(1 for t in ev if t["profit_abs"] > 0) / len(ev)
    assert rate == pytest.approx(M.BASE_WIN_RATE, abs=0.001)


def test_floor_is_not_the_study_wide_figure():
    """The study-wide 3.350 pp/yr was measured on 378 trades over 3.68 yr. Inheriting it
    here would accept deltas this span cannot resolve (research R2)."""
    assert M.FLOOR_PP_PER_YEAR > 4.0
    assert M.FLOOR_PP_PER_YEAR != 3.350


def test_recomputed_floor_reproduces(archive):
    """Recompute R2's bootstrap from the archive and confirm the pinned value."""
    ev = [t for t in archive["trades"] if t["fold_id"] in M.EVAL_FOLDS]
    per = [f for f in archive["per_fold"] if f["fold_id"] in M.EVAL_FOLDS]
    years = sum(f["test_days"] for f in per) / 365.0
    pnl = np.array([t["profit_abs"] for t in ev]) * ev[0]["scale_pp_per_abs"]
    rng = np.random.default_rng(M.BOOTSTRAP_SEED)
    bs = rng.choice(pnl, size=(M.N_RESAMPLES, len(pnl)), replace=True).sum(axis=1)
    hw = (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2
    assert hw / years == pytest.approx(M.FLOOR_PP_PER_YEAR, abs=0.01)


def test_per_fold_model_stores_exist():
    """The feature is unbuildable without these; reinfer.py refuses to run on a missing one
    rather than silently training on a subset."""
    missing = [f for f in M.EVAL_FOLDS + M.WARMUP_FOLDS
               if not (ROOT / M.MODEL_STORE / f"NNNC_WF_{f}").is_dir()]
    assert not missing, f"missing per-fold model stores: {missing}"


# --- Gate A (T005, T006): the debug columns ---------------------------------

PROB_COLS = ("%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy")


def test_prob_columns_are_dropped_as_debug_columns():
    """T005. The '%' prefix is what stops the model's own output entering the feature list.
    Without it this is a predict-your-own-prediction leak (research R1)."""
    import importlib.util
    import pandas as pd

    # Load DataframeUtils by FILE, not by mutating sys.path. Inserting ROOT/"utils" at the
    # front of sys.path leaks into every later test in the same session and shadows sibling
    # modules -- it made test_purge_embargo.py fail when run after this file, while both
    # passed in isolation.
    _spec = importlib.util.spec_from_file_location(
        "_dfutils_for_test", ROOT / "utils" / "DataframeUtils.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DataframeUtils = _mod.DataframeUtils

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0],
                       **{c: [np.nan, 0.9, 0.4] for c in PROB_COLS}})
    out = DataframeUtils().remove_debug_columns(df)
    assert "close" in out.columns
    for c in PROB_COLS:
        assert c not in out.columns


def _write_prob_columns(dataframe, pred_probs, original_length, predicted_length, offset):
    """Mirror of the write in BaseNNStrategy.get_predictions(), for shape testing."""
    for cls, name in enumerate(PROB_COLS):
        col = np.full(original_length, np.nan, dtype=float)
        if pred_probs is not None and pred_probs.ndim == 2:
            col[offset:offset + predicted_length] = pred_probs[:predicted_length, cls]
        dataframe[name] = col
    return dataframe


def test_short_predictions_are_written_at_the_right_offset_and_padded_with_nan():
    """T006. `predictions` is front-padded when the model emits fewer rows than the
    dataframe; pred_probs gets no such treatment, so a raw assignment misaligns every row.

    NaN, not 0.0: zero is a real probability meaning 'confidently not this class', so
    zero-padding would mark every warm-up bar a confident non-buy and silently corrupt the
    earliest feature rows."""
    import pandas as pd
    original, predicted = 10, 6
    offset = original - predicted
    probs = np.tile(np.array([[0.1, 0.2, 0.7]]), (predicted, 1))
    df = _write_prob_columns(pd.DataFrame({"close": np.arange(original, dtype=float)}),
                             probs, original, predicted, offset)

    buy = df["%predict_prob_buy"].to_numpy()
    assert np.isnan(buy[:offset]).all(), "warm-up region must be NaN"
    assert not np.isnan(buy[offset:]).any()
    assert buy[offset:] == pytest.approx(0.7)
    assert (buy[:offset] != 0.0).all() or np.isnan(buy[:offset]).all()


def test_full_length_predictions_use_zero_offset():
    import pandas as pd
    n = 5
    probs = np.tile(np.array([[0.3, 0.3, 0.4]]), (n, 1))
    df = _write_prob_columns(pd.DataFrame({"close": np.arange(n, dtype=float)}),
                             probs, n, n, 0)
    assert not df["%predict_prob_buy"].isna().any()


def test_probabilities_land_in_the_right_class_columns():
    """A transposed or mis-indexed write would be silent: all three columns would be
    populated and only their meaning would be wrong."""
    import pandas as pd
    n = 4
    probs = np.tile(np.array([[0.1, 0.2, 0.7]]), (n, 1))   # sell, hold, buy
    df = _write_prob_columns(pd.DataFrame({"close": np.arange(n, dtype=float)}),
                             probs, n, n, 0)
    assert df["%predict_prob_sell"].iloc[0] == pytest.approx(0.1)
    assert df["%predict_prob_hold"].iloc[0] == pytest.approx(0.2)
    assert df["%predict_prob_buy"].iloc[0] == pytest.approx(0.7)
