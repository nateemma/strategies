"""040 T008 (research R8): NNMT exposes every head's class probabilities as %-columns, aligned with the
predictions, NaN outside the predicted rows, and consistent with the argmax classes it trades on."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from NNMT.NNMTStrategy import NNMTStrategy  # noqa: E402

N, SEQ = 40, 8
HEADS = ("trading", "regime", "risk", "momentum", "flow", "profit")


def _probs(rows, seed):
    p = np.random.default_rng(seed).uniform(0.05, 1.0, (rows, 3))
    return p / p.sum(axis=1, keepdims=True)


def _strategy(threshold=0.0):
    s = NNMTStrategy.__new__(NNMTStrategy)
    rows = N - SEQ + 1
    preds = {h: _probs(rows, i) for i, h in enumerate(HEADS)}
    s.scale_dataframe = lambda df: df
    s.dataframeUtils = SimpleNamespace(df_to_tensor=lambda df, seq, method=0: np.zeros((rows, seq, 2)))
    s.prediction_threshold = SimpleNamespace(value=threshold)
    s.profit_prediction_threshold = SimpleNamespace(value=threshold)
    s.apply_task_filters = SimpleNamespace(value=False)
    s.min_consecutive_buys = SimpleNamespace(value=1)
    s.seq_len = SEQ
    for name in ("debug_print", "print_probability_stats", "print_distribution_compact"):
        setattr(s, name, lambda *a, **k: None)
    return s, SimpleNamespace(predict=lambda t: preds), preds


def _run(threshold=0.0):
    s, clf, preds = _strategy(threshold)
    df = pd.DataFrame({"close": np.arange(N, dtype=float)})
    out = s.process_predictions(df, s.get_predictions(df, clf))
    return out, preds


def test_every_head_has_its_probability_columns_with_nan_padding():
    out, preds = _run()
    off = N - len(preds["trading"])
    for c in ("%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy"):
        assert out[c].iloc[:off].isna().all() and out[c].iloc[off:].notna().all()
    np.testing.assert_allclose(out["%predict_prob_buy"].iloc[off:], preds["trading"][:, 2])
    for head, names in NNMTStrategy.HEAD_CLASSES.items():
        for k, c in enumerate(names):
            col = out[f"%p_{head}_{c}"]
            assert col.iloc[:off].isna().all()
            np.testing.assert_allclose(col.iloc[off:], preds[head][:, k])


def test_argmax_of_the_columns_matches_the_traded_classes_without_a_threshold():
    out, preds = _run(threshold=0.0)
    off = N - len(preds["trading"])
    for head, names in NNMTStrategy.HEAD_CLASSES.items():
        p = out[[f"%p_{head}_{c}" for c in names]].iloc[off:].to_numpy()
        assert (p.argmax(axis=1) == out[f"%{head}"].iloc[off:].to_numpy()).all()
    buy = out[["%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy"]].iloc[off:].to_numpy().argmax(axis=1) == 2
    assert (buy == (out["predict_buy"].iloc[off:].to_numpy() == 1)).all()


def test_probability_columns_never_become_model_inputs():
    """The feature path drops every ^% column (DataframeUtils.remove_debug_columns), so none can leak."""
    from utils.DataframeUtils import DataframeUtils

    out, _ = _run()
    kept = DataframeUtils.remove_debug_columns(object.__new__(DataframeUtils), out.copy())
    assert not [c for c in kept.columns if c.startswith(("%p_", "%predict_prob"))]
    assert "close" in kept.columns
