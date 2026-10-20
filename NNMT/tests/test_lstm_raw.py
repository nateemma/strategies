"""040 T024 (research R6): the multi-task LSTM_RAW -- features straight into one LSTM, ARO's decode
head per task, no resize/conv preamble."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx  # noqa: E402
from Predictors.MLXClassifierMultiTask import TASK_NAMES  # noqa: E402


def _model(nfeat=24, seq=16):
    clf, name = create_classifier_mlx(ClassifierTypeMLX.LSTM_RAW, "SOL/USDT", nfeat, seq)
    return clf.create_model(seq, nfeat), name


def test_forward_gives_one_softmax_per_task():
    m, name = _model()
    assert name == "LSTM_RAW"
    out = m(mx.random.normal((5, 16, 24)))
    assert set(out) == set(TASK_NAMES)
    for t in TASK_NAMES:
        assert tuple(out[t].shape) == (5, 3)
        assert mx.allclose(out[t].sum(axis=-1), mx.ones(5), atol=1e-5).item()


def test_no_conv_and_the_lstm_reads_the_raw_features():
    m, _ = _model(nfeat=10)
    assert "Conv1d" not in str(m)
    assert m.lstm.Wx.shape[1] == 10  # input weight reads the 10 features directly (no resize projection)
    assert len(m.task_heads) == len(TASK_NAMES)
