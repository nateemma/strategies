"""040 architecture amendment (preregistration_arch.json): LSTM_STACK -- a shared LSTM returns the whole
sequence and each task runs its OWN LSTM over it before ARO's decode head (option 3; LSTM_RAW is option 1:
shared LSTM, MLP head per task)."""
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
    clf, name = create_classifier_mlx(ClassifierTypeMLX.LSTM_STACK, "SOL/USDT", nfeat, seq)
    return clf.create_model(seq, nfeat), name


def test_forward_gives_one_softmax_per_task():
    m, name = _model()
    assert name == "LSTM_STACK"
    out = m(mx.random.normal((5, 16, 24)))
    assert set(out) == set(TASK_NAMES)
    for t in TASK_NAMES:
        assert tuple(out[t].shape) == (5, 3)
        assert mx.allclose(out[t].sum(axis=-1), mx.ones(5), atol=1e-5).item()


def test_each_task_has_its_own_lstm_on_the_shared_sequence():
    m, _ = _model(nfeat=10)
    assert "Conv1d" not in str(m)
    assert m.lstm.Wx.shape[1] == 10  # the shared LSTM reads the features directly
    nf = m.lstm.Wh.shape[1]
    assert len(m.task_lstms) == len(m.task_heads) == len(TASK_NAMES)
    for lstm in m.task_lstms:
        assert lstm.Wx.shape[1] == nf  # each task LSTM reads the shared LSTM's per-step output
    assert len({id(x) for x in m.task_lstms}) == len(TASK_NAMES)  # separate weights, not one shared module

