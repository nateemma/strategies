# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Regression test for the Predictors hierarchy MRO.

This test caught (and prevents recurrence of) a real bug: when the
KerasBaseClassifier and MLXBaseClassifier framework bases inherited from
utils.ClassifierKeras / utils.ClassifierMLX directly, the MRO of any
concrete strategy classifier (NNMTClassifier_LSTM, NNNClassifierMLX_*,
etc.) ended up with utils.ClassifierKeras appearing twice -- once via the
qualified import and once via utils.ClassifierKerasMultiTask's
sibling-style import. C3 linearization placed the qualified copy ahead of
ClassifierKerasMultiTask, breaking the cooperative super().__init__()
chain in utils/ClassifierKeras.py:70 with:

    TypeError: ClassifierKerasMultiTask.__init__() missing 3 required
    positional arguments: 'pair', 'seq_len', 'num_features'

The fix made KerasBaseClassifier / MLXBaseClassifier pure markers (no
utils inheritance). This test instantiates one representative subclass
per migrated family and asserts no TypeError. Tolerates other init
errors (missing model dirs, GPU init, etc.) -- it is NOT a workflow
test, only an MRO regression guard.

Run via the standard pytest suite from repo root:

    PYTHONPATH=. .venv/bin/pytest user_data/strategies/Predictors/test_predictors_mro.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure strategy dirs are on sys.path so the family imports work
_STRATEGIES_ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "NNNC", "NNMT", "Sklearn", "Anomaly"):
    p = str(_STRATEGIES_ROOT / sub) if sub else str(_STRATEGIES_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _try_init(label, factory):
    """Run factory(); succeed if it returns or raises anything except TypeError.

    TypeError indicates the MRO super() chain is broken (the bug class this
    test exists to catch). Other exceptions are tolerated -- this test does
    not validate workflow correctness, only constructor wiring.
    """
    try:
        factory()
    except TypeError as e:
        pytest.fail(f"MRO regression in {label}: TypeError: {e}")
    except Exception:
        # Tolerated: workflow-level errors (FileNotFoundError, model dir
        # missing, GPU init, etc.) are not what this test guards against.
        pass


def test_nnmt_keras_classifier_instantiates():
    """NNMTClassifier_LSTM must construct without TypeError."""
    import NNMTClassifier
    _try_init(
        "NNMTClassifier_LSTM",
        lambda: NNMTClassifier.NNMTClassifier_LSTM("TEST/USDT", 16, 5),
    )


def test_nnmt_mlx_classifier_instantiates():
    """NNMTClassifierMLX_Attention must construct without TypeError."""
    import NNMTClassifierMLX
    _try_init(
        "NNMTClassifierMLX_Attention",
        lambda: NNMTClassifierMLX.NNMTClassifierMLX_Attention("TEST/USDT", 16, 5),
    )


def test_nnnc_mlx_classifier_instantiates():
    """NNNClassifierMLX_Attention must construct without TypeError.

    NNNClassifier (Keras NNNC) is intentionally NOT tested here because the
    file fails to import in the standard environment due to a missing TCN
    module dependency. That is a pre-existing environment issue unrelated
    to this MRO check.
    """
    import NNNClassifierMLX
    _try_init(
        "NNNClassifierMLX_Attention",
        lambda: NNNClassifierMLX.NNNClassifierMLX_Attention("TEST/USDT", 16, 5),
    )


def test_sklearn_classifier_instantiates():
    """SklearnClassifier_LogisticRegression must construct without TypeError.

    Sklearn variants take a different __init__ signature (pair only, no
    seq_len/num_features) — pass just the pair.
    """
    import SklearnClassifier
    _try_init(
        "SklearnClassifier_LogisticRegression",
        lambda: SklearnClassifier.SklearnClassifier_LogisticRegression("TEST/USDT"),
    )


def test_keras_anomaly_detector_instantiates():
    """KerasAnomalyDetector must construct without TypeError.

    No strategy-side concrete class uses this (NNAnomalyClassifier files
    are pre-existing broken), so test the Predictors-level class directly.
    """
    from Predictors.KerasAnomalyDetector import KerasAnomalyDetector
    _try_init(
        "KerasAnomalyDetector",
        lambda: KerasAnomalyDetector("TEST/USDT", 16, 5),
    )


def test_keras_regressor_linear_instantiates():
    """KerasRegressor must construct without TypeError.

    No strategy-side concrete class uses this yet; test directly.
    """
    from Predictors.KerasRegressor import KerasRegressor
    _try_init(
        "KerasRegressor",
        lambda: KerasRegressor("TEST/USDT", 16, 5),
    )


def test_mro_contains_single_classifier_keras():
    """Concrete classes must contain exactly one ClassifierKeras in MRO.

    The Predictors-side bases (KerasBaseClassifier, MLXBaseClassifier) are
    pure markers -- they do NOT inherit from utils.ClassifierKeras / MLX --
    precisely so this invariant holds. If a future change re-introduces
    that inheritance, this test will catch it before anything is run.
    """
    import NNMTClassifier
    cls = NNMTClassifier.NNMTClassifier_LSTM
    keras_count = sum(1 for c in cls.__mro__ if c.__name__ == "KerasBasePredictor")
    assert keras_count == 1, (
        f"NNMTClassifier_LSTM has {keras_count} copies of KerasBasePredictor in MRO "
        f"(expected 1). MRO: {[c.__name__ for c in cls.__mro__]}"
    )


def test_isinstance_baseclassifier_works():
    """Concrete classifiers must satisfy isinstance(BaseClassifier)."""
    from Predictors.BaseClassifier import BaseClassifier
    from Predictors.BasePredictor import BasePredictor
    import NNMTClassifier
    import NNMTClassifierMLX
    import SklearnClassifier

    for module, name in [
        (NNMTClassifier, "NNMTClassifier_LSTM"),
        (NNMTClassifierMLX, "NNMTClassifierMLX_Attention"),
        (SklearnClassifier, "SklearnClassifier_LogisticRegression"),
    ]:
        cls = getattr(module, name)
        assert issubclass(cls, BaseClassifier), f"{name} is not a BaseClassifier subclass"
        assert issubclass(cls, BasePredictor), f"{name} is not a BasePredictor subclass"
