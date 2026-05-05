# Predictors

Task-type-organized predictor hierarchy. Sits parallel to `utils/Classifier*`,
which remains authoritative for behavior. Strategies should target classes
in this package.

## Hierarchy

```
BasePredictor
├── BaseClassifier
│   ├── KerasBaseClassifier   → KerasClassifierNary, KerasClassifierMultiTask
│   ├── MLXBaseClassifier     → MLXClassifierNary,   MLXClassifierMultiTask
│   └── SklearnBaseClassifier (no further subclasses; sklearn variants live in Sklearn/)
├── BaseRegressor
│   └── KerasRegressorLinear  (only one regressor today; no framework base)
└── BaseAnomalyDetector
    └── KerasAnomalyDetector  (only one detector today; no framework base)
```

## Design

The bases (`BasePredictor`, `BaseClassifier`, `BaseRegressor`,
`BaseAnomalyDetector`) are pure marker classes. They define no methods —
they exist so `isinstance(x, BaseClassifier)` works for dispatch and so
the inheritance tree expresses the task-type taxonomy.

`KerasBaseClassifier` and `MLXBaseClassifier` are also pure markers — they
do NOT inherit from `utils.ClassifierKeras` / `utils.ClassifierMLX`. Doing
so created a broken diamond MRO at runtime: `utils/ClassifierKeras.py`
gets loaded twice (once via `utils.ClassifierKeras`, once via the bare
sibling-style `ClassifierKeras` import that
`utils/ClassifierKerasMultiTask.py` and friends use), and the two copies
become different class objects. C3 linearization then placed
`utils.ClassifierKeras.ClassifierKeras` BEFORE
`utils.ClassifierKerasMultiTask.ClassifierKerasMultiTask` in the MRO,
breaking the cooperative `super().__init__()` chain.

Concrete classes (`KerasClassifierNary`, `KerasClassifierMultiTask`,
`MLXClassifierNary`, `MLXClassifierMultiTask`) inherit from a marker
base PLUS the corresponding `utils.Classifier*` variant — that single
utils inheritance gives them all behavior, with no diamond.

`SklearnBaseClassifier` is the exception: it DOES inherit from
`utils.ClassifierSklearn`, because the strategy-side `SklearnClassifier_*`
concrete classes inherit from it directly (not via a `Sklearn*Variant`
intermediate), and there's no equivalent diamond risk in the sklearn
chain (no `utils.ClassifierSklearnVariant(ClassifierSklearn)` files).

`KerasRegressorLinear` and `KerasAnomalyDetector` skip the framework base
entirely — each is the only variant in its category — and inherit
`(BaseRegressor, ClassifierKerasLinear)` / `(BaseAnomalyDetector,
ClassifierKerasAnomaly)` directly. No diamond either.

This is a parallel hierarchy: `utils/Classifier*` is unmodified, and
non-migrated code can keep importing from there. Newer strategies should
import from `Predictors/` so the task-type axis is explicit.

## Adding a new predictor

1. Decide its task type (classifier / regressor / anomaly detector).
2. Create the file in this directory. Use multiple inheritance to combine
   the appropriate marker base with the implementation source:

   ```python
   from Predictors.KerasBaseClassifier import KerasBaseClassifier
   from utils.ClassifierKerasMyNewVariant import ClassifierKerasMyNewVariant

   class KerasClassifierMyNewVariant(KerasBaseClassifier, ClassifierKerasMyNewVariant):
       """Description."""
       pass
   ```

3. If your variant introduces a new framework (e.g. PyTorch), add the
   framework-task base first (`PyTorchBaseClassifier`) just as the existing
   ones do, then the concrete class.

## Currently migrated strategies

| Family   | File                              | Predictors parent            | Subclass count |
| -------- | --------------------------------- | ---------------------------- | -------------- |
| NNNC     | `NNNC/NNNClassifier.py`           | `KerasClassifierNary`        | 17             |
| NNNC     | `NNNC/NNNClassifierMLX.py`        | `MLXClassifierNary`          | 13             |
| NNMT     | `NNMT/NNMTClassifier.py`          | `KerasClassifierMultiTask`   | 15 (via _Base) |
| NNMT     | `NNMT/NNMTClassifierMLX.py`       | `MLXClassifierMultiTask`     | 15 (via _Base) |
| Sklearn  | `Sklearn/SklearnClassifier.py`    | `SklearnBaseClassifier`      | 18             |
| Anomaly  | `Anomaly/NNAnomalyStrategy.py`    | `KerasAnomalyDetector`       | 0 (type hints) |

`NNMTClassifier` and `NNMTClassifierMLX` use a `_Base` intermediate, so only
the `_Base` class actually inherits from the Predictors hierarchy — the
LSTM/Transformer variants in those files inherit from `_Base`.

**Anomaly caveat:** `NNAnomalyStrategy.py` references `KerasAnomalyDetector`
only in type hints (`get_classifier` return type, `get_predictions` parameter
type). The actual classifier files `Anomaly/NNAnomalyClassifier.py` and
`Anomaly/NNGANomalyClassifier.py` are pre-existing broken (missing
`utils.AnyAnomaly` module) and were not migrated. The type hints are
aspirational — when those files are fixed, they should be re-parented onto
`Predictors.KerasAnomalyDetector` (or a sibling) so the annotations match
runtime behavior.
