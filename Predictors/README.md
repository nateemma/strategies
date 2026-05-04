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

Framework-specific classes (`KerasBaseClassifier`, `MLXClassifierNary`, etc.)
combine a marker base with the corresponding `utils.Classifier*` class via
multiple inheritance. All actual training, prediction, save, load behavior
comes from `utils/`. The Predictors classes are thin facades.

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
