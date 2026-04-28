# Sklearn — Classical-ML Classifier Strategies

Strategies that use classical sklearn classifiers (RandomForest,
XGBoost, etc.) instead of neural networks.  They share the
`BaseNNStrategy` training/augmentation pipeline so they benefit from
the same GAN-based class balancing as the NN strategies, but they work
with 2-D DataFrames `(samples, features)` rather than 3-D tensors
`(samples, seq_len, features)`.

Inherits from `BaseNNStrategy` (`Framework/BaseNNStrategy.py`) via
`SklearnStrategy`.

## Main files

| File | What it does |
|---|---|
| `SklearnStrategy.py` | Family base class.  Overrides `train_model()` and `get_predictions()` to operate on DataFrames; sets `seq_len = 1`.  Otherwise inherits everything from `BaseNNStrategy`. |
| `SklearnClassifier.py` | Classifier factory — `ClassifierType` enum and `create_classifier()`.  Wraps sklearn's classifiers (`RandomForestClassifier`, `XGBClassifier`, `LogisticRegression`, `MLPClassifier`, `LinearSVC`, `LinearDiscriminantAnalysis`, `VotingClassifier`, `StackingClassifier`, …) under the project's `ClassifierSklearn` adapter so train/save/load match the NN pipeline's expectations. |
| `Skl_RandomForest.py`, `Skl_XGBoost.py` | Concrete strategies — RF and XGBoost without augmentation. |
| `Skl_RandomForest_WGAN.py`, `Skl_XGBoost_WGAN.py` | Same classifiers with WGAN-GP augmentation via `wgan_enhance_training_data`. |
| `Skl_XGBoost_CGP.py` | XGBoost with CTAB-GAN+ augmentation via `ctab_gan_enhance_training_data`. |
| `NNDetector.py`, `test_lightgbm_detector.py` | Experimental: LightGBM-based anomaly detection.  Standalone — not yet wired into the strategy pipeline. |

## Adding a new variant

1. Create `Skl_<Name>.py` here.
2. Inherit from `SklearnStrategy` (no GAN) or one of the existing
   `_WGAN` / `_CGP` shims (with augmentation).
3. Override `get_classifier_type()` to return a `ClassifierType` value.
4. Run a long-timerange backtest to train and save.

## Why no MLX variant

sklearn classifiers run on CPU and don't benefit from Metal
acceleration.  The MLX optimisations apply only to the deep-learning
strategy families (NNNC/NNMT/Anomaly).
