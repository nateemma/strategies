# NNMT — Neural Network Multi-Task Classification Strategies

Multi-task classifiers — instead of predicting only the trading action,
the model predicts six related labels at once: trading, regime, risk,
momentum, flow, profit.  Forcing the network to balance across multiple
heads acts as a regulariser and produces a model that's less prone to
overfitting any single task.

Inherits from `BaseNNStrategy` (`Framework/BaseNNStrategy.py`) via
`NNMTStrategy`.

## Main files

| File | What it does |
|---|---|
| `NNMTStrategy.py` | Family base class.  Builds the multi-task label dict (`{"trading": …, "regime": …, …}`), runs `mt_ctab_gan_enhance_training_data` for augmentation, and wires the multi-task classifier into the standard NN pipeline. |
| `NNMTClassifier.py` | TF/Keras multi-task classifier factory.  `ClassifierType` enum + `create_classifier()`; 14 architecture variants split into "Normal" (LSTM, Transformer, CNN, GRU, Wavenet, Wavenet_Fast, Attention) which override the shared backbone, and "Multi_*" variants which override every per-task head. |
| `NNMTClassifierMLX.py` | Apple-MLX multi-task classifier factory.  `ClassifierTypeMLX` enum + `create_classifier_mlx()`.  Mirrors the TF version's 14 architectures, but every Multi_\* variant uses the same head architecture for all six tasks (no per-task branching). |
| `NNMT_CGP.py`, `NNMT_CGP_Attention.py` | Concrete strategies using CTAB-GAN+ multi-task augmentation. |
| `NNMT_WGAN.py`, `NNMT_WGAN_MLX.py` | Concrete strategies using multi-task WGAN-GP augmentation. |

## Multi-task label format

The trainer expects labels as a dict, not a single one-hot array:

```python
{
    "trading":  one_hot,  # (N, 3) sell/hold/buy
    "regime":   one_hot,  # (N, 3) bear/sideways/bull
    "risk":     one_hot,  # (N, 3) low/normal/high
    "momentum": one_hot,  # (N, 3) negative/stable/positive
    "flow":     one_hot,  # (N, 3) decrease/neutral/increase
    "profit":   one_hot,  # (N, 3) loss/neutral/profit
}
```

This is what the classifiers (`NNMTClassifier` / `NNMTClassifierMLX`)
and the multi-task GANs (`MT_WGAN`, `MT_CTAB_GAN`) all consume.

## Adding a new variant

1. Create `NNMT_<Name>.py` here.
2. Inherit from `NNMTStrategy` (or one of the `NNMT_CGP*` / `NNMT_WGAN*`
   shims if the augmentation is the same).
3. Override `get_classifier_type()` to pick a `ClassifierType` /
   `ClassifierTypeMLX` value.
4. Run a long-timerange backtest to train and save.
