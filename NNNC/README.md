# NNNC — Neural Network N-ary Classification Strategies

Single-task classifiers that predict one of three classes per bar:
sell / hold / buy.  Inherits from `BaseNNStrategy`
(`Framework/BaseNNStrategy.py`).

Each strategy here pairs an architecture (LSTM, CNN, Transformer, …)
with an optional augmentation method (CTAB-GAN+, WGAN-GP).  The MLX
variants run natively on Apple Silicon.

## Main files

| File | What it does |
|---|---|
| `NNNCStrategy.py` | Family base class.  Wires `get_classifier_type` / `get_classifier` into the `BaseNNStrategy` pipeline; concrete strategies just override these to pick their architecture. |
| `NNNClassifier.py` | TF/Keras classifier factory.  Defines `ClassifierType` enum + `create_classifier()` returning per-architecture classifier instances (LSTM, CNN, GRU, Transformer, Wavenet, Attention, MLP, etc.). |
| `NNNClassifierMLX.py` | Apple-MLX classifier factory.  Mirrors the API of `NNNClassifier.py` but with MLX-native `nn.Module` implementations.  Drop-in via `create_classifier_mlx()`. |
| `NNNC_CGP.py` | Concrete strategy: base architecture (MLP) + CTAB-GAN+ augmentation. |
| `NNNC_CGP_LSTM2.py`, `NNNC_CGP_GRU.py`, `NNNC_CGP_CNN.py`, `NNNC_CGP_Transformer.py`, `NNNC_CGP_Attention.py`, `NNNC_CGP_TCN.py`, `NNNC_CGP_VAE.py`, `NNNC_CGP_Wavenet.py` | TF variants — same augmentation, different architecture. |
| `NNNC_CGP_MLX.py`, `NNNC_CGP_MLX_*.py` | Apple-MLX variants of the same architectures. |
| `NNNC_WGAN.py`, `NNNC_WGAN_MLX.py` | WGAN-GP augmentation instead of CTAB-GAN+. |

## Adding a new variant

1. Create `NNNC_<Name>.py` here.
2. Inherit from `NNNCStrategy` (or one of the `NNNC_CGP*` shims if the
   augmentation is the same).
3. Override `get_classifier_type()` to return your `ClassifierType` (or
   `ClassifierTypeMLX`) value.
4. Run a long-timerange backtest to train and save the model.

See top-level `README.md` for build/test commands.
