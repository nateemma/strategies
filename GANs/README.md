# GANs — Synthetic Data Augmentation

Provides GAN-based minority-class oversampling for Freqtrade strategies.
All GAN types are accessed through a single unified entry point: `GANInterface`.

---

## GAN types

| Type | Class | Input | Labels | Backend |
|---|---|---|---|---|
| `WGAN` | WGAN-GP | numpy `(N, F)` | one-hot `(N, C)` | TF or MLX |
| `MT_WGAN` | WGAN-GP multi-task | numpy `(N, seq_len, F)` | dict of one-hot arrays | TF or MLX |
| `CTAB_GAN` | CTAB-GAN+ | DataFrame | one-hot `(N, C)` | TF or MLX |
| `MT_CTAB_GAN` | CTAB-GAN+ multi-task | DataFrame | dict of one-hot arrays | TF |
| `CGAN` | Conditional GAN | numpy `(N, seq_len, F)` | one-hot `(N, C)` | TF |
| `BOTH` | WGAN pre-pass + CTAB-GAN | — | — | TF |
| `NONE` | No augmentation | — | — | — |

**WGAN-GP** trains fast on 2-D tabular data and is a good default.
**MT_WGAN** extends WGAN to handle multiple simultaneous label tasks (e.g. trading signal + market regime).
**CTAB-GAN+** uses VGM preprocessing to model the real data distribution more faithfully, especially for mixed continuous/categorical tables.
**CGAN** is a sequential conditional GAN suited to time-series inputs.

Note that the MT variants operate on tensors, not dataframes

---

## API

All GAN types use the explicit lifecycle: `fit()` / `generate()` / `save()` / `load()`.

```python
from GANs.GANInterface import GANInterface
from GANs.GANType import GANType
```

### WGAN

```python
iface = GANInterface(GANType.WGAN, save_path="/path/to/model/dir")

# Train
iface.fit(data, labels_one_hot)          # data: (N, F), labels: (N, C)
iface.save()

# Later — load and generate
iface.load()
one_hot = np.zeros((50, num_classes), dtype="float32")
one_hot[:, target_class] = 1.0
gen_data = iface.generate(50, one_hot=one_hot)   # returns (50, 1, F)
```

### MT_WGAN

```python
iface = GANInterface(GANType.MT_WGAN, save_path="/path/to/model/dir")
iface.fit(data, {"trading": trading_oh, "regime": regime_oh})
iface.save()

iface.load()
gen_data, gen_labels = iface.generate(
    50,
    task_labels={"trading": trading_oh_50, "regime": regime_oh_50},
)
# gen_data: (50, seq_len, F),  gen_labels: dict of (50, C) arrays
```

### CTAB_GAN

```python
import pandas as pd

iface = GANInterface(GANType.CTAB_GAN, save_path="/path/to/model/dir")

# Train — data must be a DataFrame; labels are one-hot numpy (N, C)
iface.fit(train_df, labels_one_hot)
iface.save()

# Inference
iface.load()
gen_df = iface.generate(100, class_label=1)   # returns pd.DataFrame
```

### MT_CTAB_GAN

```python
iface = GANInterface(GANType.MT_CTAB_GAN, save_path="/path/to/model/dir")
iface.fit(train_df, {"trading": trading_oh, "regime": regime_oh})
iface.save()

iface.load()
gen_df, gen_labels = iface.generate(
    100,
    task_labels={"trading": trading_oh_100, "regime": regime_oh_100},
)
```

### CGAN

```python
iface = GANInterface(GANType.CGAN, save_path="/path/to/model/dir")

# Train — data must be 3D (N, seq_len, features)
iface.fit(data_3d, labels_one_hot)
iface.save()

# Inference
iface.load()
one_hot = np.zeros((50, num_classes), dtype="float32")
one_hot[:, target_class] = 1.0
gen_data = iface.generate(50, one_hot=one_hot)   # returns (50, seq_len, F)
```

---

## Saving optional metadata

`save()` accepts keyword arguments that are stored alongside the model and returned by `load()`.  This is useful for recording strategy-level thresholds that were determined at training time:

```python
iface.save(
    min_buy_gain_threshold=0.016,
    min_sell_loss_threshold=-0.012,
    training_type=2,
)

meta = iface.load()
threshold = meta["min_buy_gain_threshold"]
```

---

## MLX acceleration

On Apple Silicon, `CTAB_GAN` and `WGAN` automatically use an MLX backend when available.
Pass `prefer_mlx=False` to force the TensorFlow backend:

```python
iface = GANInterface(GANType.WGAN, save_path="...", prefer_mlx=False)
```

---

## Tests

See [`tests/README.md`](tests/README.md) for instructions on running the functional and quality test suites.
