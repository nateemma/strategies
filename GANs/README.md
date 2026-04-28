# GANs — Synthetic Data Augmentation

Provides GAN-based minority-class oversampling for Freqtrade strategies.
All GAN types are accessed through a single unified entry point: `GANInterface`.

The strategy never picks a GAN-specific code path itself — it declares
`gan_type` and the rest is dispatched.  Single-task and multi-task variants
share the same lifecycle, the same save layout, and the same balanced-augmentation
helpers (`balance_single_task` / `balance_multi_task` in `balance.py`).

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

Note that the MT variants operate on tensors, not dataframes.

---

## Save layout

Every GAN type for a strategy lives under one parent directory, keyed by
`gan_type.name.lower()`:

```
<storage>/GANs/
    wgan/                  GANType.WGAN
    ctab_gan/              GANType.CTAB_GAN
    mt_wgan/               GANType.MT_WGAN
    mt_ctab_gan/           GANType.MT_CTAB_GAN
    cgan/                  GANType.CGAN
```

PCA-reduced strategies use `GANs_PCA/<type>/` instead.  The convention is
defined in `GANs/paths.py::gan_save_path` — every consumer (creator
scripts, `BaseNNStrategy.enhance_training_data`, debug tools) goes
through it, so changing the layout is a one-place edit.

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

## Saving and validating metadata

`save()` accepts keyword arguments that are stored alongside the model and returned by `load()`.  Use this to record strategy-level thresholds, training_type, column order, or anything else the consumer needs to verify when it loads the model later:

```python
iface.save(
    min_buy_gain_threshold=0.016,
    min_sell_loss_threshold=-0.012,
    training_type=2,
)
```

`load()` accepts an optional `expected` mapping.  Every key is compared
against the persisted metadata; any mismatch (or missing key) raises
`GANMetadataMismatchError` with a per-key diff so the operator can
decide whether to retrain the GAN or update the strategy.  This is
strict by design — silently using a GAN whose thresholds drifted from
the strategy's current ones would produce labels generated under one
threshold combined with a generator trained under another, corrupting
training:

```python
from GANs.GANInterface import GANInterface, GANMetadataMismatchError

iface = GANInterface(GANType.CTAB_GAN, save_path="...")
try:
    metadata = iface.load(expected={
        "min_buy_gain_threshold": self.MIN_BUY_GAIN_THRESHOLD,
        "min_sell_loss_threshold": self.MIN_SELL_LOSS_THRESHOLD,
        "training_type": int(self.TRAINING_TYPE),
    })
except GANMetadataMismatchError as e:
    # The exception's message lists each drifted key, the saved value,
    # and the value the strategy expected — surface and stop.
    raise
```

`load()` without `expected` returns the metadata dict unchecked
(backwards-compatible).

`BaseNNStrategy.enhance_training_data` already does this — strategies
get strict validation for free by setting `gan_type`.  Override
`_gan_expected_metadata(train_df)` to add type-specific keys (e.g.
`column_order`, `num_features`) on top of the default thresholds /
training_type checks.

---

## Class-balanced augmentation (`balance.py`)

Two public helpers orchestrate the per-class generation loop on top of
`GANInterface`.  The strategy code never picks GAN-specific kwargs —
the helpers dispatch on `interface.gan_type`:

### `balance_single_task` — for WGAN / CTAB_GAN / CGAN

```python
from GANs.balance import balance_single_task

aug_data, aug_labels = balance_single_task(
    interface=iface,
    data=train_minmax,                   # (N, F) ndarray or DataFrame
    labels=train_labels,                 # 1-D class indices or 2-D one-hot
    target_ratio=0.8,                    # float, or {class_idx: float}
    passthrough_columns=["dow_sin", ...],  # names for DataFrame, indices for ndarray
)
```

Loops over classes, computes per-class deficits against
`ratio * majority_count`, calls `interface.generate(...)` with the right
kwarg for the backend (`one_hot=` for WGAN/CGAN, `class_label=` for
CTAB-GAN), swaps passthrough columns from real samples, and concatenates.

### `balance_multi_task` — for MT_WGAN / MT_CTAB_GAN

```python
from GANs.balance import balance_multi_task

aug_data, aug_labels = balance_multi_task(
    interface=iface,
    data=train_minmax,                   # 2-D or 3-D ndarray, or DataFrame
    labels={"trading": ..., "regime": ...},   # dict of one-hot arrays
    target_ratios=0.8,                   # float (broadcast), Dict[task, float],
                                         # or Dict[task, Dict[class, float]]
)
```

Runs a deficit-driven greedy loop: each round picks the largest
`(task, class)` deficit, asks the GAN for one batch with that class
one-hot for the *target* task and the *other* tasks' labels sampled
from each task's own current deficit distribution.  Solves the
cross-task interference problem that a naive per-task loop can't
(where balancing one task re-skews the others).

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
