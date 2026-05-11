# TabDDPM — Tabular Diffusion GAN Type (Design)

**Status:** Draft for review
**Date:** 2026-05-11
**Scope:** Add a new `GANType.TAB_DDPM` backed by a continuous-only,
single-task, MLX-native implementation of TabDDPM (Kotelnikov et al.,
ICML 2023) to the existing GAN subsystem under
`user_data/strategies/GANs/`.

## 1. Goals and non-goals

### Goals
- Add a new GAN type that produces higher-fidelity synthetic samples
  than CTAB-GAN+ MLX on the codebase's continuous-feature training
  data, at acceptable training and sampling cost on Apple Silicon.
- Slot in through the existing `GANBackend` registry without disturbing
  any other backend or any strategy that doesn't declare
  `gan_type = GANType.TAB_DDPM`.
- Be unit-testable without spinning up a model (diffusion math is its
  own module).

### Non-goals (v1)
- **Categorical multinomial diffusion.** Categoricals are warned about
  and dropped, mirroring the MLX CTAB-GAN backends. Multinomial
  diffusion is a known-shape follow-up; the API and save format
  reserve room for it.
- **Multi-task variant.** No `MT_TAB_DDPM` in v1. Once the single-task
  version proves stable, the same model class extends to a
  dict-of-one-hot conditioning scheme with minimal restructuring.
- **PyTorch backend.** No `TabDDPMTFBackend`. The codebase already
  carries torch as an optional dep but doesn't install it in this venv.
  We don't ship a backend we won't run in production.
- **Pair conditioning.** The MLX CTAB-GAN supports it; TabDDPM v1 does
  not. Same trivial extension point (add a `pair_embed` parallel to
  `class_embed`) — deferred.
- **Classifier-free guidance.** The paper doesn't use it; we don't either.

## 2. Decisions locked in during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Backend(s) | MLX-only | Production runs MLX; no torch in venv; no public MLX port of TabDDPM exists so we port the ~400 LOC of math ourselves rather than take a new pip dep. |
| Feature support | Continuous-only | Matches MLX CTAB-GAN; trading features are predominantly continuous; multinomial diffusion is the larger half of TabDDPM's complexity and we defer it. |
| Task scope | Single-task only | Validates diffusion math on a simpler test bed; MT follow-up reuses the same scaffolding. |
| Sampling speed | T=1000 train, DDIM-50 sample | Paper-default training quality with ~20× faster sampling than full DDPM reverse. |
| File layout | Approach B (split diffusion math into its own module) | Pure-math module is independently unit-testable; isolates the highest-bug-density code from model state. |

## 3. Architecture and integration points

### Files created

```
GANs/
├── diffusion_mlx.py          ← pure-MLX diffusion math (no model imports)
├── df_tabddpm_mlx.py         ← TabDDPMMLX class (model + train/save/load)
├── backends/
│   └── tabddpm.py            ← TabDDPMMLXBackend adapter
└── tests/
    └── test_diffusion_mlx.py ← unit tests for the math module
```

### Files modified

| File | Change |
|---|---|
| `GANs/GANType.py` | Add `TAB_DDPM = auto()` enum entry |
| `GANs/GANInterface.py` | Add `_DEFAULTS[GANType.TAB_DDPM]`; add `TAB_DDPM` to `_BACKEND_MIGRATED` |
| `GANs/backends/__init__.py` | `from . import tabddpm  # noqa` |
| `GANs/balance.py` | Add `GANType.TAB_DDPM` to `_SQUEEZE_SEQ_DIM_TYPES` |
| `GANs/CreateGAN.py` | Add `_DEFAULTS_BY_TYPE[GANType.TAB_DDPM]` |
| `GANs/README.md` | Add row to type table; add usage subsection + sampling-speed note |
| `GANs/tests/README.md` | List new test class names |
| `GANs/tests/test_functional_suite.py` | Add parameterised TabDDPM test classes |
| `GANs/tests/test_quality_suite.py` | Add `TestTabDDPMQuality` |
| `user_data/strategies/AGENT_GUIDE.md` | Add row to GAN type table |

### Files unchanged

- `GANs/paths.py` — `gan_save_path` keys on `gan_type.name.lower()`, so
  saves land in `<storage>/GANs/tab_ddpm/` automatically.
- `balance.py` — TabDDPM uses the same `one_hot=` calling convention as
  WGAN and returns `(n, 1, F)`. `_generate_for_class` already handles
  this; only the squeeze set needs updating.

### No new pip dependencies

The whole implementation is pure MLX. Existing utilities used:
`mlx.core`, `mlx.nn`, `mlx.optimizers`, `mlx.utils.tree_map` (for EMA).

## 4. `diffusion_mlx.py` — the math module

Three responsibilities only. Pure-MLX, no model imports.

### Schedule

```python
@dataclass(frozen=True)
class Schedule:
    betas: mx.array                      # (T,)
    alphas: mx.array                     # (T,)
    alphas_cumprod: mx.array             # (T,)
    sqrt_alphas_cumprod: mx.array        # (T,)
    sqrt_one_minus_alphas_cumprod: mx.array  # (T,)
    posterior_variance: mx.array         # (T,)  for reference / sigma_t

def cosine_beta_schedule(T: int, s: float = 0.008) -> mx.array: ...
def make_schedule(T: int) -> Schedule: ...
```

Implements the Nichol & Dhariwal cosine β schedule (also TabDDPM's
default). All quantities precomputed once at construction; downstream
code only does indexed reads.

### Forward (training-time noising)

```python
def q_sample(x0: mx.array, t: mx.array, noise: mx.array, sched: Schedule) -> mx.array:
    """x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε"""
```

`t` is `(B,)` int32 and is used to gather the cumprod terms.

### Reverse (DDIM η=0 sampling)

```python
def ddim_sample(
    model_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    shape: Tuple[int, ...],
    cond: mx.array,
    sched: Schedule,
    num_steps: int = 50,
    key: Optional[mx.array] = None,
) -> mx.array:
    """Deterministic 50-step DDIM reverse process.

    model_fn(x_t, t, cond) -> ε̂  — the caller passes a closure over
    its trained network. No model dependency in this module.

    Returns raw x_0 (no clipping). The model class's _postprocess
    handles clipping and inverse-minmax to original feature ranges.
    """
```

Algorithm:
1. Build a sub-sequence of `num_steps` evenly spaced timesteps from
   `T-1` down to `0`.
2. Start from `x_T ~ N(0, I)` (or zeros if a deterministic key is given).
3. For each `(t, t_prev)` pair:
   - `ε̂ = model_fn(x_t, t, cond)`
   - `x̂0 = (x_t - sqrt(1-ᾱ_t)·ε̂) / sqrt(ᾱ_t)`
   - `x_{t_prev} = sqrt(ᾱ_{t_prev})·x̂0 + sqrt(1-ᾱ_{t_prev})·ε̂`
4. Return raw `x_0` (no clipping in the math module).

η=0 makes the sampler deterministic given the start noise — important
for reproducibility and unit tests.

## 5. `TabDDPMMLX` — model class (in `df_tabddpm_mlx.py`)

### Backbone — verbatim per the TabDDPM paper

```python
class _MLPBlock(nn.Module):                    # Linear → ReLU → Dropout
class _SinusoidalTimeEmbed(nn.Module):         # 128-dim sinusoidal → Linear(128, d_model) → SiLU → Linear → SiLU

class _TabDDPMMLP(nn.Module):
    def __init__(self, num_features, num_classes,
                 d_model=256, d_layers=(256, 256), dropout=0.0):
        self.x_proj      = nn.Linear(num_features, d_model)
        self.t_embed     = _SinusoidalTimeEmbed(d_model)
        self.class_embed = nn.Embedding(num_classes, d_model)
        self.blocks      = [_MLPBlock(in_, out_, dropout) for in_, out_ in pairs(d_layers)]
        self.head        = nn.Linear(d_layers[-1], num_features)

    def __call__(self, x_t, t, class_idx):
        h = self.x_proj(x_t) + self.t_embed(t) + self.class_embed(class_idx)
        for blk in self.blocks: h = blk(h)
        return self.head(h)        # predicts ε
```

### Outer class

```python
class TabDDPMMLX:
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        d_model: int = 256,
        d_layers: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        num_timesteps: int = 1000,
        num_sample_steps: int = 50,
        epochs: int = 300,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ema_decay: float = 0.999,
        eval_frequency: int = 20,
        verbose: bool = True,
    ):
        self._mlp     = _TabDDPMMLP(...)
        self._ema_mlp = _TabDDPMMLP(...)        # EMA copy for sampling
        self._sched   = make_schedule(num_timesteps)
        # store hyperparams as attrs so save() can persist them
```

### Training loop (`fit`)

`fit(data, labels, categorical_columns=None, **kwargs)`:
1. **Input shape** — coerce `data` to 2-D `(N, F)` float32 via the
   same `_data_to_2d` helper the WGAN backends use (raises on 3-D
   with `seq_len != 1`).
2. **Categoricals** — if `categorical_columns` is non-empty, log a
   warning and drop those columns (matches MLX CTAB-GAN behaviour).
3. **Feature stats** — per-column min/max → minmax-scale to `[-1, 1]`.
   Stored as `self.feature_min` / `self.feature_max` for `_postprocess`.
4. **Optimizer** — AdamW on `self._mlp.parameters()`, lr=1e-3, wd=1e-5.
5. **Per step**:
   - Draw batch `(B,)` indices into the data tensor.
   - Sample `t ~ U{0, T-1}`, shape `(B,)`.
   - Sample noise `ε ~ N(0, I)`, shape `(B, F)`.
   - `x_t = q_sample(x_0, t, ε, self._sched)`.
   - `ε̂ = self._mlp(x_t, t, class_idx)`.
   - Loss: `MSE(ε - ε̂)`.
   - Backprop + step + EMA update on `self._ema_mlp` (`tree_map`).
6. **Eval-quality early stopping** — every `eval_frequency` epochs,
   sample a small batch, compute mean-shift / std-ratio diagnostics
   against a held-out real slice, early-stop on `patience=3` rounds
   without improvement. Same pattern as `mlx_ctab_helpers.py`.
7. **Verbose logging** — per-epoch line `epoch X/Y  loss=...  ema_loss=...`.

### Sampling (`generate`)

`generate(n, one_hot) -> ndarray of shape (n, 1, F)`:
1. `class_idx = argmax(one_hot, axis=1).astype(int32)`.
2. `x_T ~ N(0, I)` shape `(n, F)`.
3. `x_0 = ddim_sample(self._ema_model_fn, (n, F), class_idx, self._sched, num_steps=self.num_sample_steps)`.
4. **`_postprocess`** — clip to `[-1, 1]`, invert minmax back to
   original feature ranges using `feature_min` / `feature_max`.
5. Return reshaped to `(n, 1, F)` so the WGAN-style squeeze path in
   `balance_single_task` handles it.

### Save / load

Same split format as `WGANMLX`:

- `tabddpm_gen_mlx.safetensors` — EMA model weights (diffusion-model
  convention: sample from the EMA, not the live model).
- `tabddpm_metadata.pkl` — pickle of:
  - ctor params (`num_features`, `num_classes`, `d_model`, `d_layers`,
    `num_timesteps`, `num_sample_steps`, `dropout`)
  - feature stats (`feature_min`, `feature_max`)
  - extra kwargs passed via `save(**extra)` (e.g. MASTER thresholds).

`TabDDPMMLX.load_from(save_path) -> (instance, metadata)` — class method
that reads the pickle, reconstructs the class with the saved ctor args,
loads the safetensors into `_ema_mlp`, restores stats, returns metadata.

## 6. `TabDDPMMLXBackend` adapter

`backends/tabddpm.py` — one class, mirrors `WGANMLXBackend` line-for-line.

```python
_TABDDPM_CTOR_KEYS: frozenset = frozenset({
    "d_model", "d_layers", "dropout",
    "num_timesteps", "num_sample_steps",
    "epochs", "batch_size", "learning_rate", "weight_decay",
    "ema_decay", "eval_frequency", "verbose",
})

@register_backend
class TabDDPMMLXBackend(GANBackend):
    GAN_TYPE     = GANType.TAB_DDPM
    PREFERS_MLX  = True

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()                  # same helper WGAN/CTAB MLX use

    def fit(self, data, labels, categorical_columns=None, **kwargs):
        from GANs.df_tabddpm_mlx import TabDDPMMLX
        data_2d    = _data_to_2d(data)           # reuse WGAN's helper
        labels_f32 = np.asarray(labels, dtype=np.float32)
        ctor_kwargs = {k: v for k, v in kwargs.items() if k in _TABDDPM_CTOR_KEYS}
        self._model = TabDDPMMLX(
            num_features=data_2d.shape[1],
            num_classes=labels_f32.shape[1],
            **ctor_kwargs,
        )
        self._model.fit(data_2d, labels_f32, categorical_columns=categorical_columns or [])

    def generate(self, n, **kwargs):
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError("generate() for TAB_DDPM requires keyword argument one_hot=<np.ndarray>")
        return self._model.generate(n, one_hot)

    def save(self, save_path, **extra_metadata):
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path):
        meta_p    = os.path.join(save_path, "tabddpm_metadata.pkl")
        weights_p = os.path.join(save_path, "tabddpm_gen_mlx.safetensors")
        if not (os.path.exists(meta_p) and os.path.exists(weights_p)):
            raise FileNotFoundError(f"No MLX-format TabDDPM model at {save_path}")
        from GANs.df_tabddpm_mlx import TabDDPMMLX
        instance = cls()
        instance._model, metadata = TabDDPMMLX.load_from(save_path)
        return instance, metadata
```

No TF backend. `resolve_backend(GANType.TAB_DDPM, prefer_mlx=False)`
surfaces a clear "no available backend" error via the existing
diagnostic path.

## 7. `GANInterface` plumbing

Three additions in `GANs/GANInterface.py`:

```python
_BACKEND_MIGRATED = {
    GANType.CTAB_GAN, GANType.MT_CTAB_GAN, GANType.CGAN,
    GANType.WGAN, GANType.MT_WGAN,
    GANType.TAB_DDPM,                                # NEW
}

_DEFAULTS[GANType.TAB_DDPM] = {
    "epochs":            300,
    "batch_size":        4096,
    "learning_rate":     1e-3,
    "weight_decay":      1e-5,
    "num_timesteps":     1000,
    "num_sample_steps":  50,
    "d_model":           256,
    "d_layers":          (256, 256),
    "dropout":           0.0,
    "ema_decay":         0.999,
    "eval_frequency":    20,
    "verbose":           True,
}
```

`_assert_generated_finite` already handles `np.ndarray` outputs — no change.

`GANs/backends/__init__.py` gains one line:

```python
from . import tabddpm  # noqa: F401  — registers TabDDPM MLX backend
```

## 8. `balance.py` integration

One line change:

```python
_SQUEEZE_SEQ_DIM_TYPES = {GANType.WGAN, GANType.TAB_DDPM}
```

`balance_single_task`'s existing `_generate_for_class` branch already
dispatches `one_hot=` to backends that aren't CTAB-GAN, so no new code
path is needed. TabDDPM returns `(n, 1, F)` and gets squeezed to
`(n, F)` before concatenation — same as WGAN today.

## 9. `CreateGAN.py` builder integration

```python
_DEFAULTS_BY_TYPE[GANType.TAB_DDPM] = {
    "name":                      "TabDDPM",
    "description":               "TabDDPM (tabular diffusion, MLX)",
    "augmentation_target_ratio": 0.4,         # same as WGAN
    "multi_task":                False,
}
```

`run_gan_training` already routes non-CTAB types through
`_run_simple_training`, which calls `interface.fit` / `interface.save`
unchanged — TabDDPM rides that path.

A concrete `CreateTabDDPM.py` is a 4-line subclass (optional —
`CreateGAN(gan_type=GANType.TAB_DDPM)` works without it).

## 10. Tests

### `tests/test_diffusion_mlx.py` (new, fast, no GPU)

- `test_cosine_schedule_shape_and_bounds` — `len(betas) == T`, every
  β ∈ (0, 1), `α̅_T < 1e-3`.
- `test_q_sample_identity_at_t0` — `q_sample(x0, t=0, ε)` returns `x0`
  within fp tolerance.
- `test_q_sample_variance_at_tT` — for unit-variance `x0` and
  unit-variance ε, `q_sample(x0, T-1, ε)` has per-column variance ≈ 1.
- `test_ddim_oracle_inversion` — oracle `model_fn` that returns the
  exact ε; `ddim_sample` recovers `x0` to `<1e-3` absolute error.
- `test_ddim_determinism` — same seed → bit-identical output.

### Functional tests (additions to `test_functional_suite.py`)

Parameterised by type name (matches WGAN/CTAB pattern). Three classes
generated:

- `TestTabDDPMFitGenContract` — `fit` then `generate(n=50, one_hot=...)`
  returns finite `(50, 1, F)` float32 with `F` matching training data.
- `TestTabDDPMFitGenSaveLoad` — save → load → generate yields same
  shapes; metadata (including MASTER thresholds passed via
  `save(min_buy_gain_threshold=..., training_type=...)`) round-trips
  through `GANInterface.load(expected=...)`.
- `TestTabDDPMFitGenInterface` — class conditioning honoured:
  `generate(one_hot=class0)` produces feature means closer to the
  class-0 real-data centroid than to the class-1 centroid.

### Quality test (additions to `test_quality_suite.py`)

`TestTabDDPMQuality` gated on `RUN_SLOW_TESTS=1`:

- Trains ~100 epochs on the existing mixture-of-Gaussians fixture.
- Statistical fidelity bars match CTAB-GAN:
  - per-feature mean shift `< 0.15σ`
  - per-feature std ratio `∈ [0.7, 1.3]`
  - no per-class mode collapse (per-class column std `> 0.3 × real`).
- Skips `test_label_fidelity_above_chance` (same rationale as
  CTAB-GAN: general tabular synthesiser, not discriminative).

### Existing test suites

- `test_gan_interface.py` — picks up the new type automatically through
  registry-based assertions; spot-check that the `_DEFAULTS` test now
  includes `TAB_DDPM`.
- `test_balance.py` — TabDDPM uses WGAN's calling convention so most
  parameterised cases pass unchanged. Add one assertion that the
  squeeze path is taken for `GANType.TAB_DDPM`.
- `test_passthrough.py` — no change needed (TabDDPM honours the same
  `passthrough_columns` flow via `balance_single_task`).

## 11. Documentation updates

- `GANs/README.md` — add row to the GAN-types table; add a "TabDDPM"
  usage subsection paralleling the WGAN one (`fit` / `generate` /
  `save` / `load` snippet); add a "Sampling speed" note explaining
  `num_sample_steps` and the DDIM-50 default.
- `GANs/tests/README.md` — add `TabDDPM` to the "Available type names"
  lists for functional and quality suites.
- `user_data/strategies/AGENT_GUIDE.md` — add a row to the GAN type
  table in the "Adding to / extending the GAN system" section. Note
  that this is a genuinely new GANType (Case B in the guide).

## 12. Deferred work (explicit non-shipping items)

Captured here so future-me doesn't re-discover the gap:

| Item | Notes |
|---|---|
| Categorical multinomial diffusion | Add a `_multinomial.py` helper module; extend `TabDDPMMLX` to accept `categorical_columns` and a `categorical_cardinalities` map; combined loss `L_simple + (Σ L_cat_i) / C`. |
| Multi-task `MT_TAB_DDPM` | New `df_mt_tabddpm_mlx.py` + `MTTabDDPMMLXBackend`; replace `class_embed` with `Dict[task] -> Embedding`, sum their outputs; use `balance_multi_task`. |
| Pair conditioning | Add `pair_embed = nn.Embedding(num_pairs, d_model)` parallel to `class_embed`; thread `pair_labels` / `pair_names` through `fit`. Save format reserves the `num_pairs` field already. |
| TF backend | Only if MPS/MLX availability ever becomes a problem in production. |
| Classifier-free guidance | Drop conditioning randomly with prob `p_uncond` during training; sample with guidance scale `w` at inference. Useful if class-conditioning ever feels weak. |

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Diffusion schedule mismatch between train and sample (most common bug class) | Pure-math module + unit tests catch it before model debugging. |
| EMA-vs-live model confusion at sample time | `_ema_mlp` is the sampled-from model; the live `_mlp` is never directly sampled. Single attribute clearly named. |
| Slow sampling (T=1000 default) | DDIM-50 at inference, configurable via `num_sample_steps`. |
| Diffusion training is loss-decoupled from sample quality | Eval-quality early stopping (mean shift / std ratio against held-out real) — same pattern as MLX CTAB-GAN. |
| Output non-finiteness | `_assert_generated_finite` in `GANInterface.generate` catches this at the boundary. `_postprocess` clips to `[-1, 1]` before inverse-minmax. |
| Quality bar regression vs CTAB-GAN+ MLX | Quality suite includes both backends on the same fixture; CI gate fails if TabDDPM falls below the existing bar. |

## 14. Acceptance criteria

The implementation is done when:

1. All new unit tests in `test_diffusion_mlx.py` pass.
2. Functional suite passes for the three new TabDDPM test classes
   (matches the WGAN pattern of passing in <60s on M-series).
3. Quality suite passes for `TestTabDDPMQuality` under
   `RUN_SLOW_TESTS=1`, meeting CTAB-GAN-equivalent statistical bars.
4. `GANs/README.md` and `AGENT_GUIDE.md` reflect the new type.
5. End-to-end smoke: a one-line concrete strategy
   (`NNNC_TabDDPM_MLX_LSTM` inheriting from `NNNC_CGP_MLX_LSTM` with
   `gan_type = GANType.TAB_DDPM`) trains, saves, loads, and produces
   `balance_single_task` augmentation without error or NaN on a 1-pair
   smoke timerange.
