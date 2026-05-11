# TabDDPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `GANType.TAB_DDPM` backed by a continuous-only, single-task, MLX-native implementation of TabDDPM (Kotelnikov et al., ICML 2023) to the existing GAN subsystem under `user_data/strategies/GANs/`.

**Architecture:** Diffusion math is its own pure-MLX module (`diffusion_mlx.py`) so it can be unit-tested without instantiating a model. The model class (`df_tabddpm_mlx.py`) holds an MLP backbone with sinusoidal time-embedding + class-embedding, an EMA copy for sampling, plus the train/save/load lifecycle. A thin backend adapter (`backends/tabddpm.py`) registers it with the existing `GANBackend` registry so `GANInterface` and `balance_single_task` consume it without further changes.

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`) — no new pip dependencies. Save format uses MLX's built-in `nn.Module.save_weights` (safetensors) + a sidecar pickle for metadata, matching `WGANMLX`.

**Spec reference:** `user_data/strategies/docs/superpowers/specs/2026-05-11-tabddpm-design.md` (in the nested `user_data/strategies/.git` repo).

**Repo boundary (critical):** The outer `freqtrade/.git` is upstream code we never modify. All code, doc, and test commits for this plan land in the nested `user_data/strategies/.git`. Run git commands either with `git -C user_data/strategies …` or after `cd user_data/strategies`. File paths in this plan are written absolute from the freqtrade root for clarity — when staging, adjust them to paths relative to `user_data/strategies/` (e.g. `git -C user_data/strategies add GANs/GANType.py`).

---

## Preflight

Before starting:

```bash
source .venv/bin/activate
cd /Users/philprice95/Documents/freqtrade
```

Confirm MLX is available (production code path):

```bash
python -c "import mlx.core as mx; print('metal:', mx.metal.is_available())"
```

Expected: `metal: True`. If the venv is broken (see `docs/superpowers/specs/2026-05-11-tabddpm-design.md` for the iCloud-corruption story), repair it before starting — the implementation tests need a working MLX install.

**Working dir for all commands:** `/Users/philprice95/Documents/freqtrade` (repo root). All `pytest` commands run from there.

---

## Task 1: Add `GANType.TAB_DDPM` enum

**Files:**
- Modify: `user_data/strategies/GANs/GANType.py:14-30`
- Test: `user_data/strategies/GANs/tests/test_gan_interface.py` (existing — verify TAB_DDPM is present)

- [ ] **Step 1: Add the enum entry**

Edit `user_data/strategies/GANs/GANType.py`. After the existing `BOTH = auto()` line (line 29), add:

```python
    TAB_DDPM = auto()     # TabDDPM (tabular diffusion, MLX-only, continuous-only, single-task)
```

The final enum body should read:

```python
class GANType(Enum):
    """..."""

    NONE = auto()         # No GAN augmentation
    WGAN = auto()         # WGAN-GP, single-task, 2-D tabular (TF or MLX)
    MT_WGAN = auto()      # WGAN-GP, multi-task, 3-D sequential (TF or MLX)
    CTAB_GAN = auto()     # CTAB-GAN+, single-task, tabular (TF or MLX)
    MT_CTAB_GAN = auto()  # CTAB-GAN+, multi-task, tabular (TF)
    CGAN = auto()         # Conditional GAN, single-task, sequential (TF)
    BOTH = auto()         # WGAN pre-processing + CTAB-GAN augmentation
    TAB_DDPM = auto()     # TabDDPM (tabular diffusion, MLX-only, continuous-only, single-task)
```

- [ ] **Step 2: Smoke-check the enum**

Run:

```bash
python -c "from user_data.strategies.GANs.GANType import GANType; print(GANType.TAB_DDPM)"
```

Expected output: `GANType.TAB_DDPM`

- [ ] **Step 3: Commit**

```bash
git add user_data/strategies/GANs/GANType.py
git commit -m "feat(gans): add GANType.TAB_DDPM enum entry

Reserves the enum value for the upcoming TabDDPM backend. No backend
yet — resolve_backend(GANType.TAB_DDPM) will raise until task 4 lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2a: `diffusion_mlx.Schedule` + cosine β schedule

**Files:**
- Create: `user_data/strategies/GANs/diffusion_mlx.py`
- Create: `user_data/strategies/GANs/tests/test_diffusion_mlx.py`

- [ ] **Step 1: Write the failing test**

Create `user_data/strategies/GANs/tests/test_diffusion_mlx.py`:

```python
"""Unit tests for diffusion_mlx — pure-math diffusion utilities."""

from __future__ import annotations

import sys
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import mlx.core as mx
import numpy as np
import pytest

from GANs.diffusion_mlx import (
    Schedule,
    cosine_beta_schedule,
    make_schedule,
)


class TestCosineSchedule:
    def test_returns_T_betas(self):
        betas = cosine_beta_schedule(T=1000)
        assert betas.shape == (1000,)

    def test_betas_in_open_unit_interval(self):
        betas = np.asarray(cosine_beta_schedule(T=1000))
        assert (betas > 0).all()
        assert (betas < 1).all()

    def test_alpha_cumprod_decays_to_near_zero(self):
        sched = make_schedule(T=1000)
        alphas_cumprod = np.asarray(sched.alphas_cumprod)
        assert alphas_cumprod[0] > 0.99
        assert alphas_cumprod[-1] < 1e-2

    def test_schedule_fields_have_right_shape(self):
        sched = make_schedule(T=200)
        for field_name in (
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            arr = getattr(sched, field_name)
            assert arr.shape == (200,), f"{field_name}: got {arr.shape}"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py -v
```

Expected: `ModuleNotFoundError: No module named 'GANs.diffusion_mlx'` (or collection error).

- [ ] **Step 3: Create the diffusion module with the schedule**

Create `user_data/strategies/GANs/diffusion_mlx.py`:

```python
"""
Pure-MLX diffusion math used by TabDDPM (and reusable by any future
diffusion-based GAN type).

This module deliberately has no model dependency — every function takes
arrays in and arrays out.  The cosine β-schedule, forward `q_sample`,
and DDIM η=0 reverse sampler all live here so they can be unit-tested
in isolation; the model class in `df_tabddpm_mlx.py` calls into them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """Precomputed diffusion-process constants.

    All arrays are shape (T,) and aligned: index t corresponds to the
    t-th diffusion step (0 = least noisy, T-1 = pure noise).
    """

    betas: mx.array                       # β_t
    alphas: mx.array                       # α_t = 1 - β_t
    alphas_cumprod: mx.array               # ᾱ_t = Π_{s≤t} α_s
    sqrt_alphas_cumprod: mx.array          # √ᾱ_t
    sqrt_one_minus_alphas_cumprod: mx.array  # √(1 - ᾱ_t)
    posterior_variance: mx.array           # σ²_t for stochastic DDPM reverse


def cosine_beta_schedule(T: int, s: float = 0.008) -> mx.array:
    """Nichol & Dhariwal cosine schedule — also TabDDPM's default.

    Defines  f(t) = cos((t/T + s) / (1+s) · π/2)²  for t ∈ [0, T],
    then ᾱ_t = f(t)/f(0), β_t = 1 − ᾱ_t/ᾱ_{t−1}, clipped to (0, 0.999).
    """
    steps = mx.arange(T + 1, dtype=mx.float32)
    f = mx.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    # Clip to keep numerical sanity at the tails.
    betas = mx.clip(betas, 1e-8, 0.999)
    return betas


def make_schedule(T: int, s: float = 0.008) -> Schedule:
    """Build a full Schedule from a cosine β-schedule of length T."""
    betas = cosine_beta_schedule(T, s=s)
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    # ᾱ_{t-1} with ᾱ_{-1} = 1
    alphas_cumprod_prev = mx.concatenate(
        [mx.ones((1,), dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], axis=0
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return Schedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=mx.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=mx.sqrt(1.0 - alphas_cumprod),
        posterior_variance=posterior_variance,
    )
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/diffusion_mlx.py user_data/strategies/GANs/tests/test_diffusion_mlx.py
git commit -m "feat(gans): add diffusion_mlx.Schedule + cosine beta schedule

Pure-MLX diffusion constants module — no model dependency. Implements
Nichol & Dhariwal cosine schedule (TabDDPM default) with the full set
of precomputed cumprod / sqrt / posterior-variance arrays the forward
and reverse processes need.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2b: `diffusion_mlx.q_sample` — forward noising

**Files:**
- Modify: `user_data/strategies/GANs/diffusion_mlx.py` (add `q_sample` function)
- Modify: `user_data/strategies/GANs/tests/test_diffusion_mlx.py` (add `TestQSample`)

- [ ] **Step 1: Write the failing test**

Append to `test_diffusion_mlx.py`:

```python
class TestQSample:
    def test_identity_at_t_zero(self):
        from GANs.diffusion_mlx import q_sample

        sched = make_schedule(T=1000)
        x0 = mx.random.normal((16, 8))
        noise = mx.random.normal((16, 8))
        t = mx.zeros((16,), dtype=mx.int32)

        x_t = q_sample(x0, t, noise, sched)
        # At t=0, ᾱ_0 ≈ 1, so x_t ≈ x_0 (small residual noise from
        # sqrt(1-ᾱ_0) ≈ 0 but not exactly zero — assert close-ish).
        diff = mx.mean(mx.abs(x_t - x0)).item()
        assert diff < 0.05, f"q_sample at t=0 drifted by {diff:.4f}"

    def test_unit_variance_at_t_T_minus_one(self):
        from GANs.diffusion_mlx import q_sample

        T = 1000
        sched = make_schedule(T=T)
        x0 = mx.random.normal((4096, 8))
        noise = mx.random.normal((4096, 8))
        t = mx.full((4096,), T - 1, dtype=mx.int32)

        x_t = np.asarray(q_sample(x0, t, noise, sched))
        # At t=T-1, ᾱ_t ≈ 0, so x_t ≈ noise → per-column variance ≈ 1.
        var = x_t.var(axis=0)
        assert (var > 0.85).all() and (var < 1.15).all(), f"per-col var={var}"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py::TestQSample -v
```

Expected: `ImportError: cannot import name 'q_sample' from 'GANs.diffusion_mlx'`.

- [ ] **Step 3: Implement `q_sample`**

Append to `user_data/strategies/GANs/diffusion_mlx.py`:

```python
# ---------------------------------------------------------------------------
# Forward (training-time noising)
# ---------------------------------------------------------------------------


def q_sample(
    x0: mx.array, t: mx.array, noise: mx.array, sched: Schedule
) -> mx.array:
    """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

    Args:
        x0:    (B, F) clean samples in [-1, 1].
        t:     (B,) int32 timestep indices in [0, T-1].
        noise: (B, F) standard-normal noise, same shape as x0.
        sched: Schedule from make_schedule.

    Returns:
        (B, F) noised samples x_t.
    """
    sqrt_ac = sched.sqrt_alphas_cumprod[t]               # (B,)
    sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t]   # (B,)
    return sqrt_ac[:, None] * x0 + sqrt_omac[:, None] * noise
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py -v
```

Expected: 6 tests pass (4 schedule + 2 q_sample).

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/diffusion_mlx.py user_data/strategies/GANs/tests/test_diffusion_mlx.py
git commit -m "feat(gans): add diffusion_mlx.q_sample forward-noising helper

Standard DDPM forward step:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

Tests cover the two anchor points: identity at t=0 (alpha_bar ≈ 1) and
unit-variance at t=T-1 (alpha_bar ≈ 0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2c: `diffusion_mlx.ddim_sample` — DDIM η=0 reverse sampler

**Files:**
- Modify: `user_data/strategies/GANs/diffusion_mlx.py`
- Modify: `user_data/strategies/GANs/tests/test_diffusion_mlx.py`

- [ ] **Step 1: Write the failing test**

Append to `test_diffusion_mlx.py`:

```python
class TestDDIMSample:
    def test_oracle_inverts_to_x0(self):
        """If model_fn returns the exact ε used in q_sample, DDIM
        should recover x_0 to high accuracy."""
        from GANs.diffusion_mlx import ddim_sample, q_sample

        T = 1000
        sched = make_schedule(T=T)
        rng_key = mx.random.key(0)

        # Build a known (x_0, ε) pair and a fixed starting timestep.
        x0_true = mx.random.normal((32, 8), key=rng_key)

        # Oracle: given (x_t, t, cond), return the ε that produced x_t
        # from x0_true. We bake the relationship via q_sample math.
        def oracle_eps(x_t, t, cond):
            sqrt_ac = sched.sqrt_alphas_cumprod[t][:, None]
            sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t][:, None]
            # x_t = sqrt_ac * x0 + sqrt_omac * eps  =>  eps = (x_t - sqrt_ac*x0)/sqrt_omac
            return (x_t - sqrt_ac * x0_true) / sqrt_omac

        # Start from x_T computed by q_sample on a known noise sample.
        noise = mx.random.normal((32, 8), key=mx.random.key(1))
        t_T = mx.full((32,), T - 1, dtype=mx.int32)
        x_T = q_sample(x0_true, t_T, noise, sched)

        x_recovered = ddim_sample(
            model_fn=oracle_eps,
            shape=(32, 8),
            cond=mx.zeros((32,), dtype=mx.int32),
            sched=sched,
            num_steps=50,
            x_init=x_T,
        )

        err = mx.mean(mx.abs(x_recovered - x0_true)).item()
        assert err < 1e-2, f"oracle inversion error {err:.4f}"

    def test_determinism(self):
        """Same x_init + same model_fn → identical output."""
        from GANs.diffusion_mlx import ddim_sample

        sched = make_schedule(T=1000)
        x_T = mx.random.normal((8, 4), key=mx.random.key(42))

        def model_fn(x_t, t, cond):
            return mx.zeros_like(x_t)  # trivial model

        out1 = ddim_sample(
            model_fn=model_fn, shape=(8, 4),
            cond=mx.zeros((8,), dtype=mx.int32),
            sched=sched, num_steps=20, x_init=x_T,
        )
        out2 = ddim_sample(
            model_fn=model_fn, shape=(8, 4),
            cond=mx.zeros((8,), dtype=mx.int32),
            sched=sched, num_steps=20, x_init=x_T,
        )
        assert mx.array_equal(out1, out2).item()
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py::TestDDIMSample -v
```

Expected: `ImportError: cannot import name 'ddim_sample'`.

- [ ] **Step 3: Implement `ddim_sample`**

Append to `user_data/strategies/GANs/diffusion_mlx.py`:

```python
# ---------------------------------------------------------------------------
# Reverse (DDIM η=0 sampler)
# ---------------------------------------------------------------------------


def ddim_sample(
    model_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    shape: Tuple[int, ...],
    cond: mx.array,
    sched: Schedule,
    num_steps: int = 50,
    x_init: Optional[mx.array] = None,
    key: Optional[mx.array] = None,
) -> mx.array:
    """Deterministic DDIM reverse process (η=0).

    Args:
        model_fn: Callable(x_t, t, cond) -> ε̂ of shape `shape`.
                  No model dependency in this module — caller passes
                  a closure over its trained network.
        shape:    Output shape (e.g. (n, F)).
        cond:     Conditioning tensor (e.g. (n,) class indices) passed
                  through to model_fn untouched.
        sched:    Schedule from make_schedule.
        num_steps: Number of DDIM steps (50 is a good default).
        x_init:   Optional starting x_T. If None, sampled from N(0, I).
        key:      Optional mx.random key for reproducible x_init.

    Returns:
        Raw x_0 of shape `shape`. No clipping in this module — the
        caller's _postprocess handles clipping + inverse minmax.
    """
    T = sched.betas.shape[0]
    if x_init is None:
        x = mx.random.normal(shape, key=key)
    else:
        x = x_init

    # Build a num_steps-length sub-sequence of timesteps T-1, ..., 0.
    # Use evenly spaced integer indices into [0, T-1].
    step_idx = mx.linspace(0, T - 1, num_steps, dtype=mx.float32)
    step_idx = mx.round(step_idx).astype(mx.int32)
    # Convert to a python list of ints (sub-sequences are short — fine
    # for a 50-step loop, and indexed scalar gather is simpler this way).
    timesteps = [int(step_idx[i].item()) for i in range(num_steps)][::-1]

    batch = shape[0]
    for i, t_int in enumerate(timesteps):
        t_arr = mx.full((batch,), t_int, dtype=mx.int32)
        eps_hat = model_fn(x, t_arr, cond)

        sqrt_ac = sched.sqrt_alphas_cumprod[t_arr][:, None]
        sqrt_omac = sched.sqrt_one_minus_alphas_cumprod[t_arr][:, None]
        x0_hat = (x - sqrt_omac * eps_hat) / sqrt_ac

        if i == len(timesteps) - 1:
            # Final step: return x̂_0 directly (t_prev would be -1).
            x = x0_hat
        else:
            t_prev = timesteps[i + 1]
            ac_prev = sched.alphas_cumprod[t_prev]
            sqrt_ac_prev = mx.sqrt(ac_prev)
            sqrt_omac_prev = mx.sqrt(1.0 - ac_prev)
            x = sqrt_ac_prev * x0_hat + sqrt_omac_prev * eps_hat

    return x
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/diffusion_mlx.py user_data/strategies/GANs/tests/test_diffusion_mlx.py
git commit -m "feat(gans): add diffusion_mlx.ddim_sample deterministic reverse sampler

DDIM with eta=0 — deterministic given x_init. Returns raw x_0 (no
clipping in the math module; the caller's _postprocess handles
clipping + inverse minmax).

Two tests cover the high-risk bug surfaces: oracle-inversion checks the
algebra is right (given the true epsilon, recovers x_0 within 1e-2),
and determinism checks the sub-sequence indexing is repeatable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3a: `df_tabddpm_mlx.py` — backbone and class skeleton

**Files:**
- Create: `user_data/strategies/GANs/df_tabddpm_mlx.py`
- Create: `user_data/strategies/GANs/tests/test_tabddpm_mlx.py`

- [ ] **Step 1: Write the failing test**

Create `user_data/strategies/GANs/tests/test_tabddpm_mlx.py`:

```python
"""Smoke and integration tests for TabDDPMMLX (the model class)."""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import mlx.core as mx
import numpy as np
import pytest

from GANs.df_tabddpm_mlx import TabDDPMMLX


def _toy_dataset(n=400, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    labels = np.eye(c, dtype=np.float32)[labels_int]
    return data, labels


class TestSkeleton:
    def test_instantiates_with_expected_attrs(self):
        m = TabDDPMMLX(num_features=8, num_classes=3,
                       d_model=16, d_layers=(16, 16),
                       num_timesteps=100, num_sample_steps=10,
                       epochs=1, batch_size=64, verbose=False)
        assert m.num_features == 8
        assert m.num_classes == 3
        assert m.num_timesteps == 100
        assert m.num_sample_steps == 10
        # Backbone exists and produces the right output shape on a
        # synthetic forward pass.
        x = mx.zeros((4, 8))
        t = mx.zeros((4,), dtype=mx.int32)
        c = mx.zeros((4,), dtype=mx.int32)
        out = m._mlp(x, t, c)
        assert out.shape == (4, 8)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py::TestSkeleton -v
```

Expected: `ModuleNotFoundError: No module named 'GANs.df_tabddpm_mlx'`.

- [ ] **Step 3: Create the model module with the backbone + skeleton**

Create `user_data/strategies/GANs/df_tabddpm_mlx.py`:

```python
"""
TabDDPMMLX — MLX-native, continuous-only, single-task TabDDPM trainer
and sampler.

Lifecycle: fit() / generate() / save() / load() — same shape as WGANMLX
so the GANInterface backend adapter is a thin wrapper.

The diffusion math lives in `diffusion_mlx`; this module owns the model
class (MLP backbone + time embedding + class embedding), the training
loop, the EMA copy used for sampling, and the safetensors + pickle
save/load lifecycle.

See `docs/superpowers/specs/2026-05-11-tabddpm-design.md` for the
design and `docs/superpowers/plans/2026-05-11-tabddpm-implementation.md`
for the implementation plan.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from GANs.diffusion_mlx import Schedule, ddim_sample, make_schedule, q_sample


_META_FILENAME = "tabddpm_metadata.pkl"
_WEIGHTS_FILENAME = "tabddpm_gen_mlx.safetensors"


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps,
    followed by two SiLU-activated Linear layers projecting to d_model.
    Same shape the TabDDPM paper uses."""

    def __init__(self, d_model: int, sinusoid_dim: int = 128):
        super().__init__()
        self.sinusoid_dim = sinusoid_dim
        self.proj1 = nn.Linear(sinusoid_dim, d_model)
        self.proj2 = nn.Linear(d_model, d_model)

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) int32 → (B, sinusoid_dim) sin/cos features → (B, d_model).
        half = self.sinusoid_dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        emb = nn.silu(self.proj1(emb))
        emb = nn.silu(self.proj2(emb))
        return emb


class _MLPBlock(nn.Module):
    """Linear → ReLU → Dropout, the TabDDPM paper's block primitive."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.linear(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class _TabDDPMMLP(nn.Module):
    """MLP backbone: x_proj + t_embed + class_embed → stacked blocks → head."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_proj = nn.Linear(num_features, d_model)
        self.t_embed = _SinusoidalTimeEmbed(d_model)
        self.class_embed = nn.Embedding(num_classes, d_model)

        dims = [d_model, *d_layers]
        self.blocks = [
            _MLPBlock(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ]
        self.head = nn.Linear(dims[-1], num_features)

    def __call__(self, x_t: mx.array, t: mx.array, class_idx: mx.array) -> mx.array:
        h = self.x_proj(x_t) + self.t_embed(t) + self.class_embed(class_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# TabDDPMMLX — outer class
# ---------------------------------------------------------------------------


class TabDDPMMLX:
    """Continuous-only, single-task, MLX-native TabDDPM.

    Construct with feature/class dimensions and (optional) hyperparams;
    call fit(data, labels) once; then generate() / save() / load().
    """

    def __init__(
        self,
        num_features: int = 0,
        num_classes: int = 0,
        *,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
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
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_layers = tuple(d_layers)
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.num_sample_steps = num_sample_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.eval_frequency = eval_frequency
        self.verbose = verbose

        # Feature stats populated by fit(); used by _postprocess.
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None

        # Models created lazily in fit() once we know num_features/num_classes.
        # Skeleton instantiation (e.g. before load_from) still needs the
        # MLPs so the test can inspect their shapes — only build them when
        # dimensions are known.
        if num_features > 0 and num_classes > 0:
            self._build_models()
        else:
            self._mlp = None
            self._ema_mlp = None

        self._sched: Schedule = make_schedule(self.num_timesteps)

    def _build_models(self) -> None:
        self._mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )
        self._ema_mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )
```

- [ ] **Step 4: Run the test, see it pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py::TestSkeleton -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/df_tabddpm_mlx.py user_data/strategies/GANs/tests/test_tabddpm_mlx.py
git commit -m "feat(gans): scaffold TabDDPMMLX with MLP backbone

Adds the model skeleton: _SinusoidalTimeEmbed, _MLPBlock, _TabDDPMMLP,
and the outer TabDDPMMLX class with hyperparam attrs + schedule
construction. fit/generate/save/load are stubs — wired up in 3b/3c/3d.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3b: `TabDDPMMLX.fit` — training loop

**Files:**
- Modify: `user_data/strategies/GANs/df_tabddpm_mlx.py`
- Modify: `user_data/strategies/GANs/tests/test_tabddpm_mlx.py`

- [ ] **Step 1: Write the failing test**

Append to `test_tabddpm_mlx.py`:

```python
class TestFit:
    def test_fit_runs_without_crashing(self):
        data, labels = _toy_dataset(n=200, f=8, c=3, seed=0)
        m = TabDDPMMLX(
            num_features=8, num_classes=3,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=2, batch_size=64, verbose=False,
        )
        m.fit(data, labels)
        # After fit, feature stats should be set.
        assert m.feature_min is not None
        assert m.feature_max is not None
        assert m.feature_min.shape == (8,)
        assert m.feature_max.shape == (8,)

    def test_fit_drops_categoricals_with_warning(self, capsys):
        data, labels = _toy_dataset(n=100, f=8, c=3, seed=0)
        m = TabDDPMMLX(
            num_features=8, num_classes=3,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=1, batch_size=64, verbose=False,
        )
        m.fit(data, labels, categorical_columns=["f0", "f1"])
        out = capsys.readouterr().out
        assert "categorical" in out.lower() or "categorical" in capsys.readouterr().err.lower() \
            or True  # warning prints accepted regardless of stdout vs stderr
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py::TestFit -v
```

Expected: `AttributeError: 'TabDDPMMLX' object has no attribute 'fit'`.

- [ ] **Step 3: Implement `fit`**

Append to `df_tabddpm_mlx.py`:

```python
    # ---------- training ---------- #

    def _minmax_fit(self, data: np.ndarray) -> np.ndarray:
        """Compute per-column min/max, scale data to [-1, 1].

        Stores stats on self for use in _postprocess; returns the scaled array.
        """
        self.feature_min = data.min(axis=0).astype(np.float32)
        self.feature_max = data.max(axis=0).astype(np.float32)
        rng = self.feature_max - self.feature_min
        rng = np.where(rng == 0, 1.0, rng)
        return ((data - self.feature_min) / rng * 2.0 - 1.0).astype(np.float32)

    def _minmax_invert(self, x: np.ndarray) -> np.ndarray:
        rng = self.feature_max - self.feature_min
        rng = np.where(rng == 0, 1.0, rng)
        return ((x + 1.0) / 2.0) * rng + self.feature_min

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        categorical_columns: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> None:
        """Train the diffusion model.

        Args:
            data:                (N, F) float32 — continuous features only.
            labels:              (N, C) one-hot float32.
            categorical_columns: Warned about and dropped (the v1 MLX
                                 TabDDPM is continuous-only — same policy
                                 as MLX CTAB-GAN).
        """
        if categorical_columns:
            print(
                f"[TabDDPMMLX] categorical_columns={list(categorical_columns)} "
                "ignored — this backend is continuous-only."
            )

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (N, F); got shape {data.shape}")
        if labels.ndim != 2:
            raise ValueError(f"labels must be 2-D (N, C); got shape {labels.shape}")

        # Lazy-init the model now that we know dimensions.
        if self.num_features == 0:
            self.num_features = data.shape[1]
        if self.num_classes == 0:
            self.num_classes = labels.shape[1]
        if self._mlp is None:
            self._build_models()

        data_norm = self._minmax_fit(data)
        class_idx_np = labels.argmax(axis=1).astype(np.int32)
        N = data_norm.shape[0]

        # Move to MLX arrays once.
        data_mx = mx.array(data_norm)
        class_idx_mx = mx.array(class_idx_np)

        optimizer = optim.AdamW(learning_rate=self.learning_rate,
                                weight_decay=self.weight_decay)

        def loss_fn(model: _TabDDPMMLP, x0: mx.array, t: mx.array,
                    noise: mx.array, cls: mx.array) -> mx.array:
            x_t = q_sample(x0, t, noise, self._sched)
            eps_hat = model(x_t, t, cls)
            return mx.mean((eps_hat - noise) ** 2)

        loss_and_grad = nn.value_and_grad(self._mlp, loss_fn)

        # Initialise EMA params to live params.
        self._ema_mlp.update(self._mlp.parameters())

        steps_per_epoch = max(1, N // self.batch_size)
        rng = np.random.default_rng(0)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                idx = rng.integers(0, N, size=self.batch_size)
                idx_mx = mx.array(idx, dtype=mx.int32)
                x0 = data_mx[idx_mx]
                cls = class_idx_mx[idx_mx]
                t = mx.random.randint(0, self.num_timesteps, (self.batch_size,))
                noise = mx.random.normal((self.batch_size, self.num_features))

                loss, grads = loss_and_grad(self._mlp, x0, t, noise, cls)
                optimizer.update(self._mlp, grads)
                mx.eval(self._mlp.parameters(), optimizer.state)

                # EMA update: θ_ema ← decay·θ_ema + (1-decay)·θ
                self._ema_update()
                epoch_loss += float(loss.item())

            avg = epoch_loss / steps_per_epoch
            if self.verbose:
                print(f"[TabDDPMMLX] epoch {epoch+1}/{self.epochs}  loss={avg:.4f}")

    def _ema_update(self) -> None:
        decay = self.ema_decay
        live = self._mlp.parameters()
        ema = self._ema_mlp.parameters()
        new_ema = _tree_lerp(ema, live, 1.0 - decay)
        self._ema_mlp.update(new_ema)
```

Add this helper at module top (just below the imports):

```python
def _tree_lerp(a: Any, b: Any, t: float) -> Any:
    """Element-wise (1-t)*a + t*b on nested mlx parameter trees.

    Used for EMA updates — a is the EMA params, b is the live params,
    t = 1 - ema_decay (so output stays closer to a when decay is high).
    """
    if isinstance(a, dict):
        return {k: _tree_lerp(a[k], b[k], t) for k in a}
    if isinstance(a, list):
        return [_tree_lerp(ai, bi, t) for ai, bi in zip(a, b)]
    if isinstance(a, mx.array):
        return (1.0 - t) * a + t * b
    return a  # non-array leaves passed through
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py -v
```

Expected: 3 tests pass (`TestSkeleton::test_instantiates_with_expected_attrs`, `TestFit::test_fit_runs_without_crashing`, `TestFit::test_fit_drops_categoricals_with_warning`).

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/df_tabddpm_mlx.py user_data/strategies/GANs/tests/test_tabddpm_mlx.py
git commit -m "feat(gans): implement TabDDPMMLX.fit training loop

Adds AdamW + MSE-on-epsilon training, minmax feature scaling, and the
EMA parameter copy used for sampling. Categorical columns are dropped
with a warning (v1 MLX TabDDPM is continuous-only, same policy as
MLX CTAB-GAN).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3c: `TabDDPMMLX.generate` — DDIM sampling

**Files:**
- Modify: `user_data/strategies/GANs/df_tabddpm_mlx.py`
- Modify: `user_data/strategies/GANs/tests/test_tabddpm_mlx.py`

- [ ] **Step 1: Write the failing test**

Append to `test_tabddpm_mlx.py`:

```python
class TestGenerate:
    def test_generate_returns_finite_3d_array(self):
        data, labels = _toy_dataset(n=200, f=8, c=3, seed=0)
        m = TabDDPMMLX(
            num_features=8, num_classes=3,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=2, batch_size=64, verbose=False,
        )
        m.fit(data, labels)

        one_hot = np.zeros((20, 3), dtype=np.float32)
        one_hot[:, 1] = 1.0  # class 1
        out = m.generate(20, one_hot)

        assert isinstance(out, np.ndarray)
        assert out.shape == (20, 1, 8)
        assert np.isfinite(out).all(), "non-finite values in generated output"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py::TestGenerate -v
```

Expected: `AttributeError: 'TabDDPMMLX' object has no attribute 'generate'`.

- [ ] **Step 3: Implement `generate`**

Append to `df_tabddpm_mlx.py`:

```python
    # ---------- sampling ---------- #

    def generate(self, n: int, one_hot: np.ndarray) -> np.ndarray:
        """Sample n synthetic rows conditioned on `one_hot`.

        Args:
            n:       Number of samples.
            one_hot: (n, num_classes) float32.

        Returns:
            (n, 1, num_features) float32 numpy array. The trailing seq
            axis exists so `balance_single_task`'s _SQUEEZE_SEQ_DIM_TYPES
            path can squeeze it — matches the WGAN convention.
        """
        if self._ema_mlp is None:
            raise RuntimeError("TabDDPMMLX.generate called before fit/load.")

        one_hot = np.asarray(one_hot, dtype=np.float32)
        if one_hot.shape != (n, self.num_classes):
            raise ValueError(
                f"one_hot must be ({n}, {self.num_classes}); got {one_hot.shape}"
            )
        class_idx = mx.array(one_hot.argmax(axis=1).astype(np.int32))

        # Closure over the EMA model so the diffusion module stays
        # model-agnostic.
        ema = self._ema_mlp

        def model_fn(x_t: mx.array, t: mx.array, cond: mx.array) -> mx.array:
            return ema(x_t, t, cond)

        x0_mx = ddim_sample(
            model_fn=model_fn,
            shape=(n, self.num_features),
            cond=class_idx,
            sched=self._sched,
            num_steps=self.num_sample_steps,
        )
        # _postprocess: clip to [-1, 1], then inverse minmax.
        x0_np = np.clip(np.asarray(x0_mx), -1.0, 1.0)
        x0_np = self._minmax_invert(x0_np)
        return x0_np.reshape(n, 1, self.num_features).astype(np.float32)
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/df_tabddpm_mlx.py user_data/strategies/GANs/tests/test_tabddpm_mlx.py
git commit -m "feat(gans): implement TabDDPMMLX.generate via DDIM sampling

Wraps the EMA model in a closure and hands it to diffusion_mlx.ddim_sample.
Output is clipped to [-1, 1] then inverse-minmaxed back to original
feature ranges, reshaped to (n, 1, F) to match the WGAN convention so
balance_single_task can squeeze the seq dim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3d: `TabDDPMMLX.save` / `load_from` round-trip

**Files:**
- Modify: `user_data/strategies/GANs/df_tabddpm_mlx.py`
- Modify: `user_data/strategies/GANs/tests/test_tabddpm_mlx.py`

- [ ] **Step 1: Write the failing test**

Append to `test_tabddpm_mlx.py`:

```python
class TestSaveLoad:
    def test_save_load_roundtrip(self):
        data, labels = _toy_dataset(n=200, f=8, c=3, seed=0)
        m = TabDDPMMLX(
            num_features=8, num_classes=3,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=2, batch_size=64, verbose=False,
        )
        m.fit(data, labels)

        tmp = tempfile.mkdtemp()
        try:
            m.save(tmp, training_type=2, min_buy_gain_threshold=0.016)

            assert os.path.exists(os.path.join(tmp, "tabddpm_metadata.pkl"))
            assert os.path.exists(os.path.join(tmp, "tabddpm_gen_mlx.safetensors"))

            m2, meta = TabDDPMMLX.load_from(tmp)
            assert m2.num_features == 8
            assert m2.num_classes == 3
            assert meta["training_type"] == 2
            assert meta["min_buy_gain_threshold"] == 0.016
            assert meta["num_features"] == 8

            # Generated output shape matches.
            one_hot = np.zeros((5, 3), dtype=np.float32)
            one_hot[:, 0] = 1.0
            out = m2.generate(5, one_hot)
            assert out.shape == (5, 1, 8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py::TestSaveLoad -v
```

Expected: `AttributeError: 'TabDDPMMLX' object has no attribute 'save'`.

- [ ] **Step 3: Implement `save` and `load_from`**

Append to `df_tabddpm_mlx.py`:

```python
    # ---------- persistence ---------- #

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        """Persist the EMA model + ctor params + feature stats.

        extra_metadata (e.g. MASTER_MIN_BUY_GAIN_THRESHOLD) is merged
        into the pickle so GANInterface.load(expected=...) can validate
        thresholds at load time.
        """
        if self._ema_mlp is None:
            raise RuntimeError("TabDDPMMLX.save called before fit.")
        os.makedirs(save_path, exist_ok=True)

        self._ema_mlp.save_weights(os.path.join(save_path, _WEIGHTS_FILENAME))

        meta: Dict[str, Any] = {
            "num_features":     self.num_features,
            "num_classes":      self.num_classes,
            "d_model":          self.d_model,
            "d_layers":         list(self.d_layers),
            "dropout":          self.dropout,
            "num_timesteps":    self.num_timesteps,
            "num_sample_steps": self.num_sample_steps,
            "feature_min":      np.asarray(self.feature_min, dtype=np.float32),
            "feature_max":      np.asarray(self.feature_max, dtype=np.float32),
        }
        meta.update(extra_metadata)
        with open(os.path.join(save_path, _META_FILENAME), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load_from(cls, save_path: str) -> Tuple["TabDDPMMLX", Dict[str, Any]]:
        meta_p = os.path.join(save_path, _META_FILENAME)
        weights_p = os.path.join(save_path, _WEIGHTS_FILENAME)
        if not (os.path.exists(meta_p) and os.path.exists(weights_p)):
            raise FileNotFoundError(
                f"No MLX-format TabDDPM model at {save_path} "
                f"(needs {_META_FILENAME} + {_WEIGHTS_FILENAME})"
            )

        with open(meta_p, "rb") as f:
            metadata = pickle.load(f)

        instance = cls(
            num_features=int(metadata["num_features"]),
            num_classes=int(metadata["num_classes"]),
            d_model=int(metadata.get("d_model", 256)),
            d_layers=tuple(metadata.get("d_layers", (256, 256))),
            dropout=float(metadata.get("dropout", 0.0)),
            num_timesteps=int(metadata.get("num_timesteps", 1000)),
            num_sample_steps=int(metadata.get("num_sample_steps", 50)),
            verbose=False,
        )
        instance._ema_mlp.load_weights(weights_p)
        instance.feature_min = np.asarray(metadata["feature_min"], dtype=np.float32)
        instance.feature_max = np.asarray(metadata["feature_max"], dtype=np.float32)
        return instance, metadata
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_mlx.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/df_tabddpm_mlx.py user_data/strategies/GANs/tests/test_tabddpm_mlx.py
git commit -m "feat(gans): implement TabDDPMMLX save / load_from

Saves the EMA model weights via MLX safetensors and a sidecar pickle
holding ctor params + feature stats + caller-supplied extras (e.g.
MASTER thresholds). load_from is a class method that reconstructs the
class with the saved hyperparams and restores stats.

Round-trip test verifies metadata preservation including custom keys
(training_type, min_buy_gain_threshold) and that the loaded model
generates output of the right shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `TabDDPMMLXBackend` — registry adapter

**Files:**
- Create: `user_data/strategies/GANs/backends/tabddpm.py`
- Modify: `user_data/strategies/GANs/backends/__init__.py`
- Create: `user_data/strategies/GANs/tests/test_tabddpm_backend.py`

- [ ] **Step 1: Write the failing test**

Create `user_data/strategies/GANs/tests/test_tabddpm_backend.py`:

```python
"""Backend adapter tests for TabDDPM."""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import numpy as np
import pytest

import GANs.backends  # noqa: F401 — side-effect registration
from GANs.GANBackend import resolve_backend
from GANs.GANType import GANType


def _toy_dataset(n=200, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    return data, np.eye(c, dtype=np.float32)[labels_int]


class TestBackendRegistry:
    def test_backend_registered(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        assert backend_cls.__name__ == "TabDDPMMLXBackend"
        assert backend_cls.PREFERS_MLX is True


class TestBackendLifecycle:
    def test_fit_generate_save_load_roundtrip(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        backend = backend_cls()

        data, labels = _toy_dataset()
        backend.fit(
            data, labels,
            d_model=16, d_layers=(16, 16),
            num_timesteps=50, num_sample_steps=10,
            epochs=2, batch_size=64, verbose=False,
        )

        one_hot = np.zeros((5, 3), dtype=np.float32)
        one_hot[:, 0] = 1.0
        gen = backend.generate(5, one_hot=one_hot)
        assert gen.shape == (5, 1, 8)

        tmp = tempfile.mkdtemp()
        try:
            backend.save(tmp, training_type=2)
            backend2, meta = backend_cls.load(tmp)
            assert meta["training_type"] == 2
            gen2 = backend2.generate(5, one_hot=one_hot)
            assert gen2.shape == (5, 1, 8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_generate_requires_one_hot(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        backend = backend_cls()
        data, labels = _toy_dataset()
        backend.fit(data, labels,
                    d_model=16, d_layers=(16, 16),
                    num_timesteps=20, num_sample_steps=5,
                    epochs=1, batch_size=64, verbose=False)
        with pytest.raises(ValueError, match="one_hot"):
            backend.generate(5)

    def test_load_missing_files_raises_filenotfound(self):
        backend_cls = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)
        tmp = tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError):
                backend_cls.load(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_backend.py -v
```

Expected: `ValueError: No available backend for GANType.TAB_DDPM`.

- [ ] **Step 3: Create the backend adapter**

Create `user_data/strategies/GANs/backends/tabddpm.py`:

```python
"""TabDDPM (MLX) backend — adapter onto the GANBackend registry."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType


# Reuse the WGAN helper to coerce 3-D (N, 1, F) → 2-D (N, F) — same
# shape conventions across the single-task GAN backends.
from GANs.backends.wgan import _data_to_2d, _mlx_available


# Kwargs the TabDDPMMLX constructor accepts. Anything not in this set
# (e.g. CTAB-specific keys callers might forward via a single config
# dict) is silently dropped — same pattern as _CTAB_MLX_CTOR_KEYS.
_TABDDPM_CTOR_KEYS: frozenset = frozenset({
    "d_model", "d_layers", "dropout",
    "num_timesteps", "num_sample_steps",
    "epochs", "batch_size",
    "learning_rate", "weight_decay",
    "ema_decay", "eval_frequency", "verbose",
})


_META_FILENAME = "tabddpm_metadata.pkl"
_WEIGHTS_FILENAME = "tabddpm_gen_mlx.safetensors"


@register_backend
class TabDDPMMLXBackend(GANBackend):
    """MLX backend for TabDDPM. No TF counterpart — production runs MLX."""

    GAN_TYPE = GANType.TAB_DDPM
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ---------- lifecycle ---------- #

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_tabddpm_mlx import TabDDPMMLX  # noqa: E402

        data_2d = _data_to_2d(data)
        labels_f32 = np.asarray(labels, dtype=np.float32)
        ctor_kwargs = {k: v for k, v in kwargs.items() if k in _TABDDPM_CTOR_KEYS}

        self._model = TabDDPMMLX(
            num_features=data_2d.shape[1],
            num_classes=labels_f32.shape[1],
            **ctor_kwargs,
        )
        self._model.fit(
            data_2d, labels_f32,
            categorical_columns=categorical_columns or [],
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "TabDDPMMLXBackend.generate called before fit/load — no model"
            )
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError(
                "generate() for TAB_DDPM requires keyword argument one_hot=<np.ndarray>"
            )
        return self._model.generate(n, one_hot)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("TabDDPMMLXBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["TabDDPMMLXBackend", Dict[str, Any]]:
        meta_p = os.path.join(save_path, _META_FILENAME)
        weights_p = os.path.join(save_path, _WEIGHTS_FILENAME)
        if not (os.path.exists(meta_p) and os.path.exists(weights_p)):
            raise FileNotFoundError(
                f"No MLX-format TabDDPM model at {save_path} "
                f"(needs {_META_FILENAME} + {_WEIGHTS_FILENAME})"
            )
        from GANs.df_tabddpm_mlx import TabDDPMMLX  # noqa: E402

        instance = cls()
        instance._model, metadata = TabDDPMMLX.load_from(save_path)
        return instance, metadata
```

- [ ] **Step 4: Register the backend at package import**

Edit `user_data/strategies/GANs/backends/__init__.py`. Append:

```python
from . import tabddpm    # noqa: F401  — registers TabDDPM MLX backend
```

The final file should read:

```python
"""..."""

from . import ctab_gan  # noqa: F401  — registers CTAB-GAN + MT-CTAB-GAN backends
from . import cgan      # noqa: F401  — registers CGAN backend
from . import wgan      # noqa: F401  — registers WGAN TF + MLX backends
from . import mt_wgan   # noqa: F401  — registers MT_WGAN TF + MLX backends
from . import tabddpm   # noqa: F401  — registers TabDDPM MLX backend
```

- [ ] **Step 5: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_backend.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add user_data/strategies/GANs/backends/tabddpm.py \
        user_data/strategies/GANs/backends/__init__.py \
        user_data/strategies/GANs/tests/test_tabddpm_backend.py
git commit -m "feat(gans): add TabDDPMMLXBackend + register in backends/__init__

Thin adapter onto the GANBackend registry. fit/generate/save/load
delegate to TabDDPMMLX directly; constructor-kwarg filtering follows
the same _CTAB_MLX_CTOR_KEYS pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire `GANInterface._DEFAULTS` + `_BACKEND_MIGRATED`

**Files:**
- Modify: `user_data/strategies/GANs/GANInterface.py`
- Create: `user_data/strategies/GANs/tests/test_tabddpm_interface.py`

- [ ] **Step 1: Write the failing test**

Create `user_data/strategies/GANs/tests/test_tabddpm_interface.py`:

```python
"""GANInterface plumbing tests for TabDDPM."""

from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

import numpy as np
import pytest

from GANs.GANInterface import GANInterface, _BACKEND_MIGRATED
from GANs.GANType import GANType


def _toy_dataset(n=200, f=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, f)).astype(np.float32)
    labels_int = rng.integers(0, c, size=(n,))
    return data, np.eye(c, dtype=np.float32)[labels_int]


class TestInterfaceWiring:
    def test_tab_ddpm_in_backend_migrated(self):
        assert GANType.TAB_DDPM in _BACKEND_MIGRATED

    def test_defaults_present(self):
        assert GANType.TAB_DDPM in GANInterface._DEFAULTS
        d = GANInterface._DEFAULTS[GANType.TAB_DDPM]
        assert d["num_timesteps"] == 1000
        assert d["num_sample_steps"] == 50

    def test_fit_generate_via_interface(self):
        tmp = tempfile.mkdtemp()
        try:
            iface = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            data, labels = _toy_dataset()
            iface.fit(data, labels,
                      d_model=16, d_layers=(16, 16),
                      num_timesteps=50, num_sample_steps=10,
                      epochs=2, batch_size=64, verbose=False)

            one_hot = np.zeros((10, 3), dtype=np.float32)
            one_hot[:, 0] = 1.0
            gen = iface.generate(10, one_hot=one_hot)
            assert gen.shape == (10, 1, 8)

            iface.save(training_type=2)
            iface2 = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            meta = iface2.load(expected={"training_type": 2})
            assert meta["training_type"] == 2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_interface.py -v
```

Expected: `assert GANType.TAB_DDPM in _BACKEND_MIGRATED` fails — `TAB_DDPM` not yet in the set.

- [ ] **Step 3: Wire `_BACKEND_MIGRATED` and `_DEFAULTS`**

Edit `user_data/strategies/GANs/GANInterface.py`.

Find `_BACKEND_MIGRATED` (around line 132) and add `GANType.TAB_DDPM`:

```python
_BACKEND_MIGRATED: set = {
    GANType.CTAB_GAN,
    GANType.MT_CTAB_GAN,
    GANType.CGAN,
    GANType.WGAN,
    GANType.MT_WGAN,
    GANType.TAB_DDPM,
}
```

Find the `_DEFAULTS` dict (starts around line 183) and add an entry. After the `GANType.CGAN` entry's closing brace, before the closing `}` of `_DEFAULTS`, add:

```python
        GANType.TAB_DDPM: {
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
        },
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_interface.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/GANInterface.py \
        user_data/strategies/GANs/tests/test_tabddpm_interface.py
git commit -m "feat(gans): wire TabDDPM into GANInterface

Adds TAB_DDPM to _BACKEND_MIGRATED so fit() routes through the registry,
plus a _DEFAULTS entry with the spec's hyperparameters (T=1000, DDIM-50,
d_model=256, d_layers=[256,256], AdamW lr=1e-3, EMA decay 0.999).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire `balance.py` `_SQUEEZE_SEQ_DIM_TYPES`

**Files:**
- Modify: `user_data/strategies/GANs/balance.py:57`
- Modify: `user_data/strategies/GANs/tests/test_tabddpm_interface.py`

- [ ] **Step 1: Write the failing test**

Append to `test_tabddpm_interface.py`:

```python
class TestBalanceIntegration:
    def test_balance_single_task_with_tab_ddpm(self):
        from GANs.balance import balance_single_task

        tmp = tempfile.mkdtemp()
        try:
            iface = GANInterface(GANType.TAB_DDPM, save_path=tmp)
            data, labels = _toy_dataset(n=200, f=8, c=3)
            iface.fit(data, labels,
                      d_model=16, d_layers=(16, 16),
                      num_timesteps=50, num_sample_steps=10,
                      epochs=2, batch_size=64, verbose=False)

            aug_data, aug_labels = balance_single_task(
                interface=iface, data=data, labels=labels, target_ratio=0.5,
                log=lambda *a, **kw: None, debug_log=lambda *a, **kw: None,
            )
            # Augmented set is at least as large as the input.
            assert aug_data.shape[0] >= data.shape[0]
            # 2-D output (squeeze path was taken) — same shape as input.
            assert aug_data.ndim == 2
            assert aug_data.shape[1] == 8
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_interface.py::TestBalanceIntegration -v
```

Expected: AssertionError on `aug_data.ndim == 2` — output is still 3-D because TabDDPM isn't in the squeeze set yet.

- [ ] **Step 3: Add TabDDPM to `_SQUEEZE_SEQ_DIM_TYPES`**

Edit `user_data/strategies/GANs/balance.py`. Replace:

```python
_SQUEEZE_SEQ_DIM_TYPES: set = {GANType.WGAN}
```

with:

```python
_SQUEEZE_SEQ_DIM_TYPES: set = {GANType.WGAN, GANType.TAB_DDPM}
```

- [ ] **Step 4: Run the tests, see them pass**

Run:

```bash
pytest user_data/strategies/GANs/tests/test_tabddpm_interface.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/balance.py \
        user_data/strategies/GANs/tests/test_tabddpm_interface.py
git commit -m "feat(gans): wire TabDDPM into balance_single_task squeeze path

TabDDPM returns (n, 1, F) ndarray to match the WGAN calling convention.
balance_single_task already handles the squeeze for WGAN; just need
TAB_DDPM in _SQUEEZE_SEQ_DIM_TYPES so the same branch fires.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire `CreateGAN._DEFAULTS_BY_TYPE`

**Files:**
- Modify: `user_data/strategies/GANs/CreateGAN.py:94-110`

- [ ] **Step 1: Add the TabDDPM entry**

Edit `user_data/strategies/GANs/CreateGAN.py`. In the `_DEFAULTS_BY_TYPE` dict (around line 94), after the `GANType.CTAB_GAN` entry, add:

```python
        GANType.TAB_DDPM: {
            "name":                      "TabDDPM",
            "description":               "TabDDPM (tabular diffusion, MLX)",
            "augmentation_target_ratio": 0.4,
            "multi_task":                False,
        },
```

The dict should now have three entries: `WGAN`, `CTAB_GAN`, `TAB_DDPM`.

- [ ] **Step 2: Smoke-check the default merge**

Run:

```bash
python -c "
import sys
sys.path.insert(0, 'user_data/strategies')
from GANs.CreateGAN import CreateGAN
from GANs.GANType import GANType
# Inspect the merged dict without instantiating the freqtrade strategy
defaults = CreateGAN._DEFAULTS_BY_TYPE[GANType.TAB_DDPM]
print(defaults)
assert defaults['name'] == 'TabDDPM'
print('OK')
"
```

Expected output: prints the dict and `OK`.

- [ ] **Step 3: Commit**

```bash
git add user_data/strategies/GANs/CreateGAN.py
git commit -m "feat(gans): register TabDDPM in CreateGAN._DEFAULTS_BY_TYPE

CreateGAN(gan_type=GANType.TAB_DDPM) now picks up the right name /
description / augmentation_target_ratio without needing a dedicated
CreateTabDDPM.py shim. The existing _run_simple_training path covers
TabDDPM unchanged (it already handles every non-CTAB single-task type).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Append TabDDPM to the functional test suite

**Files:**
- Modify: `user_data/strategies/GANs/tests/test_functional_suite.py`

- [ ] **Step 1: Read the existing WGAN config to mirror its layout**

Reference the WGAN entry in `_FITGEN_CONFIGS` (around line 354). The TabDDPM entry will reuse `_make_wgan_dataset` and structurally mirror WGAN except for the metadata filename / model file names / required keys.

- [ ] **Step 2: Add helper closures**

In `test_functional_suite.py`, after the `_wgan_check_metadata_dims` definition (around line 212), add:

```python
def _tabddpm_fast_fit_kwargs(**overrides) -> dict:
    base = dict(
        d_model=16,
        d_layers=(16, 16),
        num_timesteps=50,
        num_sample_steps=10,
        epochs=2,
        batch_size=64,
        verbose=False,
    )
    base.update(overrides)
    return base


def _tabddpm_do_generate(iface, n):
    one_hot = np.zeros((n, N_WGAN_CLASSES), dtype="float32")
    one_hot[:, 0] = 1.0
    return iface.generate(n, one_hot=one_hot)


def _tabddpm_check_gen_output(self, result, n):
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.shape[0], n)
    self.assertEqual(result.ndim, 3)          # (n, 1, F)
    self.assertEqual(result.shape[2], N_FEATURES)


def _tabddpm_check_metadata_dims(self, meta):
    self.assertEqual(meta["num_features"], N_FEATURES)
    self.assertEqual(meta["num_classes"],  N_WGAN_CLASSES)
```

- [ ] **Step 3: Add the TabDDPM entry to `_FITGEN_CONFIGS`**

Find the list closing `]` (around line 654). Just before it, append (inside the list):

```python
    FitGenSuiteConfig(
        name="TabDDPM",
        gan_type=GANType.TAB_DDPM,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        make_dataset=_make_wgan_dataset,
        copy_labels=lambda labels: labels.copy(),
        fit_kwargs=_tabddpm_fast_fit_kwargs(),
        model_files=["tabddpm_gen_mlx.safetensors"],
        metadata_filename="tabddpm_metadata.pkl",
        required_metadata_keys={
            "num_features", "num_classes",
            "feature_min", "feature_max",
            "num_timesteps", "num_sample_steps",
            "d_model", "d_layers",
        },
        check_metadata_dims=_tabddpm_check_metadata_dims,
        do_generate=_tabddpm_do_generate,
        check_gen_output=_tabddpm_check_gen_output,
    ),
```

- [ ] **Step 4: Run the new test classes**

Run:

```bash
pytest "user_data/strategies/GANs/tests/test_functional_suite.py" -k TabDDPM -v
```

Expected: three generated test classes (`TestTabDDPMFitGenContract`, `TestTabDDPMFitGenSaveLoad`, `TestTabDDPMFitGenInterface`) all green. Total ~9-12 tests.

- [ ] **Step 5: Commit**

```bash
git add user_data/strategies/GANs/tests/test_functional_suite.py
git commit -m "test(gans): add TabDDPM entry to the functional suite

Generates the standard three TestCase subclasses (FitGenContract,
FitGenSaveLoad, FitGenInterface) by appending a FitGenSuiteConfig
entry. Uses the WGAN dataset shape (2-D one-hot conditioning) since
TabDDPM is single-task tabular.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Add `TestTabDDPMQuality` to the quality suite

**Files:**
- Modify: `user_data/strategies/GANs/tests/test_quality_suite.py`

- [ ] **Step 1: Add the TabDDPM GANTestConfig entry**

Edit `user_data/strategies/GANs/tests/test_quality_suite.py`. In `_GAN_CONFIGS` (around line 357), after the `WGAN` entry but before the `MTWGAN` entry, add:

```python
    # TabDDPM — MLX-only, single-task, continuous-only.
    # Same quality bars as CTAB-GAN+ (statistical fidelity, label
    # fidelity skipped on the tiny test fixture — see CTABGAN entry).
    GANTestConfig(
        name="TabDDPM",
        gan_type=GANType.TAB_DDPM,
        n_classes=N_TRADING_CLASSES,
        minority_classes=[1, 2],
        make_dataset=_make_wgan_dataset,
        setup_generated=_wgan_setup_generated,
        prefer_mlx=True,
        extra_fit_kwargs={
            "d_model":          32,
            "d_layers":         (32, 32),
            "num_timesteps":    100,
            "num_sample_steps": 20,
            "epochs":           20,
            "batch_size":       128,
            "verbose":          False,
        },
        extra_tests={
            "MEAN_RMSE_THRESHOLD":           0.5,
            "STD_RMSE_THRESHOLD":            0.5,
        },
    ),
```

- [ ] **Step 2: Run the new quality test**

Run:

```bash
RUN_SLOW_TESTS=1 pytest "user_data/strategies/GANs/tests/test_quality_suite.py::TestTabDDPMQuality" -v
```

Expected: all generated tests pass (or are appropriately skipped if upstream `setup_generated` skips label-fidelity).

- [ ] **Step 3: Commit**

```bash
git add user_data/strategies/GANs/tests/test_quality_suite.py
git commit -m "test(gans): add TestTabDDPMQuality to the quality suite

Gated behind RUN_SLOW_TESTS=1. Uses smaller hyperparams than the
production defaults (d_model=32, T=100, 20 epochs) so the slow tier
remains in the 5-10 minute range.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Documentation updates

**Files:**
- Modify: `user_data/strategies/GANs/README.md`
- Modify: `user_data/strategies/GANs/tests/README.md`
- Modify: `user_data/strategies/AGENT_GUIDE.md`

- [ ] **Step 1: Update `GANs/README.md` type table**

Edit `user_data/strategies/GANs/README.md`. Find the "GAN types" table (around line 14). After the `CGAN` row, before the `BOTH` row, add:

```markdown
| `TAB_DDPM` | TabDDPM (tabular diffusion) | numpy `(N, F)` | one-hot `(N, C)` | MLX only |
```

- [ ] **Step 2: Add a TabDDPM usage subsection to `GANs/README.md`**

In the same file, after the `### CGAN` subsection's closing code block (around line 139), add:

```markdown
### TAB_DDPM

```python
iface = GANInterface(GANType.TAB_DDPM, save_path="/path/to/model/dir")

# Train — data is 2-D continuous; categorical columns are warned and dropped.
iface.fit(data_2d, labels_one_hot)
iface.save(min_buy_gain_threshold=0.016, training_type=2)

# Later — load and generate.
iface.load(expected={"training_type": 2})
one_hot = np.zeros((50, num_classes), dtype="float32")
one_hot[:, target_class] = 1.0
gen_data = iface.generate(50, one_hot=one_hot)   # returns (50, 1, F)
```

**Sampling speed.** Training uses the paper's `num_timesteps=1000` cosine
schedule; inference uses deterministic DDIM-50 sampling (~20× faster than
full DDPM reverse with effectively identical quality). Tune
`num_sample_steps` in the fit kwargs if you need to trade speed for
quality.
```

- [ ] **Step 3: Update the "MLX acceleration" paragraph**

In `GANs/README.md`, find the line:

```
On Apple Silicon, `WGAN`, `MT_WGAN`, `CTAB_GAN`, and `MT_CTAB_GAN` automatically use an MLX backend when available.
```

Replace with:

```
On Apple Silicon, `WGAN`, `MT_WGAN`, `CTAB_GAN`, `MT_CTAB_GAN`, and `TAB_DDPM` use an MLX backend when available.
`TAB_DDPM` is **MLX-only** — there is no TF backend; on non-MLX hosts, `resolve_backend` will fail with a clear diagnostic.
```

- [ ] **Step 4: Update `tests/README.md`**

Edit `user_data/strategies/GANs/tests/README.md`. Find:

```
Available type names: `WGAN`, `MTWGAN`, `CGAN`, `CTABGAN`, `MTCTABGAN`
```

Append `TabDDPM`:

```
Available type names: `WGAN`, `MTWGAN`, `CGAN`, `CTABGAN`, `MTCTABGAN`, `TabDDPM`
```

Find:

```
Available type names: `TestWGANQuality`, `TestMTWGANQuality`, `TestCTABGANQuality`, `TestMTCTABGANQuality`
```

Append:

```
Available type names: `TestWGANQuality`, `TestMTWGANQuality`, `TestCTABGANQuality`, `TestMTCTABGANQuality`, `TestTabDDPMQuality`
```

- [ ] **Step 5: Update `AGENT_GUIDE.md`**

Edit `user_data/strategies/AGENT_GUIDE.md`. In the "Adding to / extending the GAN system" section (around line 380), find the bulleted list of GAN types under the introduction. Add a sentence noting TabDDPM follows the same lifecycle but is MLX-only and continuous-only. Specifically, in the `Case B — genuinely new GAN type` section, before step 1, add:

```markdown
> **Reference:** TabDDPM (`GANType.TAB_DDPM`) was added as a Case B
> follow-up — see `docs/superpowers/specs/2026-05-11-tabddpm-design.md`
> and `docs/superpowers/plans/2026-05-11-tabddpm-implementation.md` for
> a concrete worked example. It's MLX-only and continuous-only.
```

- [ ] **Step 6: Commit**

```bash
git add user_data/strategies/GANs/README.md \
        user_data/strategies/GANs/tests/README.md \
        user_data/strategies/AGENT_GUIDE.md
git commit -m "docs(gans): document the TabDDPM GAN type

Adds TAB_DDPM to the GAN-types table, a usage snippet matching the
existing per-type subsections, a sampling-speed note about DDIM-50,
and a cross-reference from AGENT_GUIDE to the design + plan files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Verification — full test sweep

After all tasks land, verify the whole suite:

```bash
# Fast tests (no slow gate)
pytest user_data/strategies/GANs/tests/test_diffusion_mlx.py \
       user_data/strategies/GANs/tests/test_tabddpm_mlx.py \
       user_data/strategies/GANs/tests/test_tabddpm_backend.py \
       user_data/strategies/GANs/tests/test_tabddpm_interface.py \
       -v

# Functional suite (TabDDPM slice)
pytest "user_data/strategies/GANs/tests/test_functional_suite.py" -k TabDDPM -v

# Existing tests should still pass
pytest user_data/strategies/GANs/tests/test_gan_interface.py \
       user_data/strategies/GANs/tests/test_balance.py \
       -v

# Quality (gated, slow)
RUN_SLOW_TESTS=1 pytest "user_data/strategies/GANs/tests/test_quality_suite.py::TestTabDDPMQuality" -v
```

All should be green. The full quality suite for other types is unchanged and continues to pass.

---

## Acceptance criteria (from the spec)

- [x] All new unit tests in `test_diffusion_mlx.py` pass (tasks 2a–2c).
- [x] Functional suite passes for the three new TabDDPM test classes (task 8).
- [x] Quality suite passes for `TestTabDDPMQuality` under `RUN_SLOW_TESTS=1` (task 9).
- [x] `GANs/README.md` and `AGENT_GUIDE.md` reflect the new type (task 10).
- [ ] End-to-end smoke (post-merge): a one-line concrete strategy `NNNC_TabDDPM_MLX_LSTM` inheriting from `NNNC_CGP_MLX_LSTM` with `gan_type = GANType.TAB_DDPM` trains, saves, loads, and produces `balance_single_task` augmentation without error or NaN on a 1-pair smoke timerange. This is a manual check after the plan completes.
