# NoisyCoconut latent multi-path for NNNC — design spec

**Date:** 2026-07-18
**Status:** approved, implementing
**Family:** NNNC (single-task 3-class Buy/Hold/Sell classifier)

## Motivation

COCONUT (Chain of Continuous Thought) is an LLM technique: replace discrete
text chain-of-thought tokens with continuous hidden states so a model "ponders"
in latent space using the same parameters plus more test-time compute.
**NoisyCoconut** is the training-free variant — perturb the continuous latent K
times to spawn diverging reasoning paths, then aggregate them by probability-mass
voting. On reasoning benchmarks this reduces errors by exploring multiple paths
in a single pass.

We want to test whether this mechanism improves the NNNC direction classifier.

### Honest framing (kept in scope on purpose)

Translated to NNNC this is a **test-time-compute change on the same OHLCV
inputs** — not new information. This project's repeatedly-confirmed finding is
that the information ceiling moves with *new information*, not a cleverer head
(triple-barrier is a data problem; the 85-indicator battery added zero). So this
is run as a **cheap-gated experiment** (like the fibonacci / breakout evals): the
first steps are the cheapest tests that could show life, and every stage
rejects-fast and is documented either way. A known plausible failure mode:
latent voting *smooths* predictions, and smoothing has backfired here twice
(label-smoothing → rolling-quantile inversion; adaptive-magnitude filter ran
inverse to intent). The σ-sweep surfaces that immediately.

## Production baseline being wrapped

`NNNC_DDPM_MLX` (gbb labels H=48 / thr=0.007, pred_threshold 0.6, TabDDPM
augmentation). Model backbone `_LSTMModel` in `NNNClassifierMLX.py`:

```
(B, seq, features) → Dense resize → Conv1d residual → LSTM
    → last timestep (B, F) → LayerNorm → bottleneck ELU → out → softmax(3)
```

The trained weights live in `saved_data/NNNC_DDPM_MLX/NNNC_DDPM_MLX.safetensors`.
**No retraining** is performed in this spec — every stage wraps this existing
model.

## Non-goals

- No retraining, no curriculum, no looped "pondering" cell (that is Stage 2,
  deferred and only justified if Stage 0 or 1 shows life).
- No new features, no changes to guards, threshold, labels, GAN chain, or any
  production config. Each stage is a clean A/B on the **prediction mechanism
  alone**.
- No pair-specific code.

## Staging (cheapest → most expensive)

### Stage 0 — Input-space jitter (pre-gate)

Weakest form of the hypothesis: does *any* test-time stochastic ensembling help?
Touches **no production model code** — wraps the full model.

```
seed once (fixed)
for k in range(K):
    p_k = model(X + sigma_in * eps_k)   # X already normalized → sigma_in scalar
p = mean_k(p_k)                          # probability-mass voting
```

- σ_in sweep `{0.01, 0.02, 0.05, 0.1}`, K = 16.
- Runs the full LSTM forward K times (K× cost) — still seconds on a backtest batch.

### Stage 1 — Latent-space jitter (the chosen COCONUT mechanism)

Perturb the continuous latent, not the input. Proceeds regardless of Stage 0's
sign (it is the mechanism selected); Stage 0 only calibrates expectations.

Requires a **byte-identical split** of `_LSTMModel.__call__` into:
- `encode(x)` → `lstm_last` `(B, F)` (after LSTM + LayerNorm, before bottleneck)
- `decode(h)` → bottleneck ELU → out → softmax

`__call__` becomes `decode(encode(x))` — identical math, identical parameter
names, so the existing `.safetensors` loads unchanged.

```
h = encode(X)                                   # (B, F), computed ONCE
seed once (fixed)
for k in range(K):
    h_k = h + sigma * per_dim_std(h) * eps_k    # perturb at lstm_last
    p_k = decode(h_k)                           # cheap decode-only path
p = mean_k(p_k)
```

- **Injection point `lstm_last`** (not post-bottleneck): each path decodes
  through the ELU bottleneck + output nonlinearity so paths genuinely diverge;
  perturbing after the bottleneck leaves only a linear decode and the votes
  collapse to the mean.
- **σ scaled per latent-dim batch std** (dims have different scales).
  σ sweep `{0.05, 0.1, 0.2, 0.4}`, K = 16.
- Encode runs once; only the decode head runs K times.

### Stage 2 — Looped pondering (retrain) — DEFERRED

Only justified if Stage 0 or 1 shows life. Not designed here.

## Components

### `NoisyCoconutMixin` (in `Predictors/`)

One shared mixin parametrized by:
- `noisy_perturb_space ∈ {'input', 'latent'}`
- `noisy_sigma` (float)
- `noisy_k` (int, default 16)
- `noisy_seed` (int, default 42)

Overrides only `predict()`. Aggregation is mean-softmax in both modes.

- `input` mode: calls the full model K times on jittered inputs. Works with the
  unmodified backbone.
- `latent` mode: calls `encode` once + `decode` K times. Requires the backbone
  to expose `encode`/`decode` (the `_LSTMModel` split). If the active backbone
  does not expose them, `latent` mode raises a clear error.

The predictor subclasses combining this mixin with `NNNClassifierMLX_LSTM` are
registered so a strategy can select them via `ClassifierTypeMLX`.

### Determinism

`predict()` is batch-called once over the dataframe in backtest. Seed the MLX
RNG once at the top of the perturbation loop → identical backtests reproduce
exactly (matches the seed=42 discipline). σ=0 must return exactly the production
softmax (identity) — a built-in correctness check.

### Strategies (in `NNNC/`)

Thin subclasses of `NNNC_DDPM_MLX`, changing only the classifier selection:
- `NNNC_DDPM_MLX_InJit` — input-space jitter (Stage 0)
- `NNNC_DDPM_MLX_Noisy` — latent jitter (Stage 1)

σ / K set as class attributes so the sweep varies one strategy attribute (no
config changes). Everything else inherited verbatim.

## Validation gate

Backtests only — no training loop.

- A/B: production `NNNC_DDPM_MLX` vs the jitter strategy across the σ sweep,
  K=16, **same timerange, seed=42, volume filter ON**.
- Report profit / Calmar / DD / trade-count / Buy-precision side by side.
- **σ=0 == production** (identity sanity).
- **Reject criterion (upfront):** if no σ beats production within run-to-run
  noise, the stage is rejected and documented. Watch specifically for the
  smoothing failure mode — fewer or lower-precision confident Buys.

## Success criteria

- Stage 0/1 code: σ=0 reproduces production byte-for-byte; `latent` mode loads
  the existing safetensors with no retrain; backtests run deterministically.
- Experimental verdict: a σ that beats production on risk-adjusted metrics
  (Calmar / DD) without a profit collapse → escalate to Stage 2. Otherwise →
  documented rejection, production untouched.
