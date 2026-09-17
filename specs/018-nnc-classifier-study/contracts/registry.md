# Registry Contract: new `ClassifierTypeMLX` members and training-loop attributes

## Training-loop attributes (`Predictors/MLXClassifierNary.py`)
| attribute | default (= current literal) | effect |
|---|---|---|
| `max_epochs` | 100 | replaces the local `max_epochs` |
| `early_patience` | 20 | replaces the local `early_patience` |
| `weight_decay` | 0.0 | `> 0` → `optim.AdamW(learning_rate, weight_decay)`; else `optim.Adam` (unchanged) |
| `epochs_run`, `best_epoch` | set by `train()` | read by the harness |
| `model.extra_loss` | optional attribute on the module | if present, added to the batch loss (used by `STG_GATE`); absent → no change |

Existing behaviour is byte-identical when the attributes are left at defaults
(`test_training_attrs.py`).

## New members (all `create_model(seq_len, num_features)` overrides reading `self.hp`)
| member | hp keys (defaults) | notes |
|---|---|---|
| `RIDGE` | `weight_decay` via `train` (1e-2) | Flatten → Linear(3) |
| `ROCKET` | `n_kernels` (1000), `kernel_len` (9), `feature_norm` (standardize\|none) | frozen random kernels, PPV+max → per-feature standardisation calibrated on the first training batches and frozen → Linear(3). `none` = pre-fix behaviour, seed-unstable (4/9 runs stalled at epoch 0) |
| `LSTM_FAM` | `hidden` (64), `depth` (1), `bidirectional` (false), `dropout` (0.25), `pool` (last) | pure RNN, no conv front-end |
| `GRU_FAM` | same as LSTM_FAM | |
| `TCN_FAM` | `channels` (64), `kernel` (3), `depth` (3), `dilation_base` (1), `residual` (false), `dropout` (0.25) | causal conv stack, global-avg-pool head |
| `TRANSFORMER_FAM` | `d_model` (64), `heads` (4), `layers` (2), `ffn_mult` (2), `pos` (learned\|none\|rope), `dropout` (0.25) | pre-LN encoder, mean pool |
| `MLP_FAM` | `width` (256), `depth` (2), `dropout` (0.25) | Flatten → MLP |
| `CNN_LSTM` | `channels` (64), `kernel` (3) + LSTM_FAM keys | phase 5 |
| `CNN_TRANSFORMER` | `channels`, `kernel` + TRANSFORMER_FAM keys | phase 5 |
| `TCN_RES` | TCN_FAM keys with `residual=true` | phase 5 |
| `RNN_ATTNPOOL` | LSTM_FAM keys + `attn_dim` (32) | attention pooling over RNN states |
| `INCEPTION_MS` | `channels` (32), `kernels` ([3,5,9]), `depth` (2) | multi-scale parallel kernels |
| `TSMIXER` | `blocks` (2), `ff_mult` (2), `dropout` | time-mix + feature-mix MLP |
| `ITRANSFORMER` | `d_model` (64), `heads` (4), `layers` (2) | feature-as-token attention; logs attention means |
| `XLSTM` | `hidden` (64), `depth` (1), `heads` (4) | mLSTM block |
| `TRANSFORMER_MODERN` | TRANSFORMER_FAM keys, fixed RMSNorm + RoPE + SwiGLU | |
| `PATCHTST` | `patch_len` (16), `stride` (8), `d_model` (64), `heads` (4), `layers` (2) | requires `seq_len ≥ 4·patch_len`; refuses otherwise |
| `STG_GATE` | `backbone` (member name), `lam` (1e-3), `sigma` (0.5) + backbone keys | per-feature stochastic gate; exposes `gate_importance()` |

Every member: enum entry + class in `NNNC/NNNClassifierMLX.py`; no other registration.
A strategy selects any member with `classifier_type = ClassifierTypeMLX.<MEMBER>` and gets
the defaults (FR-015).
