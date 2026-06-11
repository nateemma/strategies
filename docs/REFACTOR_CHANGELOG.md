# Strategy Refactor Changelog

Every class moved / merged / renamed / behaviour-fixed during the hierarchy
refactor, with the reason. Plan: `docs/REFACTOR_PLAN.md`. Each batch is one
commit, reviewed with `/code-review`, and verified behaviour-neutral unless
explicitly marked as a behaviour change.

No class was renamed in any batch below — all names are kept stable so
`saved_data/<Name>/`, `<Name>.json` sidecars, and backtest/config references
keep resolving.

---

## B1 — NNNC config-only leaves → declarative `classifier_type` + MLX `get_classifier` hoist

**Behaviour:** neutral (verified). Net −79 lines across 34 files.

**New:**
- `NNNC/NNNClassifierMLX.py :: MLXClassifierMixin` — single home for the MLX
  `get_classifier()` (metal-availability guard + `create_classifier_mlx`) that
  was previously copy-pasted byte-for-byte into 3 bases. Also carries the MLX
  default `classifier_type = ClassifierTypeMLX.LSTM`.

**Changed (bases):**
- `NNNCStrategy` — added `classifier_type = NNNClassifier.ClassifierType.LSTM`;
  `get_classifier_type()` now returns `self.classifier_type` (was a hard-coded
  return). Keras `get_classifier()` unchanged.
- `NNNC_MLX`, `NNNC_CGP_MLX`, `NNNC_WGAN_MLX` — now inherit
  `(MLXClassifierMixin, <familybase>)`; deleted their duplicated `get_classifier`
  and `get_classifier_type`. All class attributes (gan_type, buy_params,
  gan_target_ratio, gan_synth_autoencoder_threshold, gan_run_diagnostics,
  use_post_gan_scaling, augment_training_data, entry_trend_*) preserved.

**Changed (28 leaves)** — each `def get_classifier_type(self): return <Enum>`
replaced by `classifier_type = <Enum>` (same enum value, verified):
`NNNC_CGP_{Attention,CNN,GRU,LSTM2,MLP,Multihead,TCN,Transformer,VAE,Wavenet,Wavenet3}`,
`NNNC_CGP_MLX_{Attention,CNN,GRU,LSTM,LSTM2,LSTM_KAN,MLP,Mamba,Mamba2,Multihead,TSMamba,Transformer,Wavenet}`,
`NNNC_MLX_{Mamba2,TSMamba,Transformer}`, `NNNC_WGAN_MLX_MLP`.

**Bug fixed (H1, behaviour-neutral):**
- `NNNC/NNNC_CGP.py` — removed the dead duplicate
  `gan_synth_autoencoder_threshold = 0.010`; the active value was already
  `0.005` (the later assignment won at class-body execution). Now a single
  `= 0.005`.

**Verification:**
- AST equivalence: every converted leaf's new attribute text == its old
  `get_classifier_type()` return expression.
- Import smoke (venv): all 32 affected modules import; MRO resolves
  `get_classifier` from the mixin for MLX classes and `classifier_type` to the
  correct enum (including production `NNNC_DDPM_MLX`).
- `pytest Framework/test_base_nn_strategy.py` → 118 passed.
- `/code-review` (2 finder angles + verify) → no findings.

**Known, not changed:** 4 MLX leaves
(`NNNC_CGP_MLX_{Mamba,Mamba2,Transformer}`, `NNNC_MLX_Transformer`) keep a
pre-existing unused `create_classifier_mlx` / `ClassifierKeras` import
(predates this refactor, `# flake8: noqa: F401`). Left as-is per surgical scope.

---

## B2 — NNMT consolidation (behaviour-neutral) + H4

**Behaviour:** neutral (verified). Net −156 lines across 15 files.

**New:**
- `NNMT/NNMTClassifierMLX.py :: MLXMultiTaskClassifierMixin` — single home for
  the multi-task MLX `get_classifier()` (the 4-arg `create_classifier_mlx` +
  `_apply_classifier_overrides` call) that was byte-identical across 5 classes.
  Deliberately does NOT define `_apply_classifier_overrides` — that stays
  resolved per-class via MRO (see below).

**Changed (bases):**
- `NNMTStrategy` — added `classifier_type = NNMTClassifier.ClassifierType.Multi_LSTM`;
  `get_classifier_type()` returns `self.classifier_type`.
- `BaseNNMTStrategy` — hoisted the **identical** `_balance_iteratively` and
  `_format_for_gan_scaler` (byte-for-byte from NNMT_DDPM/NNMT_WGAN), plus the
  `_apply_classifier_overrides` variant shared by NNMT_DDPM/NNMT_WGAN (hash
  `bd0f`). Added `Tuple` to the typing import (runtime-evaluated annotations).
- `NNMT_MLX`, `NNMT_DDPM_MLX`, `NNMT_WGAN_MLX` — now `(MLXMultiTaskClassifierMixin,
  <familybase>)`; dropped duplicated `get_classifier`/`get_classifier_type`;
  all attrs preserved. **`NNMT_MLX` keeps its own richer
  `_apply_classifier_overrides`** (hash `582a`, adds entropy-penalty wiring) —
  the mixin does not shadow it, so NNMT_MLX and its leaves behave exactly as
  before, while NNMT_DDPM_MLX/NNMT_WGAN_MLX continue using the base `bd0f`
  version (verified by MRO resolution check).

**Changed (NNMT_DDPM / NNMT_WGAN):** deleted their now-hoisted
`_apply_classifier_overrides` / `_balance_iteratively` / `_format_for_gan_scaler`.

**Changed (leaves):** `NNMT_MLX_{MultiLSTM,MultiAttention,Transformer}`,
`NNMT_CGP_Attention` → `classifier_type` attr. `NNMT_DDPM_MLX_MultiLSTM`,
`NNMT_WGAN_MLX_MultiLSTM` → `(MLXMultiTaskClassifierMixin, <base>)` + attr,
dropped their identical `get_classifier`. `NNMT_CGP_MLX_MultiLSTM` →
`classifier_type` attr only; its `get_classifier` (which deliberately omits the
`_apply_classifier_overrides` call) was preserved unchanged.

**Bug fixed (H4, behaviour-neutral):**
- `NNMT_DDPM` — removed dead per-task `gan_target_ratio` dict (it was shadowed
  by the later scalar `0.5`, which stays active).
- `NNMT_WGAN` — removed dead scalar `gan_target_ratio = 0.8` (it was shadowed by
  the later per-task dict, which stays active).

**Verification:** resolution check on 13 classes (`get_classifier`,
`_apply_classifier_overrides` providers + `classifier_type` + `gan_target_ratio`
all match pre-refactor); hoisted bodies hash-identical to HEAD; import smoke;
`pytest NNMT/tests + Framework/test_base_nn_strategy.py` → 120 passed;
`/code-review` (2 finder angles) → no findings.

**Observed, not changed (out of scope):** `NNMT_WGAN.gan_augment = True` while
its docstring says the 2-D dispatcher is turned off (`= False`) — a pre-existing
code/comment mismatch, flagged for your review (not touched).
