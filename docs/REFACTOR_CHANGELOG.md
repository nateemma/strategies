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

---

## B2b — H3 fix (⚠ BEHAVIOUR CHANGE — isolated commit)

- `NNMT_WGAN.preprocess_training_data` now calls
  `self._apply_gan_inference_overrides(interface)` after loading the GAN, at the
  same point `NNMT_DDPM` already did. Previously `NNMT_WGAN` skipped it.
- **This changes NNMT_WGAN/NNMT_WGAN_MLX training behaviour** (inference-time GAN
  overrides are now applied). `NNMT_WGAN_MLX` has a saved model — **re-backtest /
  retrain to validate** before relying on it. Committed separately so it can be
  reverted independently of the neutral B2 consolidation.
- `_apply_gan_inference_overrides` is defined on `BaseNNStrategy:1442`.

---

## H2 — NOT changed (decision: keep current inheritance)

`NNMT_CGP_MLX` inherits `NNMT_MLX` (not `NNMT_CGP`) and re-declares
`gan_type = GANType.MT_CTAB_GAN`. This looks like an oddity, but re-parenting it
to `NNMT_CGP` would drop the MLX trading-head tuning that comes from `NNMT_MLX`
(`_CLASSIFIER_TASK_WEIGHTS = {trading:4, regime:2, …, profit:3}`,
`_CLASSIFIER_ENTROPY_PENALTY = {trading:0.5, …}`, and NNMT_MLX's richer
`_apply_classifier_overrides`). That tuning is load-bearing for entry quality,
so **the current inheritance is intentional and is kept as-is.** The `gan_type`
re-declaration is the correct trade-off, not a bug to "fix" by re-parenting.

---

## B3 — extract `utils/ClassifierBase`

**Behaviour:** neutral (verified, incl. bit-identical backtest). Net −25 lines.

**New:** `utils/ClassifierBase.py` — shared base for the 5 classifier backends
(`ClassifierKeras/MLX/Sklearn/Darts/PyTorch`), which previously inherited from
nothing and each re-declared the same stubs. Hoisted only methods that are
**byte-identical across every backend that defines them** (verified by AST hash):

- Universal (identical in all 5, deleted from all): `needs_clean_data`,
  `needs_dataframes`, `prescale_data`, `returns_single_prediction`.
- `mad_score` (identical in Keras/MLX/Darts/PyTorch; Sklearn lacked it → now
  inherits it, purely additive).
- `model_is_trained` and `get_model_root_dir` — base carries the
  Sklearn/Darts/PyTorch body (`return self.is_trained` / `utils/models/`);
  **Keras and MLX keep their own overrides** (different bodies), so their
  behaviour is unchanged. (`get_model_root_dir` uses `Path(__file__)`; base lives
  in `utils/` like the originals, so the resolved path stays `utils/models/`.)

**Changed:** the 5 backends now `class ClassifierX(ClassifierBase)` and had their
redundant copies deleted (codemod asserted each deleted body's hash matched the
expected value before removing it).

**Verification:**
- `ClassifierBase`'s 7 method bodies hash-match the canonical originals.
- Runtime resolution check (Keras/MLX/Sklearn + subclasses MLXNary / MLXMultiTask
  / KerasNary): every one of the 7 methods resolves via MRO to the **same body**
  it had pre-refactor. Darts/PyTorch can't be imported in this env (no `torch`,
  pre-existing) but are statically proven (deleted-body hashes == base bodies).
- Bit-identical backtest: `NNNC_DDPM_MLX` (86 result rows) and `NNMT_DDPM_MLX`
  (78 rows) byte-for-byte unchanged vs the B2 commit.
- `utils/test_classifier.py` fails identically on master (pre-existing Keras-version
  issue, unrelated).
- `/code-review` → no findings.

**Not hoisted (left as backend overrides):** methods that diverge across backends
(`save`, `load`, `train`, `predict`, `__init__`, `create_model`, `set_model_path`,
etc.) and the subset-identical ones (`reconstruct`/`transform`, `backtest`,
`new_model_created`) — a future opportunity, deferred to keep this batch
provably neutral.

---

## B6 — efficiency (E2)

**Behaviour:** neutral (bit-identical backtest). 

- `BaseNNStrategy.rolling_dataframe_normalise` (E2) — removed the redundant
  second `df.copy()`. `df_to_scale` is already a local copy of the caller's df,
  so the scaler transform now writes in place; the subsequent `np.clip` already
  returns a fresh frame. One fewer full-frame copy per normalize call (per pair,
  per iteration). Verified `NNNC_DDPM_MLX`/`NNMT_DDPM_MLX` backtests byte-identical.

**E1 dropped:** the `custom_stoploss` volume rolling-mean is gated by `after_fill`,
so it runs once per trade entry, not per-candle as the plan assumed — not worth
broadening the shared `populate_indicators` path. E3–E6 (DataframePopulator debug
scans, `adaptive_super_smoother` memoization, `sliding_window_view`) remain
available but unaddressed; lower confidence / lower value.

---

## B4 / B5 — EVALUATED, NOT EXECUTED (flagged, with reasons)

After measuring byte-identity, both were judged poor risk/reward and left for an
explicit future decision rather than executed.

**B4 — indicator-math dedup (DataframePopulator ↔ EhlersBase/EWO/TSPredict):**
Byte-identical pairs confirmed: `DataframePopulator` == `EhlersBase` for
`super_smoother`, `adaptive_super_smoother`, `rolling_normalize`, `safe_cg`;
`DataframePopulator` == `EWO` for `ewo`; `DataframePopulator` == `TSPredict` for
`williams_r`. **But** `TSPredict.super_smoother` has *drifted* (different body) —
must NOT be merged. Real dedup requires editing the production `DataframePopulator`
to delegate to a shared `utils/indicators.py`, purely to remove duplication in
non-production simple strategies. Recommendation: only do this if/when
`DataframePopulator` is being touched anyway; gate with a bit-identical backtest.

**B5 — TransformerBlock dedup (Anomaly NNGANomaly/NNAnomaly):**
The `TransformerBlock` (`tf.keras.layers.Layer`) is byte-identical between
`Anomaly/NNGANomalyClassifier.py` and `Anomaly/NNAnomalyClassifier.py` (the
`Sklearn/NNDetector.py` copy has drifted — only `call` matches). HOWEVER:
`NNGANomalyClassifier` loads its model via `keras.models.load_model(path,
compile=False)` with NO `custom_objects` and the class has no
`@register_keras_serializable`. There is a saved `saved_data/NNAnomalyStrategy.keras`.
Moving the class to a shared module risks breaking deserialization of that saved
model (Keras resolves custom layers by stored module/class name). Not worth the
risk for a non-production dedup without a model-load verification. Recommendation:
if pursued, re-export from the original modules to preserve the qualified name and
verify the saved model still loads before/after.
