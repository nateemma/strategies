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
