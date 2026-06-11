# Strategy Refactor Plan

**Status:** Analysis complete — awaiting approval before any code edits.
**Scope:** `user_data/strategies/` only (nested git repo; the outer freqtrade tree is upstream and re-cloned — never touched).
**Date:** 2026-06-11

---

## 1. How this map was built

- **Structural facts via AST, not regex.** A Python `ast` pass over all 393 in-scope `.py`
  files extracted every `ClassDef` (name, base list, methods, class attributes), hashed each
  method body (docstring-stripped, normalized `ast.dump`) to detect byte-identical duplicates,
  and resolved the inheritance graph by simple-name base matching.
- **`ast-grep` (v0.43)** installed and used for spot structural queries.
- Three parallel read-only sub-agents deep-read the NNMT family, the hot paths, and the
  GAN/classifier duplication clusters; their line-level claims were spot-verified against source.
- Raw analyzer scripts: `/tmp/refactor_analyze.py`, `/tmp/refactor_analyze2.py` (re-runnable).

**Headline counts:** 783 `ClassDef`s total (incl. tests, NN layer/model helper classes,
GAN internals). The *strategy* tree rooted at `BaseStrategy` is ~150 classes. Consolidation is
concentrated in **NNNC (94 classes)**, **NNMT (70)**, and the **GANs trainers**.

---

## 2. Current inheritance map (strategy tree)

```
BaseStrategy (Framework/BaseStrategy.py)          ← IStrategy; ROI/stop/guards/custom_exit
├── BaseNNStrategy (Framework/BaseNNStrategy.py)   ← full ML pipeline (2668 L)
│   ├── NNNCStrategy (NNNC/)                        clf=NNNClassifier.LSTM (Keras)
│   │   ├── NNNC_CGP                                gan=CTAB_GAN
│   │   │   ├── NNNC_CGP_{Attention,CNN,GRU,LSTM2,MLP,Multihead,TCN,
│   │   │   │            Transformer,VAE,Wavenet,Wavenet3}   ← 11 config-only leaves
│   │   │   ├── NNNC_CGP_MLX                        clf=ClassifierTypeMLX.LSTM (re-decl gan)
│   │   │   │   └── NNNC_CGP_MLX_{Attention,CNN,GRU,LSTM,LSTM2,LSTM_KAN,MLP,
│   │   │   │                    Mamba,Mamba2,Multihead,TSMamba,Transformer,Wavenet}
│   │   │   │                                       ← 13 config-only leaves (13–31 L each)
│   │   │   └── NNNC_CGP_PCA
│   │   ├── NNNC_MLX                                clf=ClassifierTypeMLX.LSTM
│   │   │   ├── NNNC_DDPM_MLX  ← PRODUCTION         gan=TAB_DDPM
│   │   │   ├── NNNC_MLX_HighPrecision              (only diff: prediction_threshold band)
│   │   │   └── NNNC_MLX_{Mamba2,TSMamba,Transformer}   ← config-only leaves
│   │   ├── NNNC_PCA
│   │   └── NNNC_WGAN → NNNC_WGAN_MLX → NNNC_WGAN_MLX_MLP
│   ├── BaseNNMTStrategy → NNMTStrategy (NNMT/)     clf=Multi_LSTM (Keras)
│   │   ├── NNMT_DDPM → {NNMT_DDPM_MLX, NNMT_DDPM_MLX_MultiLSTM}
│   │   ├── NNMT_WGAN → {NNMT_WGAN_MLX, NNMT_WGAN_MLX_MultiLSTM}
│   │   ├── NNMT_CGP  → {NNMT_CGP_Attention, NNMT_CGP_MLX_MultiLSTM}
│   │   └── NNMT_MLX  → {NNMT_MLX_MultiLSTM, _MultiAttention, _Transformer, NNMT_CGP_MLX⚠}
│   ├── NNPredictStrategy (NNPredict/)  → NNPredict_{LSTM,MLX_LSTM,Ridge}, MultiHorizon
│   ├── NNAnomalyStrategy (Anomaly/) → NNGANomalyStrategy
│   ├── NNProfitStrategy (NNNC/)        ⚠ get_classifier* identical to NNNCStrategy
│   ├── SklearnStrategy (Sklearn/) → Skl_{RandomForest,XGBoost}{,_WGAN,_CGP}
│   ├── CreateGAN (GANs/) → Create{WGAN,CtabGanPlus,TabDDPM,Discriminator,...}
│   └── CreateScalers (Framework/)
├── SimpleStrategy (SimpleStrategies/)  ← 60+ single-indicator leaves; EhlersBase subtree
└── TSPredict (TSPredict/)  → TS_Coeff/TS_Gain/TS_Wavelet subtrees (~29)
```

⚠ marks an inheritance oddity (see §3). Full machine-generated tree: re-run
`python /tmp/refactor_analyze2.py`.

**Key shape observation:** the concrete strategy explosion is a *cartesian product*
`{plain, CGP, WGAN, DDPM} × {Keras, MLX} × {architecture}`, where each cell is a near-empty
subclass selecting one model architecture. This is the dominant pattern and the main
consolidation target.

---

## 3. Hierarchy issues & latent bugs (FLAG — do not silently merge)

These change behavior or encode a real decision; they need your call, not an automatic rewrite.

| # | Issue | Location | Recommendation |
|---|-------|----------|----------------|
| H1 | **Duplicate class-attr assignment, second wins.** `gan_synth_autoencoder_threshold` set to `0.010` then re-assigned `0.005` at class-body bottom (after a blank line, no EOF newline). Active value is `0.005`. | `NNNC/NNNC_CGP.py:44,57` | Confirm `0.005` is intended; delete the dead `0.010`. Behavior-affecting — your call. |
| H2 | **`NNMT_CGP_MLX` inherits `NNMT_MLX`, not `NNMT_CGP`.** So it does *not* inherit `gan_type=MT_CTAB_GAN` and re-declares it. Breaks the "base + _MLX child" pattern that DDPM/WGAN follow. | `NNMT/NNMT_CGP_MLX.py:22` | Re-parent to `NNMT_CGP` + an MLX mixin, OR leave and document. Affects MRO/attrs — needs decision. |
| H3 | **Possible missing call.** `NNMT_DDPM.preprocess_training_data` calls `self._apply_gan_inference_overrides(interface)`; the otherwise-parallel `NNMT_WGAN` version does **not**. | `NNMT/NNMT_DDPM.py` vs `NNMT_WGAN.py` | Investigate intent before consolidating — may be a bug or deliberate. Not a refactor edit. |
| H4 | **Redundant per-task dict then scalar.** `gan_target_ratio` declared as a per-task dict, later overridden by a scalar in the same/child class. | `NNMT/NNMT_DDPM.py` | Drop the dead dict if scalar is intended. |
| H5 | **Redundant subclass.** `NNProfitStrategy.get_classifier`/`get_classifier_type` are byte-identical to `NNNCStrategy`'s; no other distinguishing logic surfaced. | `NNNC/NNProfitStrategy.py` | Confirm it's still used; if vestigial, mark for removal (separate from refactor). |
| H6 | **`use_post_gan_scaling` everywhere.** Set `True` on nearly every GAN-consuming class individually. Memory notes v2/post-scaling underperforms in some configs. | many | Not a refactor target — flagging the scatter. Could default in a base once you've settled it. |

**Layering:** GAN *builder* strategies (`CreateGAN`, `CreateMTGAN` and children) live in `GANs/`
but inherit from `BaseNNStrategy`/`NNMTStrategy` (Framework/NNMT). This is intentional reuse of
the pipeline, not a true violation — noted, not actioned. No base was found importing from a
leaf strategy dir.

---

## 4. Consolidation opportunities (ranked)

Risk = chance a careful, name-preserving change alters runtime behavior.
"Lines" = realistic collapse, not gross duplication.

| # | Opportunity | Files | Lines collapsible | Risk | Priority |
|---|-------------|-------|------------------|------|----------|
| C1 | **Config-only leaves → declarative attribute.** 38 leaf classes override only `get_classifier_type()` to return an enum. Replace with `classifier_type = <enum>` class attr read by ONE base method; delete the 38 method overrides and the **x3 (NNNC) + x5 (NNMT) byte-identical `get_classifier` bodies** (~14 L each) by hoisting them into the Keras-base / MLX-base. | 38 leaves + 6 bases | ~110 (overrides) + ~110 (dup `get_classifier`) ≈ **220** | **LOW** | **1** |
| C2 | **NNMT sibling copy-paste → `BaseNNMTStrategy`.** `_apply_classifier_overrides`, `_balance_iteratively`, `_format_for_gan_scaler` are byte-identical across `NNMT_DDPM`/`NNMT_WGAN`. | 2→1 | **~50** | LOW | 2 |
| C3 | **`ClassifierBase` extraction (utils).** 5 independent backends (`ClassifierKeras/MLX/Sklearn/Darts/PyTorch`) share byte-identical stubs (`needs_dataframes`, `prescale_data`, `mad_score` ×5, `model_exists`, `model_is_trained`, `set_batch_size`, `returns_single_prediction`, …). No common parent today. | 5 + new base | **~170** | LOW | 3 |
| C4 | **Indicator-math single-source.** `super_smoother`, `adaptive_super_smoother`, `rolling_normalize`, `safe_cg`, `ewo`, `williams_r` duplicated between `utils/DataframePopulator.py`, `SimpleStrategies/EhlersBase.py`, `TSPredict/TSPredict.py`. Make `DataframePopulator` (or a small `indicators` util) the source; others import. | 3 | ~80 | **MED** (feeds features/labels — must stay bit-identical) | 4 |
| C5 | **Anomaly/Sklearn shared NN blocks.** `TransformerBlock` (×3), `attention_encoder`, `create_encoder` duplicated across `Anomaly/NN*Classifier.py` and `Sklearn/NNDetector.py`. Extract to a shared `nn_blocks` module. | 3 | ~120 | MED | 5 |
| C6 | **GAN single↔multi-task trainers.** `df_ctab_gan.py` (2545) ↔ `df_mt_ctab_gan.py` (2260): ~79 L byte-identical + ~1100 L 70–90% similar (2D vs 3D label shaping). `df_wgan_gp` ↔ `df_mt_wgan_gp`: 45 L identical layer classes + shared infra. MLX pair: smaller overlap. | 6 | 124 safe; up to ~1000+ if training loops unified | **HIGH** (GAN training dynamics; needs retrain-parity) | 6 (last / optional) |
| C7 | **Debug duplicates.** `DebugRegimeIndicator` ↔ `DebugTradingType` share `populate_*` + `emulate_*_signals` (~90 L). Debug-only. | 2 | ~90 | LOW | opportunistic |
| C8 | **TrainingSignals / Wavelets / Detrenders / Forecasters** plugin families with many near-identical small methods (`get_debug_indicators`, `forecast`, `retrend_1d`, `coeff_to_array`). | 4 | ~150 | MED–HIGH (TrainingSignals drives **labels**) | defer |

**Note on C1:** the 38 leaf *files* cannot be deleted — freqtrade discovers strategies by
class-name = importable symbol, and `saved_data/<Name>/` + `<Name>.json` sidecars reference them.
The win is removing duplicated *logic* and making selection declarative, not reducing file count.

---

## 5. Efficiency findings (hot paths)

Confirmed against source (✓) or flagged for in-batch verification (?). Each fix must reproduce
identical backtest output.

| # | Location | Problem | Fix | Risk | Conf |
|---|----------|---------|-----|------|------|
| E1 | `Framework/BaseStrategy.py:1243` | `dataframe["volume"].rolling(20,min_periods=5).mean().iloc[-1]` recomputed **inside `custom_stoploss`** (called many times per trade). | Precompute a `volume_ma_20` column once in `populate_indicators`; read `.iloc[-1]`. | LOW | ✓ |
| E2 | `Framework/BaseNNStrategy.py:778,839` | `rolling_dataframe_normalise` does two full `df.copy()` per call (per pair, per iteration). | Drop the second copy; scale into `df_to_scale` (or copy only the scaled columns). | MED (aliasing) | ✓ |
| E3 | `utils/DataframePopulator.py` (~845–850) | Post-scale **full-column min/max scan** over every column purely to emit a debug warning. | Gate behind the debug flag; or `.describe()` once on scaled cols only. | LOW | ? |
| E4 | `utils/DataframePopulator.py` (~208–216) | `adaptive_super_smoother`: `np.exp/np.cos` coefficients recomputed every row even when period repeats. | Memoize `period→coeffs`. | LOW–MED | ? |
| E5 | `Framework/BaseNNStrategy.py:~1034` | `window_and_flatten` builds sequences with a per-window Python slice-copy loop. | `np.lib.stride_tricks.sliding_window_view` (view, no copy). | MED (layout-sensitive) | ? |
| E6 | `Framework/BaseStrategy.py:~939` | `check_precision_columns` chains `.apply("{:.15f}".format).str.extract().str.len()` per column. | Vectorize with `np.format_float_positional`; runs rarely. | LOW | ? |

`populate_entry_trend` / `populate_exit_trend` / `confirm_trade_entry` / `is_bear_market`
are already vectorized and clean — no action.

---

## 6. Proposed target structure

The cartesian explosion collapses to **declarative leaves + thin family bases**:

```
Framework/
  BaseStrategy, BaseNNStrategy            (unchanged API)
utils/
  ClassifierBase  (NEW)  ← Keras/MLX/Sklearn/Darts/PyTorch inherit; shared stubs live here
  DataframePopulator      ← single source for indicator math (EhlersBase/TSPredict import)
  nn_blocks  (NEW)        ← TransformerBlock / attention_encoder / encoders (Anomaly+Sklearn)

NNNC/  NNMT/  (family bases gain ONE get_classifier + ONE get_classifier_type-from-attr)
  e.g.  class NNNC_CGP_MLX_Attention(NNNC_CGP_MLX):
            classifier_type = ClassifierTypeMLX.Attention     # was a 3-line method override
```

Family bases keep two `get_classifier` implementations only where the backend genuinely differs
(Keras `create_classifier` vs MLX `create_classifier_mlx`) — not one per leaf.

GAN trainers (C6) optionally grow a shared `*_GAN_base` with single/multi-task subclasses, but
only after retrain-parity is proven — recommended as a final, separately-approved phase.

---

## 7. Batched execution plan

Each batch is one family/cluster, committed to the **nested** repo (`git -C user_data/strategies`),
then run through the `/code-review` plugin; high-confidence findings addressed before the next batch.

| Batch | Content | Verification |
|-------|---------|-------------|
| **B1** | C1 for NNNC (Keras + MLX subtrees): method→attribute, hoist `get_classifier`. | Backtest `NNNC_MLX`, `NNNC_CGP_MLX`, `NNNC_DDPM_MLX`, `NNNC_WGAN_MLX` (saved models exist) on a fixed timerange → **identical** result. Import-smoke every leaf; assert `get_classifier_type()` unchanged. |
| **B2** | C1 for NNMT + C2 (hoist helpers). Surface H2/H3/H4 for decision. | Backtest `NNMT_MLX`, `NNMT_CGP_MLX`, `NNMT_DDPM_MLX` identical. `NNMT/tests/`. |
| **B3** | C3 `ClassifierBase`. | `pytest utils/test_classifier.py` + the per-backend tests; one NN backtest identical. |
| **B4** | C4 indicator single-source. | `scripts/check_bias.sh` + identical backtest on an affected strategy. |
| **B5** | C5 `nn_blocks` extraction. | `pytest` Anomaly/Sklearn tests; import smoke. |
| **B6** | Efficiency E1–E2 (then E3–E6 individually). | Each: identical backtest before/after; profile delta. |
| **B7** | C6 GAN trainers — **separate approval**. | `GANs/tests/*` full suite + GAN retrain output parity. |

C7/C8 are opportunistic / deferred.

---

## 8. How to verify a batch didn't change results

- **Behavior contract:** pure refactor ⇒ a strategy with a saved model must produce a
  **bit-identical backtest** (same trades, same totals) on a fixed timerange. Class names are
  kept stable, so `saved_data/<Name>/` keeps resolving and no retrain is needed for the 8
  strategies with saved models (`NNNC_{MLX,CGP_MLX,DDPM_MLX,WGAN_MLX}`, `NNMT_{MLX,CGP_MLX,DDPM_MLX}`,
  `NNAnomalyStrategy`).
- **Command:** `zsh scripts/test_strat.sh NNNC NNNC_DDPM_MLX` (+ a pinned `--timerange`) before
  and after; diff the summary + trade CSV.
- **Leaves without saved models** (Attention/CNN/GRU/…): import-smoke + assert the returned
  classifier enum is unchanged (no retrain).
- **utils / GAN / Framework:** the existing `pytest` suite (Framework, utils, GANs/tests, NNMT/tests).
- **Lookahead guard:** `scripts/check_bias.sh` after any change touching feature/label math (B4).
- A `docs/REFACTOR_CHANGELOG.md` will be created at B1 and appended every batch: each class
  moved/merged/renamed + why.

---

## 9. Decisions I need from you before editing

1. **Renames:** I recommend **zero renames** (keep every class name → no config/saved_data churn).
   Confirm, or tell me which renames you'll accept (I'll provide a migration list + move
   `saved_data/`/`.json`).
2. **Bugs H1–H4:** these affect behavior. Want me to (a) only flag them in the changelog and
   leave as-is, or (b) fix H1/H4 (dead duplicate assignments) since the active value is unchanged?
   H2/H3 I'd leave untouched pending your read.
3. **Scope of first pass:** start with **B1–B3 (all LOW risk, ~440 lines, no behavior change)**
   and stop for review before the MED-risk batches? Recommended.
4. **GAN trainers (C6, B7):** in scope now, or defer until the LOW/MED batches land?

I'll wait for approval before touching any code.
