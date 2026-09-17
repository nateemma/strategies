# Research: NNC Classifier Architecture & Input-Data Study

Part A records the repo-grounded decisions (R1–R13). Part B is the web-backed literature
review for phase 6 with citations and the ranked shortlist.

---

## Part A — Repo-grounded decisions

### R1. Data path: feathers → indicators → labels, per pair, continuous

- **Decision**: `data.py` reads `user_data/data/binanceus/<PAIR>_USDT-15m.feather` for the 11
  production pairs (`config/config.json` static whitelist: ZEC, XRP, SOL, LINK, NEAR, AAVE, SUI,
  AVAX, LTC, BCH, DOT), keeps bars from 2022-01-01, and calls
  `DataframePopulator().add_indicators(df, DatasetType.MINIMAL)` on each pair's **continuous**
  series exactly as `BaseNNStrategy.populate_indicators` does (line 1390). `guard_metric`,
  `bb_width` and `rvol` are produced by `add_indicators` (DataframePopulator.py:251, 279, 335).
- **Rationale**: indicators and labels must be computed on the uncut series so that warm-up
  and the label look-ahead are correct at every window boundary; row selection by date
  happens after windowing (R5).
- **Alternatives**: `evaluation/run_folds.py` (drives freqtrade backtests, too heavy);
  `Debug/DebugSignalLearnability.py` (already does feather → `add_indicators(MINIMAL)` →
  `get_train_buy_signals`, line 724/816/277, but skips the peak filter and conflict resolution,
  so its labels are NOT production labels).

### R2. Labels: class-attribute shim over `BaseNNStrategy`'s own methods

- **Decision**: `labels.py` builds a `_LabelShim` object whose attributes are read from the
  `NNNC_MLX` **class object** (no instantiation): `HORIZON`, `MIN_BUY_GAIN_THRESHOLD`,
  `MIN_SELL_LOSS_THRESHOLD`, `TRAINING_TYPE`, `LABEL_RVOL_THRESHOLD`, `LABEL_FILTER_WINDOW`,
  `apply_label_conflict_resolution`, plus `dbg_verbose=False`. It then calls
  `BaseNNStrategy.get_training_labels(shim, df)` (unbound), which in turn calls
  `get_train_buy_signals` / `get_train_sell_signals` (peak filter at `LABEL_FILTER_WINDOW or
  HORIZON`) and `augment_training_signals` (sell overrides buy; buy deleted if a sell follows
  within N=2 bars). Output is an int array in `TradingAction` encoding SELL=0, HOLD=1, BUY=2
  (`Framework/BaseStrategy.py:84-87`).
- **Fidelity check**: `scripts/check_label_fidelity.py` loads `NNNC_MLX` through
  `freqtrade.resolvers.StrategyResolver` with `config/config.json`, calls its
  `get_training_labels` on one pair's populated dataframe, and asserts
  `np.array_equal(shim_labels, strategy_labels)`. Run once before phase 2; output pasted into
  the report. If `NNNC_MLX` overrides any of the label methods, the shim MUST use the
  override (`test_labels_shim.py` asserts the attribute/method set matches the class).
- **Alternatives**: instantiate the strategy (Parameter singletons, config coupling);
  re-implement the label (drift risk, explicitly forbidden by FR-007).

### R3. Split set v1: three windows, anchored expanding train, 96-bar embargo

| id | evaluation window | training span (anchored) | validation slice | distant? |
|---|---|---|---|---|
| A | 2023-07-01 → 2023-12-31 | 2022-01-01 → 2023-06-30 00:00 (train_end = window_start − 96 bars) | last 10% of the train span by time, preceded by its own 96-bar embargo | yes (2023) |
| B | 2025-01-01 → 2025-06-30 | 2022-01-01 → 2024-12-31 00:00 | same rule | no |
| C | 2026-01-01 → 2026-06-30 | 2022-01-01 → 2025-12-31 00:00 | same rule | no (deployment regime) |

- **Embargo**: labels look ahead `HORIZON=48` bars; the peak filter uses the same window and
  conflict resolution looks 2 bars further. 96 bars (= 1 day at 15m) is 2× the minimum and
  costs nothing. `EMBARGO_BARS = 96` is a pinned constant; `test_splits.py` asserts
  `train_end + 96 bars ≤ window_start` and `val_end + 96 bars ≤ ...` for every window.
- **Pair inclusion rule**: a pair is included in a window if it has at least 365 days of bars
  before that window's validation slice; otherwise it is excluded from that window entirely
  (train, val and test) and listed in `pairs_excluded` on every row. Under this rule XRP
  (from 2023-07-14) and SUI (from 2023-10-25) are excluded from window A only.
- **Rationale**: Constitution II needs a distant era; window C is the most recent half-year
  (Constitution I). Three windows were chosen for the 40-hour budget.
- **Alternatives**: `evaluation/foldplan.py`'s 15 anchored 90-day folds (no embargo, no
  validation slice, ~5× the runs); four windows (spec draft, dropped in clarification).

### R4. Leakage controls (Constitution XI) — the complete list

1. Split geometry per R3, validated by tests.
2. **Scaler**: `RobustScaler` fitted on the training-span rows of the included pairs only, on
   `needs_norm = include_list − pre_normalized_columns` (names taken from
   `FeatureNormalizer`, never its fitted object), then `transform` on val and test; `clip(−10,
   10)` after transform as production does (FeatureNormalizer.py:398). Never reads
   `saved_data/*/main_scaler`; never calls `scale_dataframe`.
3. **Class weights**: computed from training-span labels with the same inverse-frequency
   formula as `get_training_class_weights` and passed explicitly to `train()`, because
   `train()` otherwise derives them from the validation labels (MLXClassifierNary.py:316).
4. **Early stopping**: on the validation slice inside the training span, never on test.
5. **Windows**: built on each pair's continuous series, then rows selected by date; the first
   `seq_len−1` bars of context for a test row come from the embargo period (features only,
   causal by construction).
6. **Feature selection / importance**: STG gates are trained on train; permutation importance
   is *evaluated* on the test window but never feeds back into any configuration choice
   within the same window.
7. Tests: `test_leakage.py` perturbs every test-window bar (×1000, sign flip) and asserts the
   fitted scaler parameters, the class weights and the training tensors are bit-identical.

### R5. Reusing `MLXClassifierNary.train()` / `predict()` unchanged

- Signature: `train(X_train, X_val, y_train_onehot, y_val_onehot, force_train=False,
  class_weights=None, sample_weights=None)`; `predict(X) → (N, 3)` softmax
  (`Predictors/MLXClassifierNary.py:185, 514`). Inputs are `(N, seq_len, nf)` tensors from
  `DataframeUtils.df_to_tensor(df, seq_len, method=3)` (one causal window per row) and one-hot
  labels from `DataframeUtils.one_hot_encode(labels, 3)`.
- Seed: set `clf.train_seed = seed` before `train()` (seeds mx, numpy, random; line 207-212).
- Model creation is lazy inside `train()` via `create_model(seq_len, num_features)`; the
  harness sets `clf.hp` (R7) beforehand.
- Checkpoint: `clf.checkpoint_path` is pointed at `research/clfstudy/.cache/ckpt/<run_key>.safetensors`
  per run; best weights are restored at the end of `train()` (line 502). Weights are saved with `clf.save(path)` only
  under `--save-weights`.
- Parameter count: `sum(p.size for _, p in tree_flatten(model.trainable_parameters()))`
  (same expression as line 321).
- **Required edit** (behaviour-preserving): `max_epochs`, `early_patience`, `weight_decay`
  become instance attributes with defaults `100`, `20`, `0.0`; `optim.AdamW(lr,
  weight_decay=wd)` is used only when `weight_decay > 0`; `self.epochs_run` and
  `self.best_epoch` are set at the end of `train()`. Existing strategies see identical
  behaviour.

### R6. Trivial baselines and what "ridge" means here

- **Majority**: predict HOLD for every test row; probabilities = training-span class
  frequencies (so log-loss/AUROC/ECE are defined).
- **Persistence**: predict the previous bar's *true* label; probabilities = one-hot of that
  label smoothed with ε=1e-3 (log-loss defined). This is an oracle-ish baseline and is labelled
  as such.
- **RIDGE** (registry type): `Flatten(seq_len × nf) → Linear(3)`, trained through the shared
  loop with `weight_decay = λ` (default 1e-2) — i.e. L2-penalised multinomial logistic
  regression under the study's focal loss and class weights. This is "ridge-logistic" as the
  spec allows; it is not closed-form ridge. A one-off cross-check against
  `sklearn.linear_model.LogisticRegression` on the same tensors is recorded in the report so
  the reader can see the two agree to within seed noise.
- **ROCKET** (registry type): MiniRocket-style frozen random kernels — 1,000 kernels of
  length 9 with weights in {−1, 2}, random dilations up to `(seq_len−1)//8`, biases drawn from
  the quantiles of the kernel outputs on a training batch — applied per channel with
  `mlx.nn.Conv1d`, pooled to PPV and max per kernel (2,000 features), then `Linear(3)` with
  `weight_decay`. Kernels are frozen with `model.freeze(keys=...)` so only the head trains,
  through the identical loop. Rationale: sktime/aeon are absent; this keeps the baseline in
  the registry and on the same path (Part B, ref 12, 14).

### R7. Parametrised family types and hyperparameter injection

- Every new type reads hyperparameters from `self.hp` (a dict set by the harness after
  `create_classifier_mlx` and before `train()`), falling back to per-type defaults that are a
  valid, strategy-usable configuration. Keys per type are the contract in
  `contracts/registry.md`.
- Phase-4 anchors and axes (one axis at a time, Constitution VIII):
  - `LSTM_FAM`/`GRU_FAM`: anchor hidden=64, depth=1, uni. Size axis hidden ∈ {16, 64, 256};
    structural axes depth=2, bidirectional=True at the anchor.
  - `TCN_FAM`: anchor channels=64, kernel=3, depth=3, dilation base 1. Size axis channels ∈
    {16, 64, 256}; structural axes kernel=5, dilation base 2 (depth 4), depth ∈ {2, 6}.
  - `TRANSFORMER_FAM`: anchor d_model=64, heads=4, layers=2, ffn_mult=2, pos=learned. Size
    axis d_model ∈ {32, 64, 128}; structural axes heads=8, layers ∈ {1, 4}, ffn_mult=4,
    pos ∈ {none, rope}.
  - `MLP_FAM`: anchor width=256, depth=2. Axes width ∈ {64, 256, 1024}, depth ∈ {1, 3}.
- Liveness (Constitution IV): `test_registry_synthetic.py` asserts parameter counts differ
  across the size axis of every family.

### R8. Metrics implementation

sklearn: `balanced_accuracy_score`, `f1_score(average="macro", zero_division=0)`,
`matthews_corrcoef`, `precision_recall_fscore_support(labels=[0,1,2], zero_division=np.nan)`
(per-class, NaN where undefined and flagged), `log_loss(labels=[0,1,2])` after clipping,
`roc_auc_score(multi_class="ovr", average=None)` per class (NaN if a class is absent) and its
nanmean, `confusion_matrix(labels=[0,1,2])`. ECE: 15 equal-width bins on `max(prob)`,
`Σ (n_b/N)·|acc_b − conf_b|`, plus the bin table. Secondary: BUY precision and coverage at
`p_buy ≥ 0.8`. Across-seed std uses ddof=1 (with 2 seeds this is |a−b|/√2; the gate output prints the seed
count beside every std). Timing: `time.perf_counter` around `train()`; inference latency = predict time
over the test set / rows at batch 256, and separately the mean of 200 batch-1 calls after a
warm-up. `test_metrics.py` checks every metric against hand-computed values on a 12-row
fixture, including an absent-class case and a collapsed-prediction case.

### R9. Results storage and resumability

- Master table: `results/results.jsonl`, one JSON object per run, appended atomically
  (write to temp, `os.replace`-free append with a single `write` of one line + flush). Row
  schema in `contracts/schemas.md`.
- Run key: `(config_id, split_set, window_id, seed)`. `phase.py` loads existing keys with
  status `ok`, `collapsed` or `class_absent` and skips them; rows with status `failed` are
  retried only with `--retry-failed`.
- Derived: `leaderboard.py` writes `results.csv` and `results.parquet` (pyarrow) and the
  aggregated `leaderboard_<phase>.csv`.
- **config_id** = first 12 hex of SHA-1 over the canonical JSON (sorted keys, no whitespace)
  of `{arch, hp, input_set, seq_len, split_set, train}`; `seeds`, `phase`, `tags` are
  excluded from the id. `test_config_id.py` asserts determinism and that two configs
  differing in any id field get different ids.

### R10. Compute calibration and the 40-hour budget

Planned runs at the R7 grids: phase 2 = 2 types × 3 windows × 3 seeds = 18 (minutes);
phase 3 = 24 × 3 × 2 = 144; phase 4 ≈ 31 configs × 3 × 2 = 186; phase 5 ≈ 6 × 3 × 2 = 36;
phase 6 ≈ 4 × 3 × 2 = 24; phase 7 ≈ 17 × 3 × 2 = 102 + third seeds for the top 3 (≈ 18).
About 530 neural runs; at the ~5 min/run implied by the existing 15-fold harness (113 min
including backtests) that is ~44 h, so the 40-epoch cap must bite. **Calibration task**:
run `LSTM` (production type) once per window at cap 40 and record minutes/run; if the
projection exceeds 40 h the phase-3 grid drops the ten LSTM ablation variants to 1 seed
first, then `train.stride=2` is applied to sweep phases only, both recorded on rows.

### R11. Input sets and indicator groups

- `ind24`: the production `include_list` (24 columns).
- `raw8` (all causal, computed in `data.py`): `ret1 = log(c_t/c_{t−1})`, `ret4`,
  `log(h/c)`, `log(l/c)`, `log(o/c)`, `range = (h−l)/c`, `logvol_z` (log volume z-scored on
  a trailing 96-bar window), `vol_ratio = log(v / mean_96(v))`. Scaled with the same
  per-window RobustScaler.
- `both32`: `ind24 ∪ raw8`.
- Groups for ablation (`ind24-minus:<group>`): trend {adx_scaled, aroonosc_scaled,
  ema_fast_norm}; volatility {atr_norm, bb_position, bb_width}; momentum {rsi_scaled,
  fastk_scaled, mfi_scaled, fisher_ss, cg_ss, gain_norm, di_diff_scaled}; oscillators
  {cci_scaled, macd_pos, macd_neg, macdhist_norm}; guards {guard_metric_pos,
  guard_metric_neg}; derived {fast_diff, sar_ratio, spread_ma, vwap_pos, vwap_neg}.
- **Label concern to record, not act on**: `guard_metric` and `bb_width` are inputs to the
  label trigger *and* model features; the guards/volatility ablations therefore measure "does
  the model need to see the trigger" rather than pure predictive value.

### R12. Learned importance and permutation importance

- Learned: `STG_GATE` registry type — a per-feature stochastic gate (Part B ref 26) applied
  on the feature axis before a backbone chosen by `hp.backbone` (default the best viable
  type), with the gate penalty added inside the model's forward so the shared loop is
  unchanged (the loss reads `model.extra_loss` if present — one added line in the training
  loop, defaulting to 0). Importance = expected gate value after training.
- Reference: permutation importance on the best saved model, per test window, 3 permutations
  per feature, drop in MCC / macro-F1 / balanced accuracy; group importance by permuting a
  whole group jointly. Correlated-feature caveat (ref 24) stated in the report.
- Agreement: Spearman ρ between STG importance and permutation-importance ranking, per
  window.
- iTransformer attention weights are logged as a free secondary signal, explicitly untrusted
  (refs 28, 29).

### R13. Phase-5 compositions and phase-6 shortlist (from Part B)

- Phase 5 (only from viable phase-4 families): `CNN_LSTM`, `CNN_TRANSFORMER`, `TCN_RES`
  (residual TCN), `RNN_ATTNPOOL` (attention pooling over RNN states), `INCEPTION_MS`
  (multi-scale kernels), plus "residual on/off" as an hp on `TCN_FAM`.
- Phase 6 implementations (ranked): `TSMIXER`, `ITRANSFORMER`, `XLSTM` (mLSTM block),
  `TRANSFORMER_MODERN` (RMSNorm + RoPE + SwiGLU on the existing block); `PATCHTST` only in
  the phase-7 window-length sweep at seq_len 128. Considered, not tested: S4/S5 (superseded
  by existing Mamba types), TimesNet (no valid period inside ≤128 bars), GQA (nothing to
  amortise), stochastic depth (stacks too shallow), InceptionTime as a phase-6 slot (covered
  by phase 5's multi-scale composition).

### R14. Training protocol amendment for phases 4–7 (2026-09-17)

- **Finding**: under the study default (lr 1e-4, cap 40, patience 10) convergence on windows B
  and C is a coin flip. Third-seed runs that reached the cap (best_epoch 30–39) scored MCC
  0.64 on B/C with near-identical prediction counts across LSTM variants, Mamba2 and Wavenet —
  the same ceiling every type reaches on window A — while runs that early-stopped at epochs
  12–17 (best_epoch 1–6) scored 0.45–0.55. Phase-4 stride-2 runs stalled 23 times in 49.
- **Calibration** (production LSTM, window C, 3 seeds): lr 3e-4 / patience 10 → 0.641, 0.640,
  0.640 (std 0.001); lr 1e-4 / patience 20 → 0.641, 0.640, 0.640; both need 31–39 epochs at
  full data (11–13.5 min/run). lr 3e-4 at **stride 2** → 0.640 × 3 (std 0.000, best epochs
  35–37, 5.5 min/run).
- **Decision**: phases 4–7 run at lr 3e-4, cap 40, patience 10 (FR-037 cap and patience
  unchanged), stride 2 for phases 4–6 and stride 1 for phase 7 finalists. Phase-3 rows stand
  as the "production protocol" reference and are not re-run; `train` is part of the config id,
  so the two protocols never collide. The report states the finding explicitly: architecture
  differences on B/C under the production protocol were mostly early-stopping variance, and any
  architecture that converges reaches the same ceiling.
- **Consequence for the stability clause**: with the amended protocol the 2-seed std is
  expected to be ≤ 0.01; the every-window clause is applied as written.

---

## Part B — Literature review (phase 6)


Scope: judged only against classification metrics on 16-to-128-bar, ~24-feature (or ~8 raw
OHLCV), 3-class, HOLD-dominated windows, trained on ~1M pooled windows across 11 pairs, in
`mlx.nn`. No P&L claims. "Already exists" means a variant is already registered in
`NNNC/NNNClassifierMLX.py::ClassifierTypeMLX` (LSTM, LSTM2, GRU, CNN, MLP, Attention,
Multihead, Transformer, Wavenet, Mamba, Mamba2, TSMamba, LSTM_KAN, plus LSTM structural
variants: InJit, Noisy, Ponder, W64, D10, Pool, K5, NoConv, Raw, NoResize, NoRes) and is
**not** re-proposed here.

A structural caveat that applies throughout: almost all of the forecasting-era literature
(PatchTST, iTransformer, TSMixer, TimesNet, DLinear) reports **forecasting** MSE/MAE on
datasets with hundreds-to-thousands of timesteps (ETT, Weather, Traffic, Electricity) at
horizons of 96-720. None of it is a classification result, and none of it is at seq_len
16-128. Where a paper's own evidence is forecasting-only or long-sequence-only, that is
flagged explicitly rather than implied to transfer.

---

## 1. Sequence / time-series classification architectures (2022-2026)

| # | Candidate | Applicable to this problem? | Expected benefit | MLX cost | Rank |
|---|---|---|---|---|---|
|1|**S4** (structured state-space) [3]|Partially — superseded in-repo by Mamba/Mamba2/TSMamba, which already exist [1,2]. S4's own headline evidence is Long-Range-Arena sequences of thousands-to-16384 steps; at seq_len ≤128 its main advantage (sub-quadratic long-range recall) is untested territory.|Low incremental — Mamba's selective SSM strictly generalises S4's LTI kernel [1].|N/A (superseded)|Not a candidate|
|2|**S5** (single MIMO SSM, parallel scan) [4]|Same family as above; no evidence at short multivariate windows either.|Low incremental over existing Mamba/Mamba2/TSMamba|N/A|Not a candidate|
|3|**xLSTM (sLSTM/mLSTM)** [5]|Yes, differs from existing LSTM/LSTM2/GRU: exponential gating + matrix-memory covariance update instead of the standard cell, aimed at fixing LSTM's inability to revise a stored decision. Evidence base is language modelling and long-horizon *forecasting* (xLSTM-Mixer [5b], not classification).|Med — plausible given LSTM already the strongest recurrent variant in-repo, but no classification result exists to cite, only forecasting/LM.|Med — no MLX primitive; custom cell of similar complexity to the Mamba/Mamba2 ports already done in-repo.|4|
|4|**PatchTST** (patch tokenization + channel-independence) [6]|Marginal at seq_len=16 — patch length ≥8-16 with stride leaves ≤2-4 patches per channel per window, degenerating the "long sequence of patches" mechanism the paper's gains depend on. Becomes more sensible only at seq_len=128 (8-16 patches). Channel-independence (one shared encoder per univariate channel) is also an odd fit for ~24 *engineered* features that are not comparable units.|Low at seq_len=16, unproven-but-plausible only at the top of the window sweep.|Med — patch embedding + existing Transformer/Multihead encoder in-repo covers most of it.|8 (test only at seq_len=128, not at 16)|
|5|**iTransformer** (feature-as-token, attention over channels not time) [7]|Yes — genuinely different axis from every existing transformer variant, which all tokenize by time-step. Inverts to ~24 tokens (one per feature, each embedding its own length-L series), so attention cost is independent of seq_len — relevant precisely because the window-length sweep goes to 128. Evidence is forecasting-only [7]; no classification benchmark exists.|Med — plausible mechanism advantage (attention over indicator relationships, not over 16-128 near-identical bars) but zero classification evidence to cite.|Med — reuses existing Multihead/Transformer attention block with the token axis swapped from time to feature; a genuinely different composition, not a re-implementation.|2|
|6|**TSMixer** (interleaved time-mixing / feature-mixing MLP) [8]|Yes — cheap, no existing variant mixes explicitly along both axes (MLP in-repo is a flat MLP with no per-axis structure). Evidence again is forecasting (M5, ETT-family) [8], not classification.|Med — TSMixer beat PatchTST/FEDformer/Autoformer on forecasting [8]; simple axis-mixing is a reasonable, cheap prior for engineered indicators that are already roughly independent per bar.|Low — pure `nn.Linear` + `nn.LayerNorm`, no exotic primitive.|1|
|7|**TimesNet** (reshape 1D→2D by detected period, 2D-conv) [9]|**No** — the mechanism requires seq_len ≫ dominant period. On 15m bars the shortest "seasonal" period worth detecting (daily = 96 bars) already exceeds the seq_len=128 ceiling of the sweep, so period folding either finds no valid period or folds noise. Reject.|N/A|N/A (rejected on mechanism grounds)|Rejected|
|8|**InceptionTime / multi-scale CNN** (parallel multi-length kernels per layer) [10]|Partially exists — the existing CNN and dilated-causal Wavenet variants already cover multi-receptive-field convolution; a literal Inception module (parallel 10/20/40-length kernels concatenated) is a different but closely-adjacent mechanism, with only marginal expected differentiation from what is already scored.|Low-Med — InceptionTime's UCR wins are on long, single-channel series (typically 100s-1000s of points) [10]; unclear it survives the drop to ~16-128, ~24-channel, HOLD-dominated windows.|Low — plain `mlx.nn.Conv1d` at multiple kernel sizes, concatenated.|6|
|9|**ROCKET / MiniRocket / MultiRocket** (random-kernel features + ridge/linear read-out) [11,12,13]|Yes, and already mandated: FR-013 requires a MiniRocket-style bar-to-beat baseline as a first-class non-neural comparator, independent of this ranking. Bake-off evidence [14] shows the ROCKET family remains competitive with or ahead of deep nets on UCR/UEA classification specifically (not forecasting), which is the strongest single piece of "simple beats deep" evidence in this whole review.|High as a *baseline*, not as something to rank against neural candidates — it already has a dedicated FR.|Low — this is just a random-kernel + ridge readout; MLX only needs the existing conv1d primitive and the study's own ridge classifier (FR-012).|Already required (FR-013), not re-ranked here|
|10|**"Are Transformers Effective?" / DLinear** (linear decomposition beats transformers on forecasting) [15]|Cautionary evidence, not a candidate: shows that transformer complexity does not reliably buy accuracy over a trivial linear model **on forecasting**. Directly relevant as a prior for why the ridge baseline (FR-012) and MiniRocket bar (FR-013) are the right gate, not a rubber stamp.|N/A (methodology note)|N/A|N/A|
|11|**UEA/UCR bake-off redux 2023** [14] and the original great MTSC bake-off [16]|Context, not a candidate: across both reviews, ROCKET-family and hybrid ensembles (HIVE-COTE2) remain the leaders on classification; pure deep nets (InceptionTime, ResNet, FCN) are competitive but rarely best-in-class, and no SSM/Mamba/patch-transformer appears in these classification bake-offs at all as of the reviewed date — the SSM/patch-transformer literature is essentially untested on short, small, tabular-like multivariate windows like this task's.|N/A|N/A|N/A|

**Shortlist to implement (architectures), ranked:**
1. **TSMixer** — cheapest, most directly applicable structural prior (time-mix + feature-mix), no MLX gap.
2. **iTransformer**-style feature-as-token attention — the one candidate that changes *what* is attended over, orthogonal to every existing transformer/attention variant.
3. **xLSTM (mLSTM block)** — most credible "better recurrent cell" not already tried, at the cost of a bespoke MLX port (comparable effort to the existing Mamba2 port).
4. **PatchTST**, but only at seq_len=128 in the window-length sweep, not at the production seq_len=16, since the mechanism needs enough patches to be meaningful.

**Considered, not implemented:**
- S4, S5 — superseded in-repo by Mamba/Mamba2/TSMamba [1,2]; no incremental evidence.
- TimesNet — mechanism requires seq_len ≫ period; rejected on structural grounds for this window range [9].
- InceptionTime-style parallel-kernel CNN — too close to existing CNN/Wavenet coverage for the added cost to be worth a dedicated slot [10].

---

## 2. LLM-era transformer internals (applied as composition changes to the existing Transformer/Multihead/Attention variants, not new architectures)

| # | Candidate | Applicable? | Expected benefit | MLX cost | Rank |
|---|---|---|---|---|---|
|12|**RMSNorm** (pre-norm, drop mean-centering) [17]|Yes — drop-in replacement for LayerNorm in the existing pre-LN Transformer. Evidence is LLM-scale (d_model in the hundreds-thousands, sequences of thousands of tokens) [17]; no published evidence at d_model≤128/seq_len≤128, so benefit is a plausibility argument (cheaper, comparable accuracy [17]), not a measured one at this scale.|Low-Med — mostly a compute/stability win, not expected to move accuracy much.|**Low** — `mlx.nn.RMSNorm` is a built-in layer.|1|
|13|**RoPE** (rotary position embedding) [18]|Yes — replaces the existing learned `TransformerPositionalEmbeddingMLX`. One classification-adjacent data point exists: TimeDRL ablates RoPE-style relative position against learned absolute embeddings across 5 UEA-style TS classification datasets and reports a net average accuracy gain when using its dual-embedding scheme with relative position information [19] — this is the closest thing to direct classification transfer evidence found in this review, though TimeDRL is a contrastive pretraining method, not a plain supervised classifier.|Med — some direct-ish evidence [19], though confounded with TimeDRL's other components.|**Low** — `mlx.nn.RoPE` is a built-in layer.|2|
|14|**SwiGLU FFN** (gated FFN replacing ReLU/GELU MLP) [20]|Yes — drop-in FFN swap inside the existing Transformer block. Evidence is LLM pretraining perplexity/downstream NLU [20], not time-series classification; transfer is a plausibility argument only.|Low-Med — consistent small gains in the source domain [20], unmeasured here.|Low — two `nn.Linear` projections + SiLU + elementwise product; no MLX gap.|3|
|15|**Grouped-Query Attention (GQA)** [21]|**No**, not for this task — GQA's entire benefit is reducing KV-cache memory/bandwidth during autoregressive multi-step decoding [21]; this study runs one forward pass per window with no KV cache and no generation loop, so the mechanism has nothing to amortise. At d_model≤128 with e.g. 8 heads, the head-count reduction it proposes would just shrink capacity for no inference-speed reason.|None expected|Low to implement, but pointless here|Rejected|
|16|**Stochastic depth / DropPath** [22]|Marginal — the existing Transformer/Multihead stacks are shallow (a handful of blocks) at this budget; DropPath's evidence base is very deep residual stacks (dozens-hundreds of layers) [22]. With few layers there are few "paths" to drop, so expected effect is small; existing `DROPOUT=0.25` already regularises.|Low — shallow stacks limit how much stochastic depth can do [22].|Low — a residual-branch Bernoulli mask.|5|

**Shortlist to implement (internals, as one combined "modernised Transformer" composition, per FR-030's composition step):** RMSNorm + RoPE + SwiGLU as a single composed variant of the existing Transformer type (cheapest three, both built into `mlx.nn` or trivial, and RoPE has the only classification-adjacent evidence found [19]). Test at both seq_len=16 and 128 since RoPE's relative-position benefit should scale with window length.

**Considered, not implemented:** GQA (mechanism doesn't apply without autoregressive decoding [21]); stochastic depth (stacks too shallow for the mechanism's own evidence base to bite [22]).

---

## 3. Feature-importance / learned feature selection (for FR-032's input-data study)

| # | Candidate | Applicable? | Expected benefit | MLX cost | Rank |
|---|---|---|---|---|---|
|17|**Permutation importance** (Breiman-style, reference method) [23]|Yes — required by FR-032 as the reference ranking against which any learned method is checked. Known failure mode: biased/inflated importance for **correlated** features, since permuting one feature while its correlated partners stay put creates unrealistic combinations the model never saw in training, misattributing shared signal [24]. A conditional variant exists to correct this but is markedly more expensive to compute [24].|N/A (reference method, not ranked)|N/A|Low — pure inference-time re-scoring, no training change.|Reference, not ranked|
|18|**Concrete Autoencoder** (differentiable top-k feature selection via Gumbel-softmax "concrete" selector layer) [25]|Yes, in spirit — but it's designed as an *unsupervised* reconstruction-based selector (pick k features to reconstruct all features), not a supervised classification-importance method; would need re-purposing (selector trained jointly with the classification head) to answer "which of ~24 indicators matter for the label," which is a modification beyond the published method [25].|Med — plausible once repurposed, unproven for this exact use.|Med — Gumbel-softmax relaxation + temperature annealing schedule; no MLX primitive, but straightforward.|2|
|19|**Stochastic Gates (STG)** [26]|Yes — directly supervised, per-feature Bernoulli-like gate with an L0-style penalty trained jointly with the task loss in one run, which is exactly FR-032's "learned single-run feature-importance" requirement. Evidence is regression/classification on tabular and genomics data [26], not time-series-window data specifically, but the mechanism (elementwise gate + penalty) is architecture-agnostic and applies unchanged to a flattened or per-channel window input.|Med-High — best mechanism fit to the stated requirement.|Low-Med — one extra elementwise gate layer + a differentiable penalty term added to the existing loss; no MLX primitive needed, straightforward custom layer.|1|
|20|**LassoNet** [27]|Yes, alternative to STG — enforces a hierarchy (a feature can only feed hidden units if its own linear skip-connection is active), giving a full regularisation path of sparsity levels in one training run [27]. More machinery than STG (proximal gradient step alongside standard SGD) for a similar deliverable.|Med — comparable goal to STG, no time-series-specific evidence [27].|Med — needs a custom proximal-gradient optimizer step, not just a standard Adam step; higher integration cost than STG inside the existing MLX training loop.|3|
|21|**Attention weights as importance** (feature-as-token attention, e.g. from iTransformer-style or Multihead over channels)|Partially — cheap to read off if a feature-as-token composition (candidate #5) is built anyway, but known unreliable as an importance measure: attention weights do not reliably track gradient-based importance and are not robust to small perturbations of the attention mechanism itself, undermining their use as an explanation in NLP settings [28], a finding contested but not resolved by a follow-up study [29]. Treat as a free by-product to sanity-check against permutation importance, never as the primary learned-importance method FR-032 asks for.|Low as a primary method; free as a secondary cross-check.|Zero marginal cost if the feature-as-token composition (candidate #5) is already built.|4 (secondary only)|

**Shortlist to implement (feature selection, one primary method as FR-032 requires):** **Stochastic Gates (STG)** [26] as the single learned single-run method, checked for rank-correlation against permutation importance [23,24] as required. LassoNet [27] and Concrete Autoencoder [25] are considered but not built given STG's lower integration cost for an equivalent deliverable. Attention weights [28,29] are recorded opportunistically if a feature-as-token model is trained, but are explicitly not trusted as ground truth.

---

## Considered, not recommended (all sections, single reference table)

| Candidate | Reason |
|---|---|
| S4 [3], S5 [4] | Superseded in-repo by Mamba/Mamba2/TSMamba [1,2]; own evidence is long-sequence (thousands of steps), not this task's 16-128. |
| TimesNet [9] | Mechanism requires seq_len ≫ dominant period; no valid period exists inside a 16-128-bar window at 15m resolution. |
| InceptionTime-style parallel-kernel CNN [10] | Materially overlaps existing CNN/Wavenet coverage; UCR evidence is on much longer single-channel series. |
| Grouped-Query Attention [21] | Mechanism (KV-cache amortisation over multi-step decoding) does not apply to single-pass classification. |
| Stochastic depth / DropPath [22] | Evidence base is very deep stacks; existing Transformer/Multihead depth is too shallow for the mechanism to plausibly matter. |
| Concrete Autoencoder [25] | Needs re-purposing from unsupervised reconstruction to supervised classification importance; STG achieves the same FR-032 deliverable more directly. |
| LassoNet [27] | Comparable goal to STG at higher optimizer-integration cost. |
| Attention-weight importance [28,29] | Documented unreliability as a stand-alone importance signal; kept only as a free secondary cross-check. |
| PatchTST at seq_len=16 [6] | Patching degenerates to ≤4 patches; only worth testing at the top of the window-length sweep (128). |

---

## References

1. Gu, A. & Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. https://arxiv.org/abs/2312.00752 (already in-repo: Mamba, Mamba2, TSMamba).
2. Dao, T. & Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* arXiv:2405.21060. https://arxiv.org/abs/2405.21060 (Mamba-2; already in-repo).
3. Gu, A., Goel, K. & Ré, C. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* arXiv:2111.00396. https://arxiv.org/abs/2111.00396
4. Smith, J.T.H., Warrington, A. & Linderman, S.W. *Simplified State Space Layers for Sequence Modeling (S5).* arXiv:2208.04933. https://arxiv.org/abs/2208.04933
5. Beck, M. et al. *xLSTM: Extended Long Short-Term Memory.* arXiv:2405.04517. https://arxiv.org/abs/2405.04517
5b. Kraus, M. et al. *xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories.* arXiv:2410.16928. https://arxiv.org/abs/2410.16928
6. Nie, Y. et al. *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* arXiv:2211.14730. https://arxiv.org/abs/2211.14730
7. Liu, Y. et al. *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* arXiv:2310.06625. https://arxiv.org/abs/2310.06625
8. Chen, S.-A. et al. *TSMixer: An All-MLP Architecture for Time Series Forecasting.* arXiv:2303.06053. https://arxiv.org/abs/2303.06053
9. Wu, H. et al. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* arXiv:2210.02186. https://arxiv.org/abs/2210.02186
10. Ismail Fawaz, H. et al. *InceptionTime: Finding AlexNet for Time Series Classification.* arXiv:1909.04939 / DOI:10.1007/s10618-020-00710-y. https://arxiv.org/abs/1909.04939
11. Dempster, A., Petitjean, F. & Webb, G.I. *ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels.* arXiv:1910.13051 / DOI:10.1007/s10618-020-00701-z. https://arxiv.org/abs/1910.13051
12. Dempster, A., Schmidt, D.F. & Webb, G.I. *MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification.* arXiv:2012.08791. https://arxiv.org/abs/2012.08791
13. Tan, C.W. et al. *MultiRocket: Multiple Pooling Operators and Transformations for Fast and Effective Time Series Classification.* arXiv:2102.00457. https://arxiv.org/abs/2102.00457
14. Middlehurst, M. et al. *Bake off redux: a review and experimental evaluation of recent time series classification algorithms.* arXiv:2304.13029. https://arxiv.org/abs/2304.13029
15. Zeng, A., Chen, M., Zhang, L. & Xu, Q. *Are Transformers Effective for Time Series Forecasting?* arXiv:2205.13504. https://arxiv.org/abs/2205.13504
16. Ruiz, A.P. et al. *The Great Multivariate Time Series Classification Bake Off.* DOI:10.1007/s10618-020-00727-3. https://dl.acm.org/doi/10.1007/s10618-020-00727-3
17. Zhang, B. & Sennrich, R. *Root Mean Square Layer Normalization.* arXiv:1910.07467. https://arxiv.org/abs/1910.07467
18. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864. https://arxiv.org/abs/2104.09864
19. Chang, C., Chan, C.-T., Wang, W.-Y., Peng, W.-C. & Chen, T.-F. *TimeDRL: Disentangled Representation Learning for Multivariate Time-Series.* arXiv:2312.04142. https://arxiv.org/abs/2312.04142 (RoPE-style relative-position ablation across 5 UEA-style classification datasets; contrastive-pretraining method, not a plain supervised classifier — evidence is adjacent, not direct).
20. Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202. https://arxiv.org/abs/2002.05202
21. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245. https://arxiv.org/abs/2305.13245
22. Huang, G., Sun, Y., Liu, Z., Sedra, D. & Weinberger, K.Q. *Deep Networks with Stochastic Depth.* arXiv:1603.09382. https://arxiv.org/abs/1603.09382
23. Breiman, L. *Random Forests.* Machine Learning 45, 5-32 (2001). DOI:10.1023/A:1010933404324.
24. Gregorutti, B., Michel, B. & Saint-Pierre, P. *Correlation and variable importance in random forests.* arXiv:1310.5726. https://arxiv.org/abs/1310.5726 (formalises the Strobl et al. 2008 conditional-permutation-importance critique of correlated-feature bias).
25. Balin, M.F., Abid, A. & Zou, J. *Concrete Autoencoders for Differentiable Feature Selection and Reconstruction.* arXiv:1901.09346. https://arxiv.org/abs/1901.09346
26. Yamada, Y., Lindenbaum, O., Negahban, S. & Kluger, Y. *Feature Selection using Stochastic Gates.* arXiv:1810.04247. https://arxiv.org/abs/1810.04247
27. Lemhadri, I., Ruan, F. & Tibshirani, R. *LassoNet: A Neural Network with Feature Sparsity.* arXiv:1907.12207. https://arxiv.org/abs/1907.12207
28. Jain, S. & Wallace, B.C. *Attention is not Explanation.* arXiv:1902.10186. https://arxiv.org/abs/1902.10186
29. Wiegreffe, S. & Pinter, Y. *Attention is not not Explanation.* arXiv:1908.04626. https://arxiv.org/abs/1908.04626
