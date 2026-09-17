# Implementation Plan: NNC Classifier Architecture & Input-Data Study

**Branch**: `018-nnc-classifier-study` (spec directory; commits go to master) | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/018-nnc-classifier-study/spec.md`

## Summary

Build one evaluation harness, `research/clfstudy/`, that drives the **existing**
`NNNClassifierMLX` build → `train()` → `predict()` path outside any freqtrade strategy, on
leakage-safe walk-forward splits of the 11 production pairs at 15m, and scores every
configuration with the same metric code. New classifier types (Ridge, ROCKET-style, the
parametrised single-family types, compositions, and the literature shortlist) are added to
`ClassifierTypeMLX` in the existing pattern so they are usable by strategies unchanged.

Five measurements taken during planning decide the shape of this plan:

1. **The production training path has no embargo.** `TrainingEngine.prepare_training_data`
   splits each pair 80/20 in time order with only a `seq_len-1` buffer (lines 155-163), and
   the production RobustScaler is fitted on the whole training history. The harness must
   therefore own its split, scaler and class weights (Constitution XI), and must NOT call
   `scale_dataframe` or `prepare_training_data`.
2. **The epoch cap and patience are locals inside `MLXClassifierNary.train()`**
   (`max_epochs = 100`, `early_patience = 20`, lines 342-343). FR-037 (cap 40 / patience 10)
   needs a small, behaviour-preserving edit: read them from instance attributes whose
   defaults equal today's literals. Same for weight decay (needed for a true ridge penalty).
3. **`train()` computes class weights from the VALIDATION labels when none are passed**
   (`calculate_class_weights(test_results_np)`, line 316). The harness always passes
   class weights computed from the training span.
4. **Windows are causal and label-aligned to the last bar.** `df_to_tensor(method=3)` makes
   one window per row, zero-padded at the series start; labels are one-hot per row. Building
   windows on each pair's continuous series and then selecting rows by date gives correct
   context at the start of every evaluation window with no padding.
5. **Labels can be reproduced without instantiating a strategy.** `get_training_labels`,
   `get_train_buy_signals`, `filter_peaks_by_future_performance` and
   `augment_training_signals` only read class attributes (`HORIZON`, `MIN_*`,
   `TRAINING_TYPE`, `LABEL_RVOL_THRESHOLD`, `LABEL_FILTER_WINDOW`,
   `apply_label_conflict_resolution`). A shim carrying those attributes read off the
   `NNNC_MLX` class object reproduces production labels; fidelity is verified once against a
   `StrategyResolver`-loaded `NNNC_MLX` (Constitution IV: audit the instrument first).

## Technical Context

**Language/Version**: Python 3.12.13 at `/Users/philprice95/projects/freqtrade/.venv/bin/python`
(guard on `sys.prefix`, never on a resolved `sys.executable`; recorded 2026-09-15).

**Primary Dependencies**: mlx 0.32.2 (`mlx.nn` has `RMSNorm`, `RoPE`, `MultiHeadAttention`,
`Conv1d`, `LSTM`, `GRU`; `mlx.optimizers.AdamW` exists), numpy, pandas, scikit-learn 1.9
(metrics only), pyarrow 25 (parquet export), matplotlib 3.11 (plots), numba 0.67 (available
but not required). **No sktime/aeon** — the ROCKET-style baseline is implemented in-house.

**Storage**: append-only `research/clfstudy/results/results.jsonl` (one row per run, crash-safe
append), with `results.csv` / `results.parquet` materialised by the leaderboard command.
Per-pair populated/labelled frame cache and per-run checkpoints live in
`research/clfstudy/.cache/` (gitignored, under `user_data/`, survives across sessions so
resumption never re-populates).
Split sets under `research/clfstudy/splits/`, gate decisions under `research/clfstudy/gates/`,
report + plots under `research/clfstudy/report/`. Weights are NOT kept except for runs started
with `--save-weights` (needed for permutation importance on the best model).

**Testing**: pytest from the repo root, `research/clfstudy/tests/` (clarified 2026-09-16).

**Target Platform**: Apple silicon, MLX/Metal. Sequential runs only.

**Project Type**: research module inside `user_data/strategies`.

**Performance Goals**: about 40 machine-hours for the whole study (clarified). A calibration
task measures one run per window before any phase is launched and the phase grids are cut to
fit; a `train.stride` knob (uniform time-stride subsampling of training windows) is the
contingency, default 1 (off), always recorded on the row.

**Constraints**: all code under `user_data/strategies/`; the study never reads
`saved_data/*/main_scaler`; classifier behaviour for existing types must be byte-identical to
today's when the new attributes are left at their defaults.

**Scale/Scope**: 11 pairs, ~1.69M bars since 2022-01-01, 3 evaluation windows, ~105
configurations, ~600 training runs.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1.*

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Deployment regime** | PASS | Window C is 2026-H1, the most recent half-year; per-window results are always shown beside the pooled figure (FR-026). |
| **II. Persistence (NON-NEGOTIABLE)** | PASS | Window A (2023-H2) is temporally distant from B (2025-H1) and C (2026-H1). Viability requires a majority of windows, and the report shows every window separately. |
| **III. Learnability ≠ edge** | **PASS by construction** | This feature is the learnability screen; it never promotes anything. The Scope Statement and FR-036 forbid P&L; the report's recommendation section is explicitly "candidates for a later P&L evaluation". |
| **IV. Parameters actually live** | PASS | Epoch cap and patience are proven live by `epochs_run ≤ max_epochs` on every row and a test that the cap changes `epochs_run` on synthetic data. Hyperparameter injection is proven live by parameter-count differences across sweep configs (test). Label reproduction is verified against a StrategyResolver-loaded strategy before any result is recorded. |
| **V. Lookahead ≠ realistic** | N/A, stated | No execution is simulated. Lookahead on the *feature/label* side is covered by XI. |
| **VI. Diagnose before tuning** | PASS | Every row carries predicted-class counts, collapse and class-absent flags, reliability bins and the confusion matrix, so a bad number is localisable before any grid is changed. |
| **VII. Validate cheaply** | PASS | Calibration run before any phase; phases 2-3 before any sweep; each later phase runs only on the viable set. |
| **VIII. One parameter space** | PASS | Phase-4 grids vary one axis at a time around a family anchor (size axis, then one structural axis), not a joint grid. |
| **IX. Orthogonality** | PASS (input study) | Permutation importance is cross-checked against a learned gate precisely because correlated indicators inflate permutation importance (research R12). |
| **X. Structural, not per-pair** | PASS | Pooled training only; no per-pair configuration anywhere. |
| **XI. No data leakage** | PASS | Full control list in research R3/R4 and `data-model.md` §Invariants; tests T-LEAK-1..4. |
| **Repo constraints** | PASS | Everything under `user_data/strategies/`; `.venv` guard; new types are `ClassifierTypeMLX` members with `create_model` overrides; `Framework/` untouched (the training-loop edit is in `Predictors/`, which is backend code, not strategy-specific); commits to master, explicit staging. |

### Principle III — how the report will be worded

Every leaderboard table carries the header note "classification metrics against fixed labels;
not evidence of trading edge". The recommendation section names architectures and an input set
as **candidates for the next, separate, P&L-based evaluation** and nothing more. The
label-concern section records, without acting on it, that `guard_metric` and `bb_width` are both
label-trigger inputs and model features (research R11), so their importance is partly the label
seeing itself.

### Principle IV — the two instruments audited before results exist

- **Label shim vs production**: `scripts/check_label_fidelity.py` loads `NNNC_MLX` through
  freqtrade's `StrategyResolver` with `config/config.json`, runs its `get_training_labels` on one
  pair, and asserts the array is identical to the harness's shim output. Runs once; its output is
  pasted into the report's Methods section.
- **Training-loop attributes**: `test_epoch_cap_is_live` trains a tiny MLP with `max_epochs=3`
  and asserts `epochs_run == 3`, then with the default and asserts it differs.

## Project Structure

### Documentation (this feature)

```text
specs/018-nnc-classifier-study/
├── plan.md              # This file
├── research.md          # Phase 0 — R1..R13 + Part B literature review (29 refs)
├── data-model.md        # Phase 1 — entities, schemas, invariants
├── quickstart.md        # Phase 1 — runnable validation sequence
├── contracts/
│   ├── clfstudy-cli.md  # command surface
│   ├── schemas.md       # ExperimentConfig, SplitSet, RunResult, GateDecision
│   └── registry.md      # new ClassifierTypeMLX members, hp keys, training-loop attributes
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks output — NOT created here
```

### Source Code (repository root = `user_data/strategies`)

```text
research/clfstudy/
├── __init__.py          # interpreter guard; HARNESS_VERSION; pinned constants (EMBARGO_BARS=96, HORIZON=48)
├── data.py              # load feathers for the 11 pairs; add_indicators(MINIMAL); raw8 features
├── labels.py            # production-identical ternary labels via the class-attribute shim (R2)
├── splits.py            # SplitSet v1 builder + validator; JSON persisted under splits/
├── features.py          # input sets (ind24 / raw8 / both32 / ind24-minus:<group>), group map,
│                        #   per-window RobustScaler fit-on-train (R3/R4)
├── windows.py           # per-pair continuous windowing via DataframeUtils.df_to_tensor, row selection by date
├── metrics.py           # FR-016..FR-025 (sklearn + own ECE)
├── baselines.py         # majority, persistence (no training)
├── registry.py          # config.arch → ClassifierTypeMLX member; hp injection; param count
├── run.py               # single experiment: build → train → predict → score → append row
├── phase.py             # run a phase file; resumable by (config_id, window_id, seed)
├── leaderboard.py       # jsonl → csv/parquet; per-window and pooled tables; plots
├── gate.py              # viability rule with adaptive margin; writes gates/phase_N.json + .md
├── importance.py        # permutation importance on a saved model; STG gate read-out
├── report.py            # assembles report/report.md from leaderboard + gates
├── configs/             # phase_2.yaml … phase_7.yaml (lists of ExperimentConfig)
├── .cache/              # gitignored: per-pair frame cache, per-run checkpoints
├── splits/v1.json
├── results/results.jsonl (+ results.csv, results.parquet derived)
├── gates/
├── report/{report.md, plots/}
└── tests/
    ├── test_splits.py           # no overlap, embargo ≥ 48, val slice geometry
    ├── test_leakage.py          # scaler/class-weight invariance to test-bar perturbation
    ├── test_metrics.py          # against hand-computed values incl. ECE and undefined-class cases
    ├── test_registry_synthetic.py # every ClassifierTypeMLX member builds/trains/scores on synthetic data
    ├── test_config_id.py        # id determinism and collision rules
    ├── test_training_attrs.py   # epoch cap / patience / weight decay are live
    └── test_labels_shim.py      # shim attribute set equals NNNC_MLX class attributes

scripts/
└── check_label_fidelity.py     # one-off: shim labels == StrategyResolver-loaded NNNC_MLX labels

NNNC/NNNClassifierMLX.py         # + RIDGE, ROCKET, LSTM_FAM, GRU_FAM, TCN_FAM, TRANSFORMER_FAM, MLP_FAM,
                                 #   CNN_LSTM, CNN_TRANSFORMER, RNN_ATTNPOOL, TCN_RES, INCEPTION_MS,
                                 #   TSMIXER, ITRANSFORMER, XLSTM, TRANSFORMER_MODERN, PATCHTST, STG_GATE
Predictors/MLXClassifierNary.py  # + instance attrs max_epochs=100, early_patience=20, weight_decay=0.0;
                                 #   AdamW when weight_decay>0; epochs_run/best_epoch recorded on self
```

**Structure Decision**: a new `research/clfstudy/` package, following `research/labelfree/` and
`research/signalctx/`. Existing instruments are reused, not reimplemented: `DataframePopulator
.add_indicators`, `FeatureNormalizer.include_list` / `pre_normalized_columns` (column
selection only, never its fitted scaler), `DataframeUtils.df_to_tensor` and `one_hot_encode`,
`BaseNNStrategy`'s label methods via the shim, and the whole `MLXClassifierNary.train/predict`
loop. `evaluation/foldplan.py` is NOT reused: it has no embargo or validation slice and emits
freqtrade timeranges for backtests; `splits.py` is ~80 lines and is unit-tested.

## Complexity Tracking

| Violation / deviation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Edit to the shared training loop `Predictors/MLXClassifierNary.py` (3 attributes + AdamW branch + `epochs_run` bookkeeping) | FR-037 epoch cap, a real ridge penalty, and recording epochs on every row | Subclassing and overriding `train()` would duplicate a ~300-line method, breaking "identical path" (FR-012) and the repo's recorded failure mode of duplicated loss logic drifting. Defaults equal the current literals, so existing behaviour is unchanged. |
| Labels via unbound `BaseNNStrategy` methods on a shim object | Production-identical labels without a live freqtrade strategy | Instantiating `NNNC_MLX` needs a full freqtrade config and pollutes class-level `Parameter` singletons for the process; the shim is verified once against the real strategy (`check_label_fidelity.py`) and its attribute set is unit-tested against the class. |
| Parametrised "family" types (`*_FAM`) read hyperparameters from an instance `hp` dict | Phase-4 sweeps need dozens of sizes without dozens of enum members | One enum member per size would bloat the registry; `hp` is set after the factory call and before the lazy `create_model`, and each member's defaults are a valid strategy-usable configuration (FR-015). |
