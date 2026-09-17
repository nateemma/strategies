# Tasks: NNC Classifier Architecture & Input-Data Study

**Input**: Design documents from `specs/018-nnc-classifier-study/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Requested explicitly (FR-034). Harness tests precede every experiment task.

**Organization**: task **stages** 1–8 group the work; study **phases** 2–7 (baselines … input-data) keep their spec numbering inside the stages. Every task carries an
owner tag: **[MI]** model-implementer (Python code and tests), **[ER]** experiment-runner (runs,
leaderboards, plots), **[GR]** gate-reviewer (pre-gate leaderboard memo), **[MAIN]** main session
(gate decisions, checkpoints, report narrative). Paths are relative to `user_data/strategies/`.

## Format: `[ID] [P?] [Story] [Owner] Description with file path`

---

## Stage 1: Setup

- [X] T001 [MAIN] Create the package skeleton `research/clfstudy/{__init__.py,configs/,splits/,results/,gates/,report/plots/,tests/__init__.py,.cache/}` with `.gitkeep` files and a `.gitignore` entry for `research/clfstudy/.cache/`; `__init__.py` holds `HARNESS_VERSION = "0.1.0"`, `HORIZON = 48`, `EMBARGO_BARS = 96`, `VAL_FRACTION = 0.10`, `MIN_HISTORY_DAYS = 365`, `PAIRS` (the 11 production pairs) and the `sys.prefix` venv guard (pattern: `research/signalctx/__init__.py`)
- [X] T002 [P] [MI] Add the training-loop attributes to `Predictors/MLXClassifierNary.py`: class attrs `max_epochs=100`, `early_patience=20`, `weight_decay=0.0`; replace the two locals at lines ~342-343 with `self.max_epochs` / `self.early_patience`; use `optim.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)` only when `weight_decay > 0`; add `model.extra_loss` (if present) to the batch loss; set `self.epochs_run` and `self.best_epoch` before returning. No other behaviour change (contracts/registry.md)
- [X] T003 [P] [MI] Write `research/clfstudy/tests/test_training_attrs.py`: on a 300-row synthetic `(N,8,6)` tensor train `ClassifierTypeMLX.MLP` with `max_epochs=3` and assert `epochs_run == 3`; assert default attrs equal `(100, 20, 0.0)`; assert `weight_decay=1e-2` selects AdamW (inspect optimizer class via monkeypatch of `optim.AdamW`) and `0.0` selects Adam

---

## Stage 2: Foundational harness (User Story 1 prerequisites; blocks every experiment)

- [X] T004 [P] [MI] Implement `research/clfstudy/data.py`: `load_pair(pair) -> DataFrame` reading `../data/binanceus/<PAIR>_USDT-15m.feather`, filtering `date >= 2022-01-01`, `populate(df) -> DataFrame` calling `DataframePopulator().add_indicators(df, DatasetType.MINIMAL)`, and `raw8(df) -> DataFrame` adding the eight causal columns from research R11 (`ret1, ret4, log_hc, log_lc, log_oc, range, logvol_z, vol_ratio`)
- [X] T005 [P] [MI] Implement `research/clfstudy/labels.py`: `_LabelShim` whose attributes are read from the `NNNC.NNNC_MLX.NNNC_MLX` class object (`HORIZON, MIN_BUY_GAIN_THRESHOLD, MIN_SELL_LOSS_THRESHOLD, TRAINING_TYPE, LABEL_RVOL_THRESHOLD, LABEL_FILTER_WINDOW, apply_label_conflict_resolution`, `dbg_verbose=False`, no-op `debug_print`), binding `get_train_buy_signals/get_train_sell_signals/filter_peaks_by_future_performance/augment_training_signals/get_training_labels` from whichever class in `NNNC_MLX.__mro__` defines them; `ternary_labels(df) -> np.ndarray[int]` in SELL=0/HOLD=1/BUY=2 (research R2)
- [X] T006 [P] [MI] Write `research/clfstudy/tests/test_labels_shim.py`: assert every attribute the shim exposes equals the `NNNC_MLX` class attribute, and that the bound methods resolve to the same functions `NNNC_MLX` would use (compare `__func__`); on a 500-row synthetic OHLCV frame with a forced +3% run-up assert at least one BUY label and that labels are in {0,1,2}
- [X] T007 [P] [MI] Implement `research/clfstudy/splits.py`: `build(set_id="v1") -> SplitSet` with the three windows of research R3, `validate(split_set)` enforcing every invariant in data-model.md §SplitSet, `save/load` JSON to `research/clfstudy/splits/v1.json`, `assign_roles(dates, window) -> array of {train, embargo, val, test, none}`, `included_pairs(window, first_dates) -> list` using `MIN_HISTORY_DAYS`, and a `python -m research.clfstudy.splits build --set v1` entry point (contracts/clfstudy-cli.md)
- [X] T008 [P] [MI] Write `research/clfstudy/tests/test_splits.py`: windows disjoint and ordered; `train_end + 96 bars ≤ test_start`; val slice is the last 10% of the train span and is separated from train rows by 96 bars; `assign_roles` never gives a row two roles; a pair with 300 days of history is excluded and one with 400 is included; `validate` raises on a hand-broken window (spec FR-008/FR-009)
- [X] T009 [P] [MI] Implement `research/clfstudy/features.py`: `INDICATOR_GROUPS` dict (research R11), `columns_for(input_set) -> list[str]` handling `ind24 | raw8 | both32 | ind24-minus:<group>` using `FeatureNormalizer.include_list` and `pre_normalized_columns` for the needs-norm split, `fit_scaler(train_df, cols) -> RobustScaler` fitted on train rows only, `apply_scaler(df, scaler, cols)` with `clip(-10, 10)`, `class_weights(train_labels) -> list[3]` using the inverse-frequency formula of `TrainingEngine.get_training_class_weights` (research R4)
- [X] T010 [P] [MI] Implement `research/clfstudy/windows.py`: `make_tensors(pair_frames, labels, roles, cols, seq_len, stride) -> dict(train=(X,y1h), val=(X,y1h), test=(X,y,dates,pair))` building windows per pair on the continuous scaled series with `DataframeUtils.df_to_tensor(df, seq_len, method=3)` then selecting rows by role; train rows subsampled by `stride`; one-hot via `DataframeUtils.one_hot_encode(labels, 3)` (research R5)
- [X] T011 [MI] Write `research/clfstudy/tests/test_leakage.py` (depends on T009, T010): build a 2-pair synthetic dataset with a toy window; perturb every test-role and embargo-role bar (`×1000`, sign flip) and assert the fitted scaler `center_/scale_`, the class weights and the train/val tensors are bit-identical; assert the first test window's context rows are all dated before `test_start` and none is zero-padded (spec SC-003)
- [X] T012 [P] [MI] Implement `research/clfstudy/metrics.py`: `score(y_true, proba) -> dict` with every field of data-model.md §RunResult headline/per-class/probabilistic/matrix/secondary groups per research R8 (sklearn calls, own ECE with 15 bins and the reliability table, NaN for undefined per-class values, `collapsed` and `class_absent` detection), plus `timing(clf, X_test) -> dict` (batch-256 and batch-1 latency after warm-up)
- [X] T013 [P] [MI] Write `research/clfstudy/tests/test_metrics.py`: a 12-row fixture with hand-computed balanced accuracy, macro-F1, MCC, per-class precision/recall, log-loss, one-vs-rest AUROC, ECE and confusion matrix; an absent-class fixture (per-class NaN, `class_absent=True`, AUROC NaN for that class, macro = nanmean); a collapsed-prediction fixture (`collapsed=True`); BUY precision/coverage at 0.8 (spec FR-016..FR-025)
- [X] T014 [P] [MI] Implement `research/clfstudy/registry.py`: `resolve(arch) -> ClassifierTypeMLX`, `build(config, nf, seq_len, seed, train_cfg, checkpoint_dir) -> clf` doing `create_classifier_mlx(...)`, `clf.hp = config.hp`, `clf.train_seed = seed`, `clf.max_epochs/early_patience/weight_decay/batch_size/learning_rate` from `train_cfg`, `clf.checkpoint_path` under `research/clfstudy/.cache/ckpt/`; `param_count(clf) -> int` via `tree_flatten(model.trainable_parameters())`; `config_id(config) -> str` per research R9; `load_phase(path) -> list[ExperimentConfig]` with validation rules from data-model.md
- [X] T015 [P] [MI] Write `research/clfstudy/tests/test_config_id.py`: determinism across key order; `seeds/phase/tags` do not change the id; each id field changes it; invalid `arch`, `input_set`, `seq_len=1`, `stride=0` are rejected
- [X] T016 [P] [MI] Implement `research/clfstudy/baselines.py`: `majority(train_labels, n_test)` and `persistence(test_labels)` returning `(proba, note)` per research R6, routed through `metrics.score` unchanged
- [X] T017 [MI] Implement `research/clfstudy/run.py` (depends on T004–T016): `run_one(config, window, seed, save_weights=False) -> RunResult` executing load → populate → labels → roles → included pairs → scaler(train) → class weights(train) → tensors → build → `clf.train(Xtr, Xval, ytr, yval, class_weights=cw)` → `clf.predict(Xtest)` → `metrics.score` + `timing` → append one JSON line to `research/clfstudy/results/results.jsonl` (status `failed` with the traceback string on any exception); baseline pseudo-archs `MAJORITY`/`PERSISTENCE` bypass training; CLI `python -m research.clfstudy.run --config ... --window A|B|C|all --seed N [--save-weights] [--dry-run]` with the exit codes in contracts/clfstudy-cli.md; a per-pair populated/labelled frame cache under `research/clfstudy/.cache/frames/` keyed by `(pair, HARNESS_VERSION)`
- [X] T018 [MI] Implement `research/clfstudy/phase.py` (depends on T017): load `configs/phase_N.yaml`, expand `(config_id, window, seed)`, skip keys already present with status `ok|collapsed|class_absent`, retry `failed` only with `--retry-failed`, honour `--only` and `--max-runs`, print skipped/ran/failed counts
- [X] T019 [MI] Implement `research/clfstudy/leaderboard.py` (depends on T017): materialise `results.csv` and `results.parquet`; per-config per-window seed mean/std and pooled figures per data-model.md §Leaderboard; write `report/leaderboard_<phase>.{csv,md}`; plots with matplotlib into `report/plots/`: pooled MCC bar chart with seed-std error bars, MCC vs `n_params` scatter, per-window MCC heatmap, reliability diagrams for the top 5; every table header carries the Principle III note
- [X] T020 [MI] Implement `research/clfstudy/gate.py` (depends on T019): FR-028 with `margin_used = max(margin_floor, 2 × pooled seed-std of RIDGE MCC from phase-2 rows)`, per-config per-window clause results, `viable_set`; write `gates/phase_N.{json,md}` per contracts/schemas.md; CLI flags `--margin-floor --std-bound`
- [X] T021 [MI] Write `research/clfstudy/tests/test_gate.py`: synthetic results with a known ridge std → `margin_used` equals `max(0.02, 2×std)`; a config passing 2 of 3 windows is viable, 1 of 3 is not; a config with one `collapsed` window is never viable; a config with seed-std 0.031 on one window fails the stability clause
- [X] T022 [MI] Write `research/clfstudy/tests/test_registry_synthetic.py` (depends on T014): for EVERY member of `ClassifierTypeMLX` (parametrised over the enum so new members are picked up automatically) build, train 2 epochs on a `(400, 16, 24)` synthetic tensor with three classes, predict, and assert output shape `(N,3)`, rows sum to 1, and `param_count > 0`; for each `*_FAM` member assert parameter counts differ across its size axis
- [X] T023 [MI] Add `RIDGE` and `ROCKET` to `NNNC/NNNClassifierMLX.py` (classes `NNNClassifierMLX_Ridge`, `NNNClassifierMLX_Rocket` + enum members) per contracts/registry.md and research R6; ROCKET freezes its kernels with `model.freeze(...)` so only the head has trainable parameters; verify with T022
- [X] T024 [P] [MI] Write `scripts/check_label_fidelity.py`: load `NNNC_MLX` via `freqtrade.resolvers.StrategyResolver.load_strategy(config)` using `config/config.json` (dry-run mode, no exchange calls), run its `get_training_labels` on `data.populate(data.load_pair(pair))`, compare with `labels.ternary_labels` and print `IDENTICAL (n=…)` or the first 10 mismatching indices; exit 1 on mismatch
- [X] T025 [MI] Implement `research/clfstudy/calibrate.py` (depends on T017): run `LSTM` at `ind24/seq_len 16/train defaults` once per window with seed 0, print minutes per run and the projected totals per phase using the run counts in research R10
- [X] T026 [MAIN] Run `pytest research/clfstudy/tests -q` and `scripts/check_label_fidelity.py --pair LINK/USDT`; paste the fidelity output into `research/clfstudy/report/methods.md`; do not proceed while either fails

**Checkpoint**: harness green, labels verified identical to production, split set built.

---

## Stage 3: User Story 1 — Baselines and existing classifiers (Priority: P1) 🎯 MVP

**Goal**: first leaderboard: MAJORITY, PERSISTENCE, RIDGE, ROCKET and the 24 existing types.
**Independent Test**: `phase --phase 2` then `--phase 3` complete; `leaderboard --phase 3` shows every mandatory metric per window and pooled, `epochs_run ≤ 40` on every row.

- [X] T027 [P] [US1] [MI] Write `research/clfstudy/configs/phase_2.yaml`: MAJORITY, PERSISTENCE, RIDGE (`train.weight_decay 1e-2`), ROCKET, all `ind24/seq_len 16/split v1/seeds [0,1,2]`
- [X] T028 [P] [US1] [MI] Write `research/clfstudy/configs/phase_3.yaml`: one entry per existing `ClassifierTypeMLX` member (the 24 pre-study members, `hp: {}`), `seeds [0,1]`, `train` defaults (`max_epochs 40, patience 10`)
- [X] T029 [US1] [ER] Run `python -m research.clfstudy.calibrate --arch LSTM`; record minutes/run and the projection in `research/clfstudy/report/calibration.md`; if the projected total exceeds 40 h apply the R10 cuts in order (LSTM ablation variants to 1 seed, then `stride 2` on sweep phases) and note them there
- [X] T030 [US1] [ER] Run `python -m research.clfstudy.phase --phase 2`, then `--phase 3`; on completion run `leaderboard --phase 2` and `--phase 3`; report skipped/ran/failed counts and any `failed`/`collapsed` rows with their errors
- [X] T031 [US1] [GR] Review `report/leaderboard_3.md`: the baseline-vs-existing gap on each window, seed-std of RIDGE, any collapsed or class-absent rows, and whether the metrics look sane (e.g. MAJORITY balanced accuracy ≈ 0.33, PERSISTENCE MCC high); write `gates/review_3.md`
- [X] T032 [US1] [MAIN] ⏸ Show the first results table and the review to the author; wait for sanity-check approval before any sweep

---

## Stage 4: User Story 2 — Family sweeps and compositions (Priority: P2)

**Goal**: one-axis-at-a-time sweeps per family; gate; compositions from viable families only.
**Independent Test**: `gates/phase_4.json` exists with `viable_set`; no phase-5 config references a non-viable family; every sweep row has `n_params` and timing.

- [ ] T033 [US2] [MAIN] Evaluate gate 3: run `gate --phase 3`; record which existing types are viable (they stay in the finalist pool for phases 5–7) and the `margin_used` in `gates/phase_3.md`; phase 4 sweeps all four families unconditionally because it defines them, and gate 3 informs their anchors only
- [X] T034 [P] [US2] [MI] Add `LSTM_FAM`, `GRU_FAM`, `TCN_FAM`, `TRANSFORMER_FAM`, `MLP_FAM` to `NNNC/NNNClassifierMLX.py` per contracts/registry.md (pure families: no conv front-end, no residual unless `hp.residual`); `TRANSFORMER_FAM.pos` supports `learned|none|rope` (`mlx.nn.RoPE`); verify with T022
- [ ] T035 [US2] [MI] Write `research/clfstudy/configs/phase_4.yaml` from the research R7 anchors and axes (~31 configs, `seeds [0,1]`, tags naming the axis)
- [ ] T036 [US2] [ER] Run `phase --phase 4`; `leaderboard --phase 4` including the MCC-vs-params scatter and per-axis line plots (`report/plots/sweep_<family>_<axis>.png`)
- [ ] T037 [US2] [GR] Review `report/leaderboard_4.md`: per family, which axis moves MCC beyond seed noise, best size, cost/benefit; proposed viable families; write `gates/review_4.md`
- [ ] T038 [US2] [MAIN] Evaluate gate 4: run `gate --phase 4`; decide the viable family set from the gate output and the review; record in `gates/phase_4.md`; ⏸ show leaderboard and proposed viable set; wait
- [ ] T039 [US2] [MI] Add `CNN_LSTM`, `CNN_TRANSFORMER`, `TCN_RES`, `RNN_ATTNPOOL`, `INCEPTION_MS` to `NNNC/NNNClassifierMLX.py` per contracts/registry.md (only those whose component families were viable in T038; the others are still added with defaults for FR-015 but not run); verify with T022
- [ ] T040 [US2] [MI] Write `research/clfstudy/configs/phase_5.yaml`: compositions built only from viable families at their best phase-4 sizes, `seeds [0,1]`, `gate_from: 4`
- [ ] T041 [US2] [ER] Run `phase --phase 5`; `leaderboard --phase 5` with paired bars "composition vs its best component"
- [ ] T042 [US2] [GR] Review `report/leaderboard_5.md`: does any composition beat its best component beyond seed noise on a majority of windows; write `gates/review_5.md`
- [ ] T043 [US2] [MAIN] Evaluate gate 5: `gate --phase 5`; record the viable set entering phases 6/7 in `gates/phase_5.md`; ⏸ show and wait

---

## Stage 5: User Story 3 — Literature-derived candidates (Priority: P3)

**Goal**: implement and score the research Part B shortlist on the same splits and seeds.
**Independent Test**: each shortlisted candidate has rows on all three windows; research.md Part B has a citation, applicability verdict and rank for every listed candidate.

- [ ] T044 [P] [US3] [MI] Add `TSMIXER` to `NNNC/NNNClassifierMLX.py` (time-mix + feature-mix MLP blocks, LayerNorm, mean pool) per contracts/registry.md; verify with T022
- [ ] T045 [P] [US3] [MI] Add `ITRANSFORMER` (tokens = features, each embedding its length-`seq_len` series; encoder from the existing pre-LN block; store mean attention weights on `model.last_attention` for the secondary importance signal); verify with T022
- [ ] T046 [P] [US3] [MI] Add `XLSTM` (mLSTM block: exponential gating, matrix memory, `heads`; port pattern follows the existing Mamba2 class); verify with T022
- [ ] T047 [P] [US3] [MI] Add `TRANSFORMER_MODERN` (existing Transformer block with `mlx.nn.RMSNorm` pre-norm, `mlx.nn.RoPE`, SwiGLU FFN); verify with T022
- [ ] T048 [P] [US3] [MI] Add `PATCHTST` (patch embedding, refuses `seq_len < 4·patch_len` with a clear error); verify with T022 using `seq_len 64` in its parametrisation
- [ ] T049 [US3] [MI] Write `research/clfstudy/configs/phase_6.yaml`: TSMIXER, ITRANSFORMER, XLSTM, TRANSFORMER_MODERN at sizes matched to the best phase-4 transformer/RNN anchors, `ind24/seq_len 16/seeds [0,1]`, `gate_from: 5`; PATCHTST deferred to phase 7d
- [ ] T050 [US3] [ER] Run `phase --phase 6`; `leaderboard --phase 6` with a bar chart against the best phase-4/5 configs
- [ ] T051 [US3] [GR] Review `report/leaderboard_6.md`: which candidates beat their nearest incumbent beyond seed noise; write `gates/review_6.md`
- [ ] T052 [US3] [MAIN] Evaluate gate 6: `gate --phase 6`; record the viable set entering phase 7 and the "considered, not tested" list in `gates/phase_6.md`; ⏸ show and wait

---

## Stage 6: User Story 4 — Input-data study (Priority: P3)

**Goal**: raw vs engineered, group ablation, permutation vs learned importance, window length.
**Independent Test**: report tables 7a–7d exist for every viable architecture; the Spearman ρ between STG and permutation importance is reported per window.

- [ ] T053 [P] [US4] [MI] Add `STG_GATE` to `NNNC/NNNClassifierMLX.py` (per-feature Gaussian-relaxed Bernoulli gate on the feature axis, penalty via `model.extra_loss`, `gate_importance() -> np.ndarray`, backbone chosen by `hp.backbone`); verify with T022 (backbone `MLP_FAM`)
- [ ] T054 [P] [US4] [MI] Implement `research/clfstudy/importance.py`: load a saved run's weights (`weights_path`) and its window tensors, permutation importance per feature and per group (3 permutations, drop in MCC/macro-F1/balanced accuracy), STG read-out for `STG_GATE` runs, Spearman ρ between the two rankings; CLI per contracts/clfstudy-cli.md; write `report/importance_<id>_<W>.csv`
- [ ] T055 [US4] [MI] Write `research/clfstudy/tests/test_importance.py`: on a synthetic dataset where only feature 0 carries signal, permutation importance ranks feature 0 first and a 2-epoch `STG_GATE` on `MLP_FAM` gives feature 0 the largest gate
- [ ] T056 [US4] [MI] Write `research/clfstudy/configs/phase_7.yaml` with sub-phases tagged `7a` (input sets `raw8`, `both32` for each viable architecture, plus `raw8` with a `TCN_FAM` front-end config), `7b` (`ind24-minus:<group>` for the six groups on the best viable architecture), `7c` (`STG_GATE` on the best viable backbone, `ind24`, `--save-weights`), `7d` (window lengths 32, 64, 128 for the top-2 architectures, PATCHTST at 128); `seeds [0,1]`; the best config of 7a/7b marked `--save-weights`
- [ ] T057 [US4] [ER] Run `phase --phase 7` (all sub-phases); then `importance --config-id <best> --window A|B|C --group`; `leaderboard --phase 7` with the window-length curve plot (joining the seq_len-16 rows of the same arch/hp from phases 4–6 to the 32/64/128 rows) and the importance bar charts
- [ ] T058 [US4] [ER] Add a third seed (`seeds [0,1,2]`) for the top-3 pooled-MCC configs of phase 7 and re-run `leaderboard --phase 7` (FR-004 finalists)
- [ ] T059 [US4] [GR] Review the 7a–7d tables and importance CSVs: raw-vs-engineered verdict, group-ablation verdict, agreement ρ, window-length sensitivity; write `gates/review_7.md`
- [ ] T060 [US4] [MAIN] Evaluate gate 7: `gate --phase 7`; record the final recommended architecture set and input set in `gates/phase_7.md`; ⏸ show and wait

---

## Stage 7: User Story 5 — Results report (Priority: P2, produced last)

**Goal**: `research/clfstudy/report/report.md` with every section of data-model.md §Report.
**Independent Test**: every leaderboard number in the report traces to `results.jsonl` by `(config_id, window_id, seed)`; SC-009 text search passes.

- [ ] T061 [US5] [MI] Implement `research/clfstudy/report.py`: assemble `report/report.md` from `methods.md`, `calibration.md`, `leaderboard_*.md`, `gates/phase_*.md`, `gates/review_*.md`, importance CSVs and plots; include the Principle III header note, the label-concern paragraph (research R11) and a reproduction-commands appendix
- [ ] T062 [US5] [ER] Run `python -m research.clfstudy.report`; run `grep -rniE "backtest|p&l|pnl|drawdown|trade" research/clfstudy/report/ research/clfstudy/*.py` and confirm hits are only the scope statement and label-concern notes (SC-009)
- [ ] T063 [US5] [MAIN] Write the narrative sections of `report/report.md`: per-phase findings, recommendation (as candidates for a later P&L evaluation), label concerns, open questions; cross-check three leaderboard numbers by hand against `results.jsonl`

---

## Stage 8: Polish & cross-cutting

- [ ] T064 [P] [MI] Confirm FR-015 / SC-005: `research/clfstudy/tests/test_strategy_selection.py` instantiates `create_classifier_mlx(ClassifierTypeMLX.RIDGE, "X/USDT", 24, 16, 3)` and every new member with `hp` unset and asserts `create_model` succeeds with defaults; additionally calls `MLXClassifierMixin.get_classifier` unbound on a shim object whose `classifier_type = ClassifierTypeMLX.RIDGE` (the exact attribute a strategy sets) and asserts a RIDGE classifier is returned
- [ ] T065 [P] [MAIN] Update `AGENT_GUIDE.md` with a short "Classifier study harness" section (commands, where results live, the three training-loop attributes)
- [ ] T066 [MAIN] Commit in logical units to master with explicit `git add <paths>` only (never `-A`): (1) training-loop attrs + test, (2) harness + tests, (3) registry members, (4) configs/results/gates/report; never stage `NNNC/NNNC_CGP_MLX.json` (the author's in-progress deletion) or any other path the change did not touch; each commit message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`

---

## Dependencies

- Stage 1 → Stage 2 → US1 → US2 → US3 → US4 → US5. The study is strictly sequential at the
  phase-gate level; parallelism exists only inside phases.
- T022 (registry synthetic test) is re-run after every task that adds registry members
  (T023, T034, T039, T044–T048, T053).
- Gate tasks (T033, T038, T043, T052, T060) block the next phase's config task.

## Parallel execution examples

- Stage 2: T004, T005, T007, T009, T010, T012, T014, T016 are independent files → run together;
  their tests (T006, T008, T013, T015) likewise.
- US3: T044–T048 are five independent classes in one file — run sequentially per file but
  they can be authored as separate patches.

## Implementation strategy

MVP = Stage 1 + Stage 2 + US1 (first leaderboard, checkpoint T032). Nothing beyond US1 is
started until the author has sanity-checked metrics and the baseline-vs-existing gap.
