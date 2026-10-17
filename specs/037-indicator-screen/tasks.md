---
description: "Task list for 037 indicator screen"
---

# Tasks: Indicator Screen Against Labels and Regimes

**Input**: `specs/037-indicator-screen/` (plan.md, spec.md, research.md R1–R15, data-model.md,
contracts/cli.md, quickstart.md)

**Tests**: included. Repo practice: every research module ships with pytest tests, and the screens'
correctness is the feature (Constitution IV, SC-007).

**Paths**: relative to the nested repo `user_data/strategies/`. The cache lives at
`user_data/data/indicator_screen/` (outside the repo).

**Run rules for every run task:**
- Run from `user_data/strategies` with `set -a; source ~/.freqtrade_env; set +a` and `PYTHONPATH=.`.
- Check `ps` first: no training or backtest chain is running.
- Never touch the live NNNC bot (`scripts/live_nnnc.sh status`) or pid 2661.
- After each run task, confirm both bots are alive (SC-006).

## Phase 1: Setup

- [X] T001 Create the package `research/indicator_screen/__init__.py` and `research/indicator_screen/tests/__init__.py`, and add `research/indicator_screen/tests` to the full-suite command in `AGENT_GUIDE.md`.
- [X] T002 Write `research/indicator_screen/preregistration.json` as a DRAFT (not yet committed as the lock) with the data-model.md fields:
  - `min_rows`, `min_fires`, `max_fire_rate` 0.95;
  - instrument `{top_k: 5, columns: [guard_metric, bb_width]}`;
  - pairs `{min_windows: 6, uplift: 1.10, min_lift: 1.20, top_n: 20}`;
  - models (R10 settings verbatim);
  - regime `{horizons: [48, 288, 672], a2_cuts: "early_test", populations: ["all", "candidate"], model_subsample: 4, a2_model_train: "prior_test_windows", dedup_market_wide: true}`;
  - `e0_required` `{single: "same_direction", pairs: "lift_ge_max_component"}` and `instrument_gate_halves` `["e0", "early"]` (Principle II, SC-002);
  - `residual_grid` 5, `mi_bins` 20, `redundancy_cut` 0.8, `sample_rows` 200000, `seed` 0.

## Phase 2: Foundational (blocks all stories)

- [X] T003 [P] Implement `research/indicator_screen/windows.py`:
  - the 15 folds from `evaluation/folds/plan_wf_aro_g24.json`, giving test/train spans and half early (f00–f07) / late (f08–f14);
  - the era `E0` = 2021-04-01 → 2022-12-31 (no train span);
  - `purge(target)` (48 bars for labels, N for regime targets);
  - `train_rows_mask(dates, fold, target)` (`date < test_start − purge`, after the 1,000-bar warm-up);
  - `late_lock()`: refuses unless `preregistration.json` is committed in git with no working-tree diff (R15).
- [X] T004 [P] Tests for T003 in `research/indicator_screen/tests/test_windows.py`:
  - fold spans match the plan;
  - E0 has no train span;
  - purge excludes the last N bars before test_start;
  - `late_lock` refuses for uncommitted, modified and missing files (using a temporary git repo).
- [X] T005 [P] Implement `research/indicator_screen/scores.py`:
  - rank AUC (Mann–Whitney, ties averaged), direction, strength;
  - binned normalised MI, given bin edges (20 quantile edges supplied by the caller);
  - lift and fire rate, with `min_fires` / `max_fire_rate` → not computable;
  - stratified AUC on a 5 × 5 grid with n_pos × n_neg weights (R7);
  - window aggregation: median, same-direction share, `min_rows` exclusion, halves e0/early/late (R6).
- [X] T006 [P] Tests for T005 in `research/indicator_screen/tests/test_scores.py`:
  - AUC equals `sklearn.metrics.roc_auc_score` on random data;
  - a perfectly separating column gives AUC 1 and MI_norm 1;
  - an independent column gives MI ≈ 0;
  - the lift and not-computable cases;
  - stratified AUC ≈ 0.5 for the stratifying column itself, and equals plain AUC with a single cell;
  - aggregation drops windows under `min_rows`.

**Checkpoint**: windows and scores are tested; the stories can start.

## Phase 3: User Story 1 — Harvest every indicator (P1) 🎯 MVP

**Goal**: one causal, cached frame per pair, with every indicator column and raw signal.

**Independent test**:
- every in-scope class is harvested or excluded with a reason;
- raw signals equal the class's own pre-filter signals;
- every kept column passes truncation.

- [X] T007 [P] [US1] Implement `research/indicator_screen/discover.py` (R2):
  - import every module in `SimpleStrategies/`;
  - keep concrete subclasses of `SimpleStrategy`, excluding `TriggerOnlyStrategy` subclasses and the trigger arms (`Breakout*Open`, `Gbb*Open`, `GeometryOpen`);
  - return `ClassEntry` items plus `excluded` with reason ∈ {trigger_arm, import_error};
  - CLI `discover [--json]`.
- [X] T008 [P] [US1] Implement `research/indicator_screen/overrides.py` (R1 table):
  - `ehlers_triggers`: calls the class's own `get_cycle_entry_trigger`, `get_trend_entry_trigger`, `get_cycle_exit_trigger` and `get_trend_exit_trigger` after its `get_entry_signals` has populated the Ehlers columns with `atr_ok = True`; emits 4 raw signals;
  - `force_trend_mode` (BBBreakout): calls `get_entry_signals` / `get_exit_signals` with the `trend_mode` column forced to 1, then to 0, giving variants `tm1` / `tm0`;
  - `ichimoku_switches`: sets `entry_use_cloud_thickness` and `entry_use_htf_filter` to False on the instance;
  - a registry mapping class → override; the default is a plain `get_entry_signals` / `get_exit_signals` call.
- [X] T009 [US1] Implement `research/indicator_screen/harvest.py` (R1, R13):
  - load classes via `StrategyResolver` (pattern `evaluation/pick_threshold._load_fold_context`) with `config/config_wf.json`;
  - build the base frame once per pair (`DataframePopulator.add_indicators(MINIMAL)` + `SimpleStrategy.add_additional_indicators`);
  - per class: copy the base, set `atr_ok = True`, apply the override, collect the new columns as `<class>:<col>` and the raw signals as `raw_buy:<class>[:variant]` / `raw_sell:...` (int8);
  - record aliases for columns identical to a base column; mark constant or all-missing columns and failing classes (`signal_error`) as excluded;
  - add the 24 ARO model inputs as `aro:<col>` from the ARO control class's own indicator step on raw OHLCV (the `include_list` of `NNNC_A_Raw_Open` loaded via `_load_fold_context`), aliasing any identical to a `base:` column (R1, FR-006);
  - add `base:gbb_candidate` via `research.outcomes.triggers.evaluate("gbb", frame)` on the frame built by the trigger's own populate step (the `research/outcomes/table.py` path), not the harvest base frame (R5);
  - write the feather to `user_data/data/indicator_screen/<PAIR>.feather` plus the sidecar JSON with `code_sha`, and update `research/indicator_screen/manifest.json`;
  - CLI `harvest --pair <PAIR|BTC/USDT|all> [--force]` and `--time-one <PAIR>`;
  - the running-chain refusal reuses the `ps` check from `research/live_pnl/parity.py`.
- [X] T010 [US1] Implement `research/indicator_screen/causality.py` (R12):
  - an adapter over `research.outcomes.causality`'s slices and cut scheme: populate = harvest one frame; compare every column up to each cut (NaN == NaN, relative tolerance 1e-9);
  - write `research/indicator_screen/causality.json` and set `causality` / `first_mismatch` in `manifest.json`;
  - the harvest drops FAIL columns from the cache and records them excluded with reason `lookahead`.
- [X] T011 [P] [US1] Tests in `research/indicator_screen/tests/test_discover.py` and `research/indicator_screen/tests/test_overrides.py`:
  - discovery includes CCI, Combo, Ehlers_BB and BBBreakout, and excludes GbbOpen and BreakoutVolOpen with `trigger_arm`;
  - each override CHANGES the signal relative to the as-built `get_entry_signals` on a real 2,000-bar SOL slice (Principle IV);
  - the BBBreakout `tm1 | tm0` union equals a version whose trend_mode alternates, exercising the class's own code path;
  - Ichimoku with the switches off differs from switches on whenever the cloud filter would bind.
- [X] T012 [P] [US1] Tests in `research/indicator_screen/tests/test_harvest.py` and `research/indicator_screen/tests/test_causality.py`:
  - for CCI on a slice, the harvested `raw_buy:CCI` equals `CCI().get_entry_signals(frame)` exactly, and no SimpleStrategy filter column (`atr_ok`, `market_state`, …) changes it;
  - aliases are detected;
  - a constant column is excluded;
  - the sidecar sha mismatch forces a rebuild;
  - the causality adapter flags a planted centred rolling mean, and passes a trailing one;
  - all 24 ARO `include_list` columns are present (as `aro:` or as an alias) and equal the control's frame on one fold (FR-006);
  - `base:gbb_candidate`'s fire count per pair equals the ARO control's candidate count on the same fold span (R5).
- [X] T013 [P] [US1] Implement `SimpleStrategies/AllIndicators.py` (R14):
  - a `SimpleStrategy` subclass whose `populate_indicators` calls the harvest functions for the pair;
  - `populate_entry_trend` / `populate_exit_trend` set no signals;
  - parameter dicts merged from the parent;
  - add `AllIndicators` to `research/indicator_screen/discover.py`'s exclusions (reason `harvest_wrapper`).
- [X] T014 [US1] RUN the cost gate:
  - `harvest --time-one SOL/USDT`; record per-class times in `research/indicator_screen/manifest.json` (`timing`);
  - if the total for 11 frames exceeds about 3 h, stop and report the slow classes before continuing.
- [X] T015 [US1] RUN `causality`; then `harvest --pair all` (10 pairs + BTC/USDT). Confirm:
  - `manifest.json` lists every class as harvested or excluded with a reason (SC-001);
  - both bots are alive.
- [X] T016 [US1] Plot check: run `freqtrade plot-dataframe -s AllIndicators -c user_data/strategies/config/config_wf.json -p SOL/USDT --timerange 20250101-20250115` (outer repo root) and confirm the HTML is written. The plot is not committed.

**Checkpoint**: the harvest is complete, causal and cached.

## Phase 4: User Story 2 — Single-indicator screen against gbb_path (P1)

**Goal**: ranked columns against BUY/SELL on all bars and on candidate bars, with redundancy and
residual strength; the instrument check gates everything.

**Independent test**: the instrument check passes; every column has early and E0 scores.

- [X] T017 [US2] Implement the label part of `research/indicator_screen/targets.py` (R4):
  - `label:buy` / `label:sell` from the ARO control class `NNNC_A_Raw_Open` with the `wf_aro_g24` params snapshot, via `get_training_labels` on each pair's full-range frame;
  - the last 48 bars are NaN;
  - stored into the harvest cache as `label:buy` / `label:sell` (a harvest step, re-run without recomputing indicators).
- [X] T018 [P] [US2] Test in `research/indicator_screen/tests/test_targets.py`: full-range labels equal the per-fold labels from `research/label_screen/audit._frames(control class, "f03")` for every pair on that fold's span (R4). Mark it `slow` if it exceeds 60 s, but keep it in the suite.
- [X] T019 [US2] Implement `research/indicator_screen/screen_single.py` for `--targets labels`:
  - per window × column × target × population: AUC, MI (edges from the fold's train span; E0 edges from itself, flagged `self`), lift/fire rate for raw signals, and stratified AUC on candidate bars with grid edges from the train-span candidate bars;
  - aggregation per R6, plus `width_sort_auc` (bb_width's AUC on the same rows);
  - instrument check first: the top-5 rule from `preregistration.json` on `label:buy` / all bars over `instrument_gate_halves` (E0 + early). On failure, write `{"instrument": "FAIL", ...}` to `single_labels.json` and exit 1. When `late` is scored, re-run it over f00–f14 and store it as `instrument_confirm` (SC-002);
  - set `candidate_flags.e0_agrees` (E0 median direction equals the early and late directions) and count a candidate flag only when it is true (FR-018, Principle II);
  - redundancy (R8): Spearman on a seed-0 sample of 200,000 early candidate bars, average linkage on 1 − |ρ| cut at 0.8, plus `rho_guard` / `rho_width` on all bars and on candidate bars;
  - `--halves` respected; `late` calls `windows.late_lock()`;
  - output `research/indicator_screen/single_labels.json` (data-model ScreenResult and WindowScore).
- [X] T020 [P] [US2] Tests in `research/indicator_screen/tests/test_screens.py` (single part), using a synthetic 3-pair cache fixture:
  - a planted column that equals the label ranks first;
  - the instrument check FAILs when guard_metric is shuffled;
  - `late` is refused without the lock;
  - a rerun is byte-identical (SC-007);
  - MI edges come from the train span (assert via a column whose train and test distributions differ);
  - a planted column that agrees early and late but flips in E0 is NOT a candidate (`e0_agrees` false).
- [X] T021 [US2] RUN `screen_single --targets labels --halves e0,early`. *(First run 2026-10-16: FAIL under the top-5 rule; rule amended with the user's approval (SC-002) and gbb_candidate removed from scored columns; re-run.)* Record the instrument verdict. If it is FAIL: stop, diagnose, and report to the user before any further run task.

**Checkpoint**: instrument PASS; early and E0 label scores exist.

## Phase 5: User Story 3 — Combinations (P2)

**Goal**: pairwise AND screen and per-fold tree models against the bb_width + guard_metric model.

**Independent test**:
- selection used only early windows;
- the confirmation is a single late pass;
- model training rows end one purge before the test window.

- [X] T022 [P] [US3] Implement `research/indicator_screen/screen_pairs.py` (R9):
  - `--select`: all unordered pairs of `raw_buy:*` signals (for `label:sell`: `raw_sell:*`) with causality PASS, combined with AND; lift per early window and E0;
  - populations: candidate bars for label targets; all bars (primary) and candidate bars (secondary) for regime targets, lift for the top tercile (FR-016, FR-019);
  - apply the pre-registered selection rule (computable in ≥ min_windows early windows; median lift ≥ uplift × max(component lifts) and ≥ min_lift; top_n);
  - `--confirm`: survivors only, on late windows, after `late_lock()`; set `confirmed` = "yes" only if late passes AND E0 lift ≥ max(component E0 lifts), else "early+late only" or "no" (R9);
  - output `research/indicator_screen/pairs_<target>.json` (PairResult).
- [X] T023 [P] [US3] Implement `research/indicator_screen/screen_models.py` (R10):
  - `--target <id> --fold fNN`: training rows = candidate bars for label targets; for regime targets, all bars (primary; every 4th bar = hourly, with the purge applied in bars BEFORE subsampling) and candidate bars (secondary; not subsampled); filtered by `windows.train_rows_mask`;
  - A2 targets: training rows are A2 bars of test windows f00…f(k−1) ending ≥ N bars before test_start; f00 is written as `computable: false, reason: "A2: no prior test windows"` (R10);
  - fit the tree (depth 3) and the HGB model with the pre-registered settings on all columns, and on [bb_width, guard_metric] (`restricted`); score test AUC;
  - add the width-sort AUC; for `label:buy` only, add the ARO p_buy AUC from `research.nnnc_framework.signals.fold_table("NNNC_A_Raw_Open"-fold class, test range)` when available;
  - export the tree paths as text, plus the top split columns;
  - late folds require `late_lock()`;
  - `--merge` aggregates `research/indicator_screen/models/<target>/<fold>.json` into `models_<target>.json` with early/late medians, the candidate rule (full > restricted in both halves), the computable fold count, and `persistence: "distant era not tested"` on every candidate.
- [X] T024 [P] [US3] Tests in `research/indicator_screen/tests/test_screens.py` (combination part):
  - a planted pair (A AND B = label) is selected and confirmed;
  - a pair is never scored on late windows during `--select` (assert the windows read);
  - `train_rows_mask` excludes the purge;
  - the restricted model uses exactly the two columns;
  - tree rules export with at most 3 conditions per path;
  - per-fold output is deterministic;
  - the hourly subsample never includes a row inside the purge (purge applied before subsampling);
  - an A2 model on f00 is written as not computable;
  - a planted pair confirmed early and late but failing E0 gets `confirmed: "early+late only"`.
- [X] T025 [US3] RUN `screen_pairs --target label:buy --select` and `screen_pairs --target label:sell --select` (early + E0 only).

## Phase 6: User Story 4 — Regime screen (P2)

**Goal**: the same singles, pairs and models against regimes A1, A2, B and C at 12 h / 3 d / 7 d,
with the pair's own and BTC's indicator values and persistence baselines.

**Independent test**:
- targets use only (t, t+N];
- baselines use only [t−N, t) and closed trades;
- tercile cut points come from the declared span.

- [X] T026 [US4] Extend `research/indicator_screen/targets.py` with the regime targets (R11):
  - A1:N from `research/outcomes/table.load_table` (Open operating point) candidate rows, pooled mean profit over (t, t+N];
  - A2:N from `w_aro` trades (`evaluation/folds/plan_w_aro.json` arm records; loader pattern from `research/live_pnl/shared_wallet._fold_trades`), pooled mean `profit_ratio` over open times in (t, t+N];
  - B-btc/B-pair forward log return; C-btc/C-pair forward std of 15m log returns;
  - terciles with cut points from the fold's train span (A2: early test windows), top = 1, bottom = 0, middle dropped;
  - baselines over [t−N, t), using only trades that CLOSED before t for A1/A2.
- [X] T027 [US4] Add the BTC join: in `research/indicator_screen/screen_single.py`, and in the pair and model loaders, read BTC/USDT's cache columns and join them by date under the `btc:` prefix for regime targets (FR-020).
- [X] T028 [P] [US4] Tests in `research/indicator_screen/tests/test_targets.py` (regime part):
  - changing any bar at or before t never changes the target at t, while changing a bar in (t, t+N] does;
  - changing a bar in [t, t+N] never changes the baseline at t;
  - an A1/A2 trade closing after t is excluded from the baseline;
  - tercile cut points come from the declared span;
  - A2 is not computable outside the w_aro test windows.
- [X] T029 [US4] Extend `screen_single --targets regime` to:
  - score every target × horizon × {pair, btc} column set, on all bars (primary) and candidate bars (secondary) (FR-019);
  - for market-wide targets (A1, A2, B-btc, C-btc) × `btc:` columns, de-duplicate to one row per timestamp before scoring and set `dedup: true` (R11);
  - apply the E0 rule (`e0_agrees`); A2 gets `e0_agrees: null` ("distant era not tested");
  - report `baseline_auc` and `beats_baseline_both_halves`;
  - report the overlap with the persistence baseline (FR-021, spec US4 scenario 3);
  - output `research/indicator_screen/single_regime.json`.
- [X] T030 [US4] RUN `screen_single --targets regime --halves e0,early` (A2 has no E0). Then run `screen_pairs --select` for each regime target id at each horizon, early + E0 only.

## Phase 7: Pre-registration lock and late windows

- [X] T031 Review `research/indicator_screen/preregistration.json` against the early results:
  - adjust only the thresholds marked calibratable (`min_rows`, `min_fires`), recording the reason in a `calibration_note` field;
  - COMMIT it — this commit is the lock. Record its sha in the file's commit message and in `report.md`.
- [X] T032 RUN the late single screens: `screen_single --targets labels --halves late` and `screen_single --targets regime --halves late`. The labels run also writes the 15-window instrument confirmation (SC-002).
- [X] T033 RUN `screen_pairs --confirm` for `label:buy`, `label:sell` and every regime target with survivors.
- [ ] T034 First time one fold (`label:buy` f07, and one regime target on all bars at f14, the largest training span); if 15 folds × all targets would exceed about 6 h, report the per-target estimate to the user before continuing (Principle VII). Then RUN `screen_models` one fold per process (f00–f14), then `--merge`, for: `label:buy`, `label:sell`, and each regime target at each horizon. Run them sequentially in one chain, confirming both bots are alive between targets.

## Phase 8: Polish & cross-cutting

- [X] T035 Implement `research/indicator_screen/report.py`:
  - `report.md` with the instrument verdict, the exclusions and causality failures, and the top columns per target/population with halves;
  - redundancy groups, and the residual table with the width-sort baseline;
  - pair survivors, the model comparison per target (rules and recurring splits), and regime results against baselines;
  - the SC-004 answer lists (a)–(e) (an empty list is a valid answer);
  - the SCREEN label (FR-025) and the autocorrelation caveat for 3 d / 7 d regimes;
  - the instrument gate (E0 + early) and its 15-window confirmation;
  - a persistence column on every candidate: "E0 agrees", "early+late only", or "distant era not tested" (models, A2), with the note that a later backtest must cover 2022.
- [X] T036 [P] Tests in `research/indicator_screen/tests/test_report.py`: the report contains every SC-004 heading; empty lists render as "none"; the FR-025 SCREEN statement is present; every candidate row has a persistence value.
- [ ] T037 RUN `report`; verify SC-007 by re-running T032's label screen into a scratch path and diffing it byte for byte.
- [ ] T038 Add a "Current state" note for 037 to `AGENT_GUIDE.md`, pointing to `research/indicator_screen/report.md` (FR-022). Add the project memory file `project_037_indicator_screen.md` plus its index line in `MEMORY.md`.
- [ ] T039 Run the full suite (AGENT_GUIDE command with `research/indicator_screen/tests` added); confirm no failures. Check that both bots are alive, then commit (named files only).

## Dependencies & execution order

- **Setup → Foundational:** T001–T002, then T003–T006.
- **US1** (T007–T016) blocks US2–US4: they all read the cache.
- **Within US1:** T007, T008 and T013 can run in parallel; T009 depends on T007 and T008; T010 depends on T009; T011 and T012 can run in parallel after their modules exist; T014 → T015 → T016.
- **US2:** T017 → T019 → T021. T018 and T020 run in parallel with T019. T021's instrument PASS gates every later RUN task.
- **US3:** T022 and T023 can be built in parallel with US2 once T005 exists; T025 needs T021.
- **US4:** T026 → T027 → T029 → T030; T028 runs in parallel with T029.
- **Phase 7:** T031 needs T021, T025 and T030. T032–T034 need T031.
- **Phase 8:** T035 and T036 can be built anytime; T037 needs T032–T034; T038 and T039 come last.
- **Run tasks are sequential** (one heavy chain), and no RUN task overlaps a training or backtest chain.

## Parallel examples

- **After Foundational:** T007 discover.py, T008 overrides.py and T013 AllIndicators.py; also T022 screen_pairs.py and T023 screen_models.py, which use synthetic fixtures until the cache exists.
- **Tests:** T011, T012, T018, T020, T024, T028 and T036 are all separate files.

## Implementation strategy

1. **MVP = US1 + US2** (T001–T021): the harvest, plus the label screen on early and E0 windows. It already answers "which indicators carry information about gbb_path, and which are guard_metric or bb_width in disguise".
2. **Then US3 and US4 selection on early data** (T022–T030).
3. **Then the lock (T031) and every late pass** (T032–T034).
4. **The report and polish** (T035–T039).
