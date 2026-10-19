---

description: "Task list for 038 close-price forecast study"
---

# Tasks: Close-Price Forecast Study

**Input**: Design documents from `specs/038-close-forecast-study/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: These are requested by the spec (known-answer tests, FR-005; instrument checks, SC-001/002;
refusal guard, FR-009), so they are included and written before the code they test.

**Organization**: Tasks are grouped by user story. All paths are relative to `user_data/strategies/`.
Run commands with `PYTHONPATH=.:evaluation ../../.venv/bin/python`, after
`source ~/.freqtrade_env`. Runs are sequential, one training job at a time. Never touch pids 2661 or
41849.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Create the package skeleton `research/forecast/__init__.py` and `research/forecast/tests/__init__.py`, and the output directory `../data/forecast_038/{frames,pred}` (git-ignored by the outer `user_data/*` rule)
- [X] T002 [P] Commit the restored exploration data `utils/test_data.npy` (993 values, from git commit 01918256) so that `utils/test_forecasters.py` runs from `utils/`

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: shared constants, candle frames and the scorer that every story reads.

- [X] T003 Write `research/forecast/common.py`:
  - constants: `PAIRS` (the 10 pairs from `config/config_wf.json`), `HORIZONS = (1,2,4,8,16,32,48,96)`, `FEE = 0.0010`, `SLIPPAGE_BPS = (0, 2.5, 5, 10)`, `VERDICT_BPS = 5`;
  - `FOLDS`, read from `evaluation/folds/plan_cs_aro.json` (fold_id → test_start/test_end);
  - `EARLY = f00–f07`, `LATE = f08–f14`, and `FWD = (2026-09-07, last candle)`;
  - the paths `OUT = ROOT/user_data/data/forecast_038` and `HERE`;
  - `cost_rt(s)`;
  - `prereg()` (loads `preregistration.json`);
  - `refuse_unless_registered(result_paths)`: reuse `research.control_sizer.judge._git`. It refuses when the file is untracked, has uncommitted changes, or any result is older than its last commit
- [X] T004 [P] Write the test `research/forecast/tests/test_frames.py`:
  - on a synthetic 300-candle frame, `target[n] == lc[t+n]-lc[t]`;
  - `trade_ret[n] == ln(open[t+1+n]/open[t+1])`;
  - the last n rows are NaN (never 0);
  - `eligible` is False when the entry or exit candle has zero volume or was gap-filled;
  - fold and half labels match `FOLDS` at the window edges (start inclusive, end exclusive)
- [X] T005 Write `research/forecast/frames.py`:
  - load `binanceus/<PAIR>-15m.feather` through freqtrade's `history.load_pair_history(fill_up_missing=True)`, and also the raw file for `has_data`;
  - compute `lc`, `target[n]`, `trade_ret[n]`, `eligible[n]`, `fold` and `half` (R1–R3);
  - join `data_ok` from the 037 cache (`harvest.read_cache(pair, ['meta:data_ok'])`, on date); after the cache ends, `data_ok = has_data`;
  - write `../data/forecast_038/frames/<PAIR>.feather`, with CLI `[--pair P]`;
  - refuse (exit 2) if the fold windows do not match the plan;
  - make T004 pass
- [X] T006 [P] Write the test `research/forecast/tests/test_score.py`:
  - the causal threshold at candle t uses only forecasts before t (perturbing a later forecast changes nothing earlier);
  - a signal needs `f > 0` as well as `f ≥` the threshold: an all-zero forecast and an always-negative forecast produce 0 trades;
  - the trade simulation takes one position per pair at a time and skips signals inside an open trade;
  - net = gross − `cost_rt(s)`;
  - the random-entry benchmark is the mean `trade_ret` of eligible candles in the same pair and half;
  - the week-block bootstrap is reproducible with a fixed seed;
  - `r2_oos` of the zero forecast is exactly 0
- [X] T007 Write `research/forecast/score.py`:
  - `signals()`: LONG when `f > 0` and `f ≥` the causal 0.9 quantile of the previous 30 days of forecasts for that pair, method and n (stride-aware lookback: 2,880 or 180);
  - `simulate()`: the trades;
  - `cell()`: every field in data-model § Cell score, including per-pair rows, the short-side mirror and `fail_share`;
  - `bootstrap_lb()`: the one-sided 95% bound over calendar-week blocks, 2,000 resamples, seed 0;
  - descriptive metrics on all candles and on traded candles;
  - CLI `[--method M]` writes `research/forecast/scores.json` (merged per method);
  - make T006 pass
- [X] T008 Write `research/forecast/preregistration.json`, holding every number in R9 and data-model § Pre-registration:
  - horizons, pairs, folds and halves, and the forward window (descriptive);
  - fee, slippage grid and `verdict_bps = 5`;
  - quantile 0.9 with a 30-day lookback;
  - bootstrap settings (week blocks, 2,000 resamples, one-sided 0.95);
  - `min_trades_per_half = 200` and `max_fail_share = 0.05`;
  - the method list with params (statistical models on a 64-candle window at stride 16; LightGBM params from R7; feature sets L, LA and LW);
  - the decision text (PASS (PROVISIONAL), FAIL, INCONCLUSIVE), the null-downgrade rule and the seed-2 rule;
  - the definitions of most viable N and break-even N

**Checkpoint**: frames exist for all 10 pairs, and the scorer tests pass.

---

## Phase 3: User Story 1 - A forecast library that does what it claims (Priority: P1) 🎯 MVP

**Goal**: every forecaster forecasts exactly the requested horizon, fails loudly, and never mixes
windows (R4: D1–D6).

**Independent Test**: `pytest utils/tests/test_forecaster_horizons.py` is green, and the plot script
runs.

- [X] T009 [US1] Write the test `utils/tests/test_forecaster_horizons.py`:
  - horizon index (the D1 defect): for Holt, AR, Exponential, SimpleExponential, StateSpace, ETS, ARIMA and Theta, with `set_detrend(False)` on a fixed random-walk window (64 points, seed 0), `forecast(x, h)[-h:]` equals the underlying statsmodels model's own `fit().forecast(h)` (same specification) to 1e-9 for h ∈ {1, 4, 96}. A one-step shift fails this;
  - straight line: Linear and Quadratic on `x = 0.01·t` give `forecast(x, h)[-1] == 0.01·(63+h)` to 1e-6 (the smoothing models are not tested on a line, because SES, ExpSmoothing, ETS and ARIMA(2,1,2) model no trend and damped Holt flattens out);
  - a constant series forecasts the constant (all models, 1e-6);
  - ARIMA on a pathological input raises `Forecasters.ForecastFailed`;
  - `FFT_EXTRAPOLATION` raises `Forecasters.ForecastUnavailable`;
  - with detrending on, `pa_forecaster`'s retrend uses the forecast window's own trend: two different windows forecast with one trained model get different restored trends;
  - confirm that the test FAILS before T010–T012
- [X] T010 [US1] Fix D1 and D2 in `utils/Forecasters.py`:
  - exponential, statespace_exponential, simple_exponential, ets and arima use `fit().forecast(steps)` and return `concat(x, preds)[-len(x):]` with exactly `steps` predictions;
  - simple_exponential and exponential pass `initialization_method="estimated"`;
  - check theta, holt and ar against the same contract
- [X] T011 [US1] Fix D3 and D5 in `utils/Forecasters.py`:
  - add `class ForecastFailed(RuntimeError)` and `class ForecastUnavailable(RuntimeError)`;
  - arima re-raises a fit failure as `ForecastFailed`, never returning zeros;
  - `fft_extrapolation_forecaster.forecast` raises `ForecastUnavailable("predict_type 4 feeds FFT coefficients to a regressor trained on returns")`
- [X] T012 [US1] Fix D4 in `utils/Forecasters.py`: `pa_forecaster`, `sgd_forecaster` and the other regressors that call `retrend_results` restore the trend of the forecast input window, using `detrend` on the input and then `retrend`, not the stored training-results detrender. Keep `detrend_results` for training only
- [X] T013 [P] [US1] Fix D6 in `TSPredict/TS_Wavelet.py`: `predict()` and `rolling_predict()` raise `NotImplementedError("dead path; add_rolling_predictions is the live path")`. They have no callers (checked with grep)
- [X] T014 [US1] Run `pytest utils/tests/test_forecaster_horizons.py utils/tests -q`; all green. Then run `cd utils && MPLBACKEND=Agg ../../../.venv/bin/python test_forecasters.py` and confirm it exits without an exception (AS5). FFT_EXTRAPOLATION is now unavailable, so switch the script's enabled forecaster to `HOLT` and catch `ForecastUnavailable` with a printed message

**Checkpoint**: Story 1 is done. The library meets the contract in `contracts/cli.md`.

---

## Phase 4: User Story 2 - Baselines and the cost line per horizon (Priority: P1)

**Goal**: for each N, the typical move against cost, and how the zero and historical-mean baselines
score.

**Independent Test**: `baselines.json` covers 10 pairs × 8 N, and `instrument.json` shows a zero skill
of exactly 0.

- [X] T015 [US2] Write `research/forecast/baselines.py`:
  - `zero` (f = 0);
  - `hist_mean`: f[n] = n × the trailing mean one-candle log return of the pair over all history before the fold start (anchored, causal; recomputed per fold);
  - write `../data/forecast_038/pred/{zero,hist_mean}.feather`;
  - write `research/forecast/baselines.json` with a horizon table per pair and pooled: `typical_move`, `cost_rt[s]` and `move_to_cost[s]`, with `untradeable` = `move_to_cost[5bps] < 1`
- [X] T016 [US2] Write `research/forecast/instrument.py`, which writes `research/forecast/instrument.json` with these checks:
  - SC-001: the exit code of `pytest utils/tests/test_forecaster_horizons.py`;
  - SC-002a: `r2_oos(zero) == 0.0` exactly;
  - SC-003: the unscored share of eligible candles not caused by gaps is < 1%;
  - R12 liveness: hist_mean forecasts differ across N
- [X] T017 [US2] Run `frames` (all pairs), `baselines`, `score --method hist_mean` and `instrument`, and print the horizon table. The instrument must PASS before Phase 6 runs any model

**Checkpoint**: the cost line is known. Horizons with `move_to_cost < 1` are marked untradeable before
any model is read.

---

## Phase 5: User Story 6 - Verdict per method and horizon (Priority: P1)

**Goal**: the pre-registered rule produces one verdict per (method, N), and the tool refuses to run when
the pre-registration is not committed.

**Independent Test**: `pytest research/forecast/tests/test_verdict.py` is green, and running `verdict`
with a dirty pre-registration exits 2.

- [X] T018 [US6] Write the test `research/forecast/tests/test_verdict.py`. On synthetic cell scores:
  - a cell with ≥200 trades per half, net > 0 in both halves, edge_vs_rand > 0 in both halves and late lb95 > 0 → `PASS (PROVISIONAL)`;
  - net ≤ 0 in either half → FAIL;
  - edge_vs_rand ≤ 0 in either half → FAIL;
  - < 200 trades → INCONCLUSIVE;
  - fail_share > 0.05 → INCONCLUSIVE;
  - a learned PASS without a seed-2 PASS → INCONCLUSIVE;
  - a null-control PASS at a tier downgrades that tier's PASSes;
  - `most_viable_n` and `break_even_n` are computed as in R9;
  - the refusal guard returns 2 for an untracked or dirty pre-registration
- [X] T019 [US6] Write `research/forecast/verdict.py` to implement R9 from `preregistration.json` (thresholds are read, never hard-coded). It:
  - writes `research/forecast/verdicts.json`;
  - refuses via `common.refuse_unless_registered` and also refuses unless `instrument.json` PASSes;
  - makes T018 pass
- [X] T020 [US6] Commit the pre-registration before any non-baseline result exists: `git add` `research/forecast/preregistration.json` and every file written in T001–T019 (named explicitly), then commit "038: pre-register forecast pass rule". Check `git log` and confirm that no `../data/forecast_038/pred/*` other than zero and hist_mean exists

**Checkpoint**: the rule is fixed and committed. From here on, results can be read.

---

## Phase 6: User Story 3 - Statistical forecasters (Priority: P2)

**Goal**: the corrected statistical tier, scored against the baselines and the cost line.

**Independent Test**: on one fold, every forecast uses only candles before t, and the forecast count
equals the number of stride-16 eligible candles.

- [ ] T021 [P] [US3] Add a test in `research/forecast/tests/test_frames.py` (stat-tier causality): on a synthetic pair, changing `close` after candle t leaves the stat-tier forecast at t unchanged, and the row count equals the scored candles at stride 16
- [ ] T022 [US3] Write `research/forecast/stat_tier.py`:
  - for each pair and scored candle at stride 16, take the trailing 64 log-close values (or 64 one-candle returns for AR), call the Forecasters model with `set_detrend(False)`, forecast 96 steps, and read `f[n] = path[n-1] − lc[t]` (for AR, the cumulative sum of the return path);
  - catch `ForecastFailed` and set `failed=True`;
  - use a `multiprocessing` pool across pairs with `--workers 4` (CPU only);
  - refuse unless the pre-registration is committed;
  - skip a window when its last candle is gap-filled or more than half of its 64 candles are (`has_data` False); write it as `skipped_gap=True` with NaN forecasts (spec Edge Cases, data-model § Forecast record);
  - write `pred/<model>.feather`;
  - models: ar, holt, ses, expsm, ets, ss_expsm, arima, theta;
  - make T021 pass
- [ ] T023 [US3] Run `stat_tier --workers 4` (target: under 1 h together with Phase 4, SC-006, so record the elapsed time), then `score` for the 8 models. Report the fail share per model and N, and the R12 liveness check (forecasts differ across N)

**Checkpoint**: tier 1 is scored.

---

## Phase 7: User Story 4 - Learned forecasters (Priority: P2)

**Goal**: LightGBM walk-forward on lagged returns (L) and with the ARO inputs (LA), behind a
shuffled-target null control.

**Independent Test**: the gap between the end of the training labels and the test start is ≥ N, and the
null control scores within noise of zero.

- [ ] T024 [P] [US4] Add a test in `research/forecast/tests/test_frames.py` (learned-tier purge): `learned_tier.train_slice(fold, n)` ends at least n candles before the fold's test start for every n, and the null control's shuffled targets are a permutation within the training span only
- [ ] T025 [US4] Write `research/forecast/features.py`:
  - set L: trailing log returns over {1,2,4,8,16,32,64,96}, realised volatility over 16 and 96 candles, and the log volume ratio 16/96;
  - set LA: L plus the 24 `base:` columns aliased by `aro:` in `research/indicator_screen/manifest.json`, read through `harvest.read_cache` and joined on date (the cache ends 2026-09-07, so LA has no forward window)
- [ ] T026 [US4] Write `research/forecast/learned_tier.py` with `--set {L,LA,LW,null} --seed {1,2}`:
  - per fold and n, train LightGBM with the R7 parameters, from 2021-01-01 to `test_start − n` candles, pooled over pairs, training rows at stride 4. Rows need `eligible`, real data at t (`has_data`), finite features and, for every set, `data_ok` at t (the 037 cache's `meta:data_ok` inside the cache range; in the forward window, after the cache ends on 2026-09-07, `has_data` only), so L, LA, LW and null train and score on the same candles (a paired comparison);
  - predict the fold's test candles, and for L, LW and null also the forward window from the f14 model;
  - null = set L with targets permuted within the training span (seed-fixed);
  - refuse unless the pre-registration is committed;
  - write `pred/lgb_<set>[_s2].feather`;
  - make T024 pass
- [ ] T027 [US4] Run `learned_tier --set null`, then `score`. The null control must show no tradeable skill (it should FAIL or be INCONCLUSIVE at every N, and its pooled IC should be within noise of zero). Record this in `instrument.json` as SC-002b. If it fails, stop and report (no learned result is read)
- [ ] T028 [US4] Run `learned_tier --set L`, then `--set LA`, sequentially, then `score`

**Checkpoint**: tier 2 is scored behind a passing null.

---

## Phase 8: User Story 5 - Decomposition features (Priority: P3)

**Goal**: SWT and FFT inputs added to L, checked for causality first.

**Independent Test**: `truncation.json` shows every LW input identical up to each cut.

- [ ] T029 [US5] Add the LW inputs to `research/forecast/features.py`: on the trailing 64 one-candle log returns, the SWT `bior3.9` level-2 coefficients at the last position (cA2, cD2, cD1), using `utils/Wavelets.py` or pywt directly, plus the FFT magnitudes of bins 1–8
- [ ] T030 [US5] Write `research/forecast/truncation.py`: for 2 pairs × 5 random cut points, recompute every LW input on data truncated at the cut and compare all values up to the cut. Write `research/forecast/truncation.json`. Any input that fails is excluded and named (FR-012)
- [ ] T031 [US5] Run `truncation`, then `learned_tier --set LW` (with only the inputs that passed), then `score`. Report the paired difference LW − L per n and half

**Checkpoint**: tier 3 is scored.

---

## Phase 9: Polish and closing

- [ ] T032 Run `verdict`. For any learned PASS, run `learned_tier --set <S> --seed 2`, then `score` and `verdict` again (quality gate 5)
- [ ] T033 Write `research/forecast/report.py` and run it to produce `research/forecast/report.md`, containing:
  - the horizon and cost table;
  - the verdict table per (method, N);
  - most viable N and break-even N;
  - the null-pass count;
  - per-pair and per-half detail;
  - the descriptive metrics;
  - the forward window (descriptive);
  - limitations (no distant era, so PROVISIONAL)
- [ ] T034 [P] Run `pytest research/forecast/tests utils/tests -q`; all green
- [ ] T035 Update `AGENT_GUIDE.md` "Current state", and the memory file `project_038_close_forecast.md` with a MEMORY.md index line (verdict, viable N, and what it implies for NNNC: a passing forecast is the candidate for a filter or sizer feature, otherwise new inputs are the lever)
- [ ] T036 Commit to master, staging only the named files (never `git add -A`; leave `NNNC/NNNCStrategy.py`, `research/live_pnl/{drift,gate}.json`, `utils/Wavelets.py` and `utils/test_fft.py` unstaged). End the message with the Co-Authored-By line

---

## Dependencies and Execution Order

- **Setup (T001–T002)**: depends on nothing.
- **Foundational (T003–T008)**: needs Setup. It blocks every story.
- **US1 (T009–T014)**: needs Foundational only for the repo layout. It is independent of the harness.
- **US2 (T015–T017)**: needs Foundational and US1 (the instrument reads the US1 test exit code).
- **US6 (T018–T020)**: needs US2. The pre-registration commit (T020) gates US3–US5.
- **US3 (T021–T023)**, **US4 (T024–T028)** and **US5 (T029–T031)**: need T020, and run sequentially (one job at a time). US5 needs US4's set L.
- **Polish (T032–T036)**: needs every story it reports on.

### Parallel opportunities

- T002 can run alongside T001.
- T004 and T006 (tests in different files) can run alongside T003.
- T013 (TS_Wavelet) can run alongside T010–T012 (Forecasters).
- T021 and T024 are tests that can be written in parallel.
- Runs never go in parallel.

## Implementation Strategy

- **MVP**: Phases 1–4 (library fixes plus the baseline and cost line). This alone answers "which N are
  even tradeable" and makes the library trustworthy for exploration.
- **Then**: Phase 5 fixes the rule.
- **Then**: Phases 6–8 add tiers, cheapest first.
- **Finally**: Phase 9 gives the verdict and closes the study.
