# Research: 040 NNMT Parity and Head Study

Decisions taken while planning, from reading the code on 2026-10-20. Each gives its reason and the
alternatives rejected. File references are to `user_data/strategies/`.

## R0 — Facts the plan rests on (code review, 2026-10-20)

- **The trading label is NNNC's label.**
  - `NNMTStrategy.get_trading_classes` (NNMTStrategy.py:1497–1549) builds BUY/SELL from
    `get_train_buy_signals` (gbb_path, TRAINING_TYPE 27, H = 48, ±2%, TrainingConfig.py) and
    returns the UNFILTERED classes.
  - The auxiliary task filter writes only `%train_buy`/`%train_sell` (debug columns). This has been
    the case since 2618e0eb (2026-09-01).
  - The older audit (memory `project_nnmt_task_audit`) recorded the label as task-filtered. It is
    stale and is corrected in memory at the end of the study.
- **Untrained heads.** `NNMT_MLX._CLASSIFIER_TASK_WEIGHTS = {trading 4, regime 0, risk 1, momentum 1,
  flow 1, profit 0}` (NNMT_MLX.py:44). Regime and profit get no gradient.
- **Auxiliary labels.**
  - Each is a centred rolling state shifted 5 bars ahead (PEAK_WINDOW 6, BaseNNStrategy.py:177):
    - regime: close vs a centred 200-bar mean, ±1.5%;
    - risk: a centred 50-bar mean of atr_norm, ±0.33;
    - flow: a centred 24-bar mean of di_diff_scaled, ±0.15;
    - momentum: a centred 24-bar mean of aroonosc_scaled, −0.5/0.6.
  - Profit is a forward-peak label over 8 bars.
  - All of them use future data. That is allowed in a label, but a screen must not credit a head
    for recovering the current state.
- **The network.** `NNMTClassifierMLX._NNMTModel` (NNMTClassifierMLX.py:165) always applies a
  resize and convolution preamble before the common layer and six heads. There is no plain-LSTM
  type.
  - The 14 `ClassifierTypeMLX` members are listed at :851.
  - NNNC's `LSTM_RAW` (NNNClassifierMLX.py:309–331, :365–373) is the plain design: no resize or
    convolution, features → LSTM → linear.
- **ARO's inputs are the 24 indicators, not raw8.** `NNNC_A_Raw_Open` is `LSTM_RAW` on the default
  24-indicator set (Framework/FeatureNormalizer.py:69–147), seq_len 16, at the open operating point
  (`NNNC_A_Raw_Open.json`: guards open, `prediction_threshold` 0.80, grace 24 h). raw8 is
  `NNNC/_raw8_features.py`. The 039 input-set mixin is `NNPredict/_inputs.py` (`InputSetMixin`,
  causal-normalised raw8 plus lagged returns).
- **NNMT's entry and exit path.**
  - Entry: the trading argmax with `prediction_threshold` 0.5, then the guard chain (guards ON:
    adx 38, atr 0.006, bb 0.06, rvol 0.2).
  - Exits: `enable_exit_signal` True (the sell head plus rvol), the bail-out on the profit/regime
    heads (`strategy_custom_exit`), and grace 3 h at −15%.
  - Model-side filters that are off by default: `min_consecutive_buys` 1, `apply_task_filters`
    False, the forward-peak regressor False.
- **There is no NNMT record on the 15-fold harness.** No `plan_*.json` names NNMT. The 202 NNMT
  records in `regime/arms/` are the old W1–W3 single windows.

## R1 — Framework port at the family base (FR-001)

**Decision**: `NNMTStrategy(EntryFrameworkMixin, BaseNNStrategy)` with:
- `ENTRY_MODE = "model"`;
- `MODEL_ENTRY_FAMILY = None`;
- `HELD_PAIR_ADD = None`.

**Rationale**:
- With `ENTRY_MODE = "model"`, `get_entry_conditions` is NNMT's own (the mixin calls `super()` and
  returns the mask unchanged).
- `MODEL_ENTRY_FAMILY` must be None. The AUTO default maps gbb_path to "reversion" (LabelFamilies.py),
  which would retag every NNMT entry `reversion:model` and change `enter_tag`. This is 039's
  adaptation.
- NNMT's own exit signal, bail-out and 3 h grace are not framework features, so they stay as they
  are.

**Identity proof (FR-001)**:
- The baseline `mt` runs before the port. The port lands, then `mt_fw0` re-runs the same folds,
  reusing the weights (the framework's names are model-key IGNORED).
- Pass = equal `trades_hash` on 15/15 folds (`research/nnpredict/judge.py identity` pattern).
- **Risk:** `StopInvariantMixin` (mode "halt") can newly raise. Before the full identity run, f00 is
  run with `FT_STOP_INVARIANTS=report`, and a violation is fixed or recorded before anything
  proceeds.

**Alternatives rejected**: porting the switches into NNMT by hand (a second copy of the framework,
which the constitution forbids).

## R2 — Model-key hygiene and mainline naming (FR-002)

**Decision**:
- **Move `use_forward_peak_regressor_filter`** from `MODEL_KEY_ATTRIBUTES` to `MODEL_KEY_IGNORED`.
  It is read only in `process_predictions` (NNMTStrategy.py:1213), after the model.
- **Classify every other post-model name** explicitly in `MODEL_KEY_IGNORED`, so the audit test
  covers them: `prediction_threshold`, `profit_prediction_threshold`, `min_consecutive_buys`,
  `apply_task_filters`, `forward_peak_regressor_threshold`, `enable_exit_signal`,
  `exit_rvol_threshold`, `stoploss_grace_*`. (Freqtrade Parameters already sit outside the key; the
  test proves it.)
- **Tests** (`NNMT/tests/test_model_key_nnmt.py`):
  - two classes differing only in a post-model setting have equal keys;
  - two differing in `_CLASSIFIER_TASK_WEIGHTS`, `classifier_type`, `INPUT_SET` or an auxiliary
    label parameter have different keys;
  - an unpinned NNMT base class resolves to a mainline location under an empty whitelist.

**Consequence**: moving a name changes the hash of every NNMT key. NNMT's 15 indexed pre-030
weights live in class-named directories, so the legacy fallback still serves them. Fold models are
trained fresh in this study, so nothing is lost.

## R3 — Harness, comparators and verdict rules

**Decision**:
- **Plans:** `evaluation/foldplan.py --family NNMT --base-strategy <class>` (15 anchored folds,
  `config_wf`, seed 1), the same shape as `plan_cs_aro.json`.
- **Runs:** `evaluation/run_folds.py` (strict evaluation, one record per fold in `regime/arms/`).
- **Verdicts:** `research.nnnc_switches.judge.judge_one`, with the rule maps and minimums of 039
  (min_control_trades 200, min_arm_trades 100).
- **Comparators** (existing records, no rerun):
  - `wf_aro_g24` (ARO);
  - `cs_aro` (NNNC reference book);
  - `w_aro` (width book);
  - `st_g24_rev` (reversion trigger).
- **Study code** lives in `research/nnmt/`. It copies and parametrises 039's NNPredict-specific
  scripts (`build_arms`, `judge` stages, `screen`) rather than generalising them in place, so 039's
  committed instruments stay byte-stable.

## R4 — Baseline and operating point (FR-003, FR-004)

**Decision**:
- `mt` = `NNMT_MLX` at its defaults (guards on, threshold 0.5, grace 3 h).
- `mt_op` = the same weights at ARO's operating point. The buy params are copied from
  `NNNC_A_Raw_Open.json` (guards open, `prediction_threshold` 0.80) through an NNMT
  `OperatingPointMixin` that merges them. Exits stay NNMT's.
- Both arms reuse one set of per-fold weights (`mt_op` differs only in post-model settings).

**No leakage across the fold boundary (Principle XI)**: each fold trains with `train_timerange`
ending at `test_start`. The training dataframe holds no candle at or after `test_start`, so the
forward labels of the last 48 bars (gbb_path H = 48) and the auxiliary labels' centred windows are
truncated, not filled from test bars. That is a purge of at least the label horizon. T011 verifies it
on f00: the last training bar's date is < `test_start`, and the label tail is HOLD/NaN rather than
looking past it.

**Model-decided share (FR-004)**: from per-fold signal tables (R8), the share of guard-passing bars
on which the model declines to enter: 1 − entries ÷ guard-passing bars. Near 0 means the guards
choose the entries. It is reported per arm. The pre-registered minimum for model and head arms to be
judged at the operating point is a share ≥ 0.5. Below that, the arm is INCONCLUSIVE.

**Alternative rejected**: a percentile floor. That was NNPredict's regression gate; NNMT is a
classifier and uses NNNC's threshold.

## R5 — Framework feature arms (FR-005)

**Decision**: each arm runs on the model selected by R7, at the operating point, and is judged
against that model's own framework baseline. All act after the model, so they reuse its weights.

| Arm suffix | Change |
|---|---|
| `_g24` | `stoploss_grace_hours` 24.0 (sell_params merged) |
| `_hpa` | `HELD_PAIR_ADD = {"in_profit_only": True}` |
| `_gate` | `ENTRY_MODE = "trigger_and_model"`, `TRIGGER_NAME = "gbb"` |
| `_tp` | `take_profit_atr_mult` 2.897 (C2, feature 028) |

**Liveness**: an arm unchanged on every fold is INERT (judge_one's `folds_changed`).

**Not tested**: family exits. NNMT's entries carry no family, and 039 found the momentum set fails.

## R6 — Model variants (FR-006, FR-007)

**Decision**: a multi-task plain LSTM, `ClassifierTypeMLX.LSTM_RAW`, in `NNMT/NNMTClassifierMLX.py`:
- inputs go straight into the LSTM common layer, with no resize or convolution preamble;
- each head is a linear layer (ARO's head);
- it is trained by the existing multi-task loop.

Input sets reuse `NNPredict/_inputs.py`'s `InputSetMixin` (causal-normalised raw8, no fitted
artifact).

| Arm | Network | Inputs |
|---|---|---|
| `mt_raw` | current NNMT network | raw8 |
| `mt_lr` | plain LSTM | the 24 indicators (ARO's inputs) |
| `mt_lr_raw` | plain LSTM | raw8 |
| `mt_lr_ri` | plain LSTM | the 24 indicators + raw8 |

- Same label, task weights, seq_len and exits as `mt`. All run at the operating point and are judged
  against `mt_op`.
- **Trading-head-only twin `<best>_tr`:** the selected variant with task weights
  `{trading 1, others 0}`, judged against the selected variant. It answers "does multi-task training
  help?" and is reported against `wf_aro_g24` (the single-task ARO) as context.

**Alternatives rejected**:
- Adding `LSTM_RAW` to `NNNClassifierMLX`: that classifier is single-task. The multi-task loop and
  heads live in `NNMTClassifierMLX`, as 039 put the regression twin in `NNPredictRegressorMLX`
  (see Complexity Tracking).
- More capacity, GAN augmentation, task-weight tuning: recorded as failed.

## R7 — Which model carries the feature, sizer and head stages

**Decision (pre-registered)**: among `mt_op`, `mt_raw`, `mt_lr`, `mt_lr_raw` and `mt_lr_ri`, pick the
highest pooled total over f00–f14 among arms with no WORSENED half against `mt_op`. Ties go to
`mt_op`.

**Caveat**: the selection sees both halves, so later arms' halves are not blind. Everything is
PROVISIONAL, as since 023.

## R8 — Signal tables with head probabilities

**Decision**:
- `research/nnnc_framework/signals.py` gains NNMT plan entries (PLAN_FAMILY "NNMT").
- Its `fold_table` keeps every head's class probabilities when the strategy exposes them.
- NNMT's `get_predictions` stores the per-head probability arrays on the instance (e.g.
  `self._head_probs[task] = probs`) before the argmax, read by a small family hook. The shared
  `Predictors/MLXClassifierMultiTask.predict` is unchanged.
- Columns: `p_<task>_<class>` for regime, risk, momentum, flow and profit, plus the trading p_buy.
- One forward pass per fold in strict mode, with no training.

## R9 — Sizer screen (FR-008)

**Decision**: 039's screen (`research/nnpredict/screen.py` pattern) on the 412 early width-book
trades (`research.new_inputs.trades.trades()`), scoring the selected model's p_buy.

**Pass (pre-registered, early half only, late half blind)**:
- gap ≥ 0.05 pp per trade (half of ARO's 0.1004);
- partial Spearman with trade pp given bb_width > 0;
- that sign in at least 5 of 8 folds.

**If it passes**:
- **Sizer arm `<best>_cs`:** `WidthSizerMixin, WidthBookMixin` over the selected NNMT class,
  `AGREE_RULE = "p_buy"`, per-fold 0.5-share threshold (`research.control_sizer.pick`).
- **Verdicts:** `judge_one(_cs, w_aro)` must PASS; against `cs_aro` it is reported.
- The arm runs on `w_aro`'s exits (framework copied by `framework_sha`, as 039's follow-ups 3 did),
  so the sizer is not confounded with NNMT's exit stack.

## R10 — Auxiliary-head screens (FR-009)

**Model**: a "heads-trained" twin `<best>_ah` of the selected variant with all six task weights
non-zero (trading 4, every auxiliary 1). This is the only way to screen the regime and profit heads
(edge case "heads that are never trained"). Its trading head is reported against the selected
variant (FR-009 side check, not a verdict on its own).

**Screen 1 (skill)**, on test windows:
- For each head's own label: the AUC (one-vs-rest for the head's "up" class, e.g. BULL, HIGH risk,
  PROFIT) of the head's probability.
- Compared with a CAUSAL baseline: the same state computed from a TRAILING window of equal length
  ending at t (e.g. close vs a trailing 200-bar mean for regime). This is "the head only tells me
  the current state".
- Pass = AUC − baseline AUC ≥ 0.02 in both halves.

**Screen 2 (trade relevance)**, early half only: the R9 metric (gap ≥ 0.05, partial > 0, 5/8 fold
signs) of the head's probability on the 412 width-book trades, in the head's pre-declared direction:
- regime BULL +;
- risk HIGH −;
- momentum up +;
- flow up +;
- profit PROFIT +.

**A head earns an arm only if it passes BOTH screens.** Its use is fixed in the pre-registration by
head:

| Head | Arm |
|---|---|
| regime | entry gate on `cs_aro`'s width book: skip entries whose BEAR probability ≥ the training-tail 80th percentile |
| risk | stake sizer on `w_aro` (cs pattern, low risk → 1.333×) |
| momentum / flow | stake sizer (cs pattern) |
| profit | bail-out exit: close when the LOSS probability ≥ the tail 90th percentile after 2 bars |

The arms read per-fold head tables (the `MixSizerMixin` table pattern of 039) and are judged against
their unmodified book.

**Thresholds come from a held-out training tail, never in-sample.** The NNMT weights never fit the last
20% of each pair's training data (`TRAIN_DATA_SPLIT` 0.8), so the head probabilities on that tail are
out-of-sample. 039's `mix.py tables` step produced exactly this (the fold strategy run over its
validation tail). A `heads tail` step writes `user_data/data/nnmt_heads/tail_<M>_ah_fNN.parquet`, and
every head-arm percentile is taken from it.

## R11 — Regime-transition target (FR-010, FR-011)

**Causal regime state** `S_L(t)` for a trailing window L ending at t:
- `z = log(close_t / SMA_L(t)) / (σ_15m(t) · √L)`, where `σ_15m` is the trailing 2880-bar standard
  deviation of 1-bar log returns (min_periods 960);
- BULL if z ≥ +0.5, BEAR if z ≤ −0.5, else SIDEWAYS.

**Transition label at horizon H** (L = H). It uses future data, as a label may:
- **UP** if `S_H(t) ≠ BULL` and `S_H(t+H) = BULL`;
- **DOWN** if `S_H(t) ≠ BEAR` and `S_H(t+H) = BEAR`;
- else NONE.

**Grid**:
- H ∈ {96, 288, 672} bars (1, 3 and 7 days). 037 used 48/288/672; 96 is added for the 1-day case.
- Two levels: pair (the pair's close) and market (BTC's close, joined by date).

**Base rates** per fold and per half are reported. The deadband (0.5) and σ window are fixed in the
pre-registration before any label is built.

**Why this definition**: both ends are trailing states, so S_H(t+H) is computed from data up to
t+H only. Nothing comes from a centred window. The label says "the trend state flips to bull over the
next H", which a trader could have known at t+H.

**Stand-in model (screen)**:
- 037's per-fold HistGradientBoosting (`research/indicator_screen/screen_models.fit_fold` settings),
  one-vs-rest per direction, trained on each fold's training window with a purge of H bars.
- Inputs: the deployed model's inputs (the 24 indicators + raw8) plus BTC's raw8, and the causal
  z of the pair's and BTC's close for L ∈ {24, 72, 96, 168, 288, 672}. The trailing-trend baseline's
  information is inside the model, so the model must beat it with something else. They are read from 037's panel (`research/indicator_screen`
  `base:*` and `raw8:*` columns, the same causal-checked harvest). BTC's z is computed by
  `transition.py` itself. Rows: tradeable bars (volume > 0, `meta:data_ok`), hourly subsample for
  training.
- **Held-out tail for thresholds:** each fold's HGB is fit on its training window minus the last 90
  days (purge H before that tail). The tail's predictions set the R12 percentiles, and the test-window
  predictions come from the same fit.

**Baselines**:
- (a) the base rate (AUC 0.5);
- (b) the trailing-trend rule: the probability of UP rises with z of the shorter state
  `S_{H/4}(t)` (the trend has already started turning), scored as the AUC of that z directly.

**Screen pass** (per direction, H and level):
- AUC − max(baseline AUCs) ≥ 0.02 in both halves;
- at least 200 transition events (run starts) per half;
- a P&L proxy that holds in both halves: long-only for UP, the forward H-bar return of the top-decile
  scored bars net of 30 bp is > 0; for DOWN, the top decile's forward H-bar return is lower than the
  fold's mean by ≥ 30 bp.

**Truncation test (SC-005)**: recompute `S_L` and every stand-in input on 16 truncated histories
(`research/indicator_screen/causality.py` method), and require equality on the overlap
(RTOL 1e-9). This covers the label's construction too: `S_H(t+H)` must equal its value when history
is truncated at t+H.

## R12 — Transition trading arms and the NNMT head version

**Arms**: only for (direction, H, level) cells that pass. At most the best cell per direction
(highest late-half AUC margin) goes forward, so the arm count stays small. Each reads per-fold
prediction tables of the passing model.

| Arm | Use | Comparator |
|---|---|---|
| `tr_gate` | `cs_aro`'s book skips entries while P(DOWN) ≥ the tail 80th percentile | `cs_aro` |
| `tr_scale` | `cs_aro`'s stake × 1.333 when P(UP) ≥ the tail 80th percentile, × 0.667 when P(DOWN) ≥ it (capital-neutral to first order) | `cs_aro` |
| `tr_long` | standalone long entry when P(UP) ≥ the tail 99th percentile, shared exits (`st_g24_rev`'s stack) | PASS if the pooled pp is > 0 in both halves AND the bootstrap 95% CI of the per-trade mean pp (`evaluation/bootstrap.py`, seed 0) excludes 0; `st_g24_rev` reported |
| `tr_exit` | `cs_aro` closes an open trade when P(DOWN) ≥ the tail 90th percentile | `cs_aro` |

**NNMT head version (only if a stand-in cell passes)**:
- `<best>_trh` retargets the unused regime head with the R11 label for the passing (H, level),
  3 classes UP/NONE/DOWN, task weight 1.
- It is screened with the same rules. If it matches the stand-in (AUC within 0.01 in both halves),
  its tables replace the stand-in's in the arms.
- A seventh task is not added: `TASK_NAMES` is shared (Predictors/MLXClassifierMultiTask.py:44), and
  the regime head is otherwise idle.

**Deployment note**: arms read prediction tables (research instrument). Wiring a passing signal into
a live strategy is a follow-up, as for `cs_mix`.

## R13 — Pre-registrations, seeds and order

| File (`research/nnmt/`) | Covers | Committed before |
|---|---|---|
| `preregistration_models.json` | identity rule, `mt`/`mt_op`, variants, R7 | any 040 fold trains |
| `preregistration_framework.json` | feature arms, twin | the selected model's fw arms run |
| `preregistration_screens.json` | sizer screen, head screens (R9, R10 rules and arm uses) | the heads-trained model's tables exist |
| `preregistration_transition.json` | the R11 definition, grid, screens, R12 arms | any transition label is built |

**Seeds**: any PASS (or a selected model) is re-run on seed 2 with its comparator. A delta < 1 pp
needs seed 2 regardless (constitution gate 5).

## R14 — Compute and order

- **Training:** NNMT MLX fold training time is measured on f00 first and the totals extrapolated.
  Trainings:
  - `mt`: 15;
  - 4 variants: 60;
  - twin: 15;
  - heads-trained: 15;
  - head version (conditional): 15;
  - seed-2 reruns.
- **Stand-in HGB:** about 3 grid cells × 2 levels × 2 directions × 15 folds of small fits (minutes
  each, CPU).
- **Order:**
  1. R1 port prep and R2 keys, with tests;
  2. `mt` baseline, then port, then `mt_fw0` identity;
  3. `mt_op`;
  4. variants;
  5. R7 select;
  6. feature arms and twin;
  7. sizer screen and arm;
  8. heads-trained model, head screens and head arms;
  9. transition labels, truncation test, stand-in screen, arms and head version.
- **One trainer at a time**, as background chains with a bot liveness check. The transition stand-in
  fits count as a trainer too: they never run alongside an MLX chain (the 2026-09-16 OOM).
