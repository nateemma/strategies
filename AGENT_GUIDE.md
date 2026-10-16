# Agent Guide

This file orients an AI agent (or a new developer) working in this repository, possibly on
someone else's behalf. It says **where to find information first**, then the rules that keep
results trustworthy, then reference material on layout, classes and commands.

**The repository is the record.** The owner's own agent sessions keep private notes outside the
repo, which you will not have. Everything you need should be reachable from here; if you find a
gap, add a pointer to this file.

---

## Start here

1. **Silent failure modes** and **Operational rules** below. Each has cost real time.
2. **`.specify/memory/constitution.md`**: the research principles every experiment follows
   (evaluate in the deployment regime, persistence across eras, learnability is not edge,
   verify parameters are live, no leakage, ...). Specs check themselves against it.
3. **Current state** below: what has been tried, what was adopted, what the comparators are.
4. **Walk-forward folds** below, before running or judging any experiment.
5. **`docs/EXIT_AND_BACKTEST_MECHANICS.md`**, before changing entry, exit or stop logic.

---

## Where to find information

| Question | Look in |
|---|---|
| What has been tried, and what was the verdict? | `specs/NNN-*/report.md` (features 021–031); `research/<study>/report.md` or `findings.md` (studies, and features 032 onward); `regime/*.md` and `regime/VERDICT.md` (single-window studies, Sept 2026: the method is superseded, the mechanism findings still hold) |
| Why was it designed that way? | `specs/NNN-*/spec.md`, `plan.md`, `research.md` (decisions and rejected alternatives), `tasks.md` |
| What exactly was pre-registered for a study? | `research/<study>/preregistration.json` / `.md` (rules, cutoffs, amendments) |
| How do entries, exits, stops and freqtrade's backtester really behave? | `docs/EXIT_AND_BACKTEST_MECHANICS.md` |
| How should I examine a hyperparameter? | `docs/HYPERPARAMETER_STUDY.md` |
| How are experiments run and judged? | **Walk-forward folds** below; `evaluation/README.md`; `regime/WALKFORWARD_POWER.md` |
| How does the shared entry layer work (NNNC and SimpleTrigger)? | `utils/EntryFramework.py` docstring; `SimpleTrigger/README.txt`; `specs/029-nnnc-simpletrigger-merge/` |
| Where are a strategy's weights? | **Model keys** below; `Framework/ModelKey.py`; `scripts/model_index.py key <Class>` |
| Which labels exist, and which family does each imply? | `Framework/TrainingSignals.py` (`LabelMethod`); `utils/LabelFamilies.py` |
| Which entry triggers exist? | `research/outcomes/triggers.py` (the registry) |
| Market depth, liquidity, venues | `us_spot_market_study.md`; `regime/LIQUIDITY_STUDY.md`; `regime/EXCHANGE_VENUES.md` |
| GAN augmentation results | `docs/GAN_SCORECARD.md`; `GANs/` |
| Human-facing overview | `README.md` |

---

## Current state (2026-10-15)

- **Live:** only a dry-run of the momentum basket (`Basket/`, `config/config_mom_dryrun_15m.json`).
  Never kill it (see Operational rules).
- **NNNC defaults adopted by experiment:** stop-loss grace 24 h (was 6.5 h); the in-profit
  held-pair add (`HELD_PAIR_ADD = {"in_profit_only": True}`, feature 031). The ATR take-profit
  (`take_profit_atr_mult`, feature 028) exists as a default-off switch.
- **The comparator every model change must beat: the width book.** Within the gbb trigger's
  candidates, what the NNNC models learned is mostly a `bb_width` sort; a model-free book that
  takes gbb entries with `bb_width` above a per-fold cut (`NNNC/_width_book.py`, arms `w_aro`,
  `w_gbr`) matches or beats every NNNC model book (`research/model_calibration/findings.md`).
- **The model's measured value is as a sizer inside the width book.** `cs_aro` (ARO's p_buy as a
  capital-neutral 1.333×/0.667× stake on the width book's own trades) beats the width book by
  +8.5 pp on identical trades (PASS PROVISIONAL, `research/control_sizer/report.md`). Labels (034)
  and inputs (035, both32) have not improved ARO's ranking of those trades.
- **Primary model:** ARO (`NNNC_A_Raw_Open`, a plain LSTM on raw inputs; control arm
  `wf_aro_g24`). GBR (`NNNC_GbbRaw_Open`, control `fs_gbr_gr`) is the usual replication.
- **Results are PROVISIONAL.** The data has no distant era (see Walk-forward folds), so nothing
  has been recorded as a finding; confirmation routes are folds after 2026-09-07 or dry-run.

Recent features, newest first (each `specs/NNN-*` or the named `research/` directory):

| # | topic | outcome |
|---|---|---|
| 035 | new-input screen: 17 non-OHLCV / market-state inputs, plus both32 (ind24 + raw8) (`research/new_inputs`) | no input beats the control at S2 |
| — | control sizer: current ARO as a width-book stake sizer (`research/control_sizer`) | PASS (PROVISIONAL) |
| 034 | label-type screen: ARO trained on 7 other label types, ranked inside the width book (`research/label_screen`) | all 8 FAIL at S1 |
| 033 | width-relative label (`research/width_relative`) | all arms FAIL |
| 032 | train only on gbb-eligible rows (`research/candidate_training`) | FAIL at S2 |
| 031 | held-pair add (position adjustment on a fresh signal) | in-profit add adopted for NNNC |
| 030 | weights found by model key (hash of weight-shaping inputs) | done; byte-identical |
| 029 | NNNC and SimpleTrigger merged onto one entry framework | done; byte-identical |
| 028 | recovery-aware exits, judged on an unseen 2022 window | ATR take-profit adopted as a switch |
| 027 | target-aware exits | stopped at the feasibility gate |
| 026 | reversion regime gate | not adopted (fails on NNNC models) |
| 025 | family coexistence | nothing adopted |
| 024 | per-family exits (reversion vs momentum) | built; family exits exact |
| 023 | SimpleTrigger: the framework with no model | built; the study vehicle for 024–028 |
| 021–022 | label/exit coordination; per-fold guard tuning | see each `report.md` |

Also: `research/nnnc_framework/report.md` and `report_r2.md` (the framework study that adopted the
24 h grace), `research/nnnc_switches/report.md`, `research/st_grace/report.md`,
`research/held_pair/report.md`. Features 001–020 are in `specs/`; 007 built the fold harness.

---

## Environment

- **Shell**: `zsh` (macOS default). All scripts use `zsh`, not `bash`.
- **Python**: A virtualenv at `.venv` in the repo root. Activate with `source .venv/bin/activate`, or call `.venv/bin/python` directly. (Not conda.)
- **Secrets**: runs need the environment from `~/.freqtrade_env`: `set -a; source ~/.freqtrade_env; set +a`. A missing variable looks like a missing config value; don't edit configs to work around it.
- **Working directory**: All scripts in `scripts/` and freqtrade commands **must** be run from the **repo root** -- the directory containing `user_data/` (its location is machine-specific; this guide never assumes one) -- not from `scripts/` or any subdir. The scripts use relative paths (`strat_dir="user_data/strategies"`, config at `user_data/strategies/config/…`) and prepend `.` to `PYTHONPATH` so the in-tree `freqtrade` package resolves — both only work when the CWD is the repo root. Running from the wrong directory fails with a *misleading* `ModuleNotFoundError: No module named 'freqtrade'` + `config file not found` (see Troubleshooting). Research modules (`research.<study>.<tool>`) are run from `user_data/strategies` with `PYTHONPATH=.` and `../../.venv/bin/python -m ...`, as their docstrings show.
- **PYTHONPATH**: The scripts set this automatically. If running manually, export:
  ```
  export PYTHONPATH="$PWD/user_data/strategies:$PYTHONPATH"   # from the repo root
  ```
- **Tests**: from `user_data/strategies`:
  ```
  ../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests evaluation research/<study>/tests -q -p no:cacheprovider
  ```

---

## Silent failure modes — read this first

Every item here fails with **no error and a plausible-looking number**. They have each cost
real time in this repo.

> **Exits, stops, entries and freqtrade's backtest mechanics:** read
> `docs/EXIT_AND_BACKTEST_MECHANICS.md` before changing exit/stop/entry logic or writing any
> offline model of a backtest. It covers:
> - the per-candle exit order;
> - `custom_stoploss` being called at the candle HIGH;
> - stop rounding to the price tick;
> - per-candle ROI suppression by entry signals;
> - stake caps from liquidity;
> - what the trigger families have shown.

**Weights follow the model key, not the class name (feature 030).** A subclass that changes only
exits, stops, guards, gates, triggers or the threshold loads its parent's weights automatically.
A subclass that changes anything weight-shaping (label, targets, features, classifier, window,
seed) gets a new key and has no weights: outside walk-forward it then **trains inside its own
backtest**, fitting the evaluation window, and reports different numbers without error. Check
with `scripts/model_index.py key <Class>` before a backtest. Walk-forward runs are strict and
halt with `ModelMissing` instead. Details: **Model keys** below.

**`<Strategy>.json` beats the class, and a *missing* json is not neutral.**
Params load as `deep_merge_dicts(file_params, class_params)` — the file wins, the class only
fills gaps. So:
- An arm subclassing `NNNC_MLX` **without** its own json falls back to `NNNCStrategy`'s class
  params, which have every entry guard **open**. Completely different strategy, no warning.
- The json's `"strategy_name"` must match the class name, or freqtrade aborts with
  `Invalid parameter file provided`. Copying a json between arms without updating it fails
  this way.
- Give every arm its own json copied from the parent's, **unless** the parent has none
  (e.g. `NNMT_MLX`), in which case the arm should also have none — that is the matching config.

**Parameter dicts on a subclass must MERGE with the parent's.** A fresh `buy_params = {...}` (or
any dict-valued class setting) silently drops every parent key you did not repeat. This once ran
a strategy with its stop-loss grace disabled. Build from the parent: `{**Parent.x, "k": v}`.

**`test_strat.sh` exits 0 even when training crashed.** The only reliable check is whether the
weights exist (`scripts/model_index.py key <CLS>` prints whether they do).

**freqtrade `Parameter` objects are class-level singletons.** Loading two strategies in one
process contaminates the second — reading NNNC then NNMT reported NNMT's take-profit as NNNC's.
**Read configuration one strategy per process** (threshold picks run one fold per process for
the same reason). Values are only real after `ft_load_hyper_params()` (called inside
`ft_bot_start()`, which you must call when scoring a strategy outside backtesting); before that
you see raw defaults. `prediction_threshold` has 2 decimals on the base class, so a picked
threshold is rounded unless the class redeclares it with 4.

**`self.stoploss` and `trailing_*` do nothing for NN strategies.** `BaseNNStrategy` sets
`use_atr_adaptive_stoploss = True`, so `--stoploss` only changes the cosmetic
`initial_stop_loss_abs`. The live levers are the **class attributes**
`atr_stoploss_multiplier` / `_floor` / `_cap` (unreachable via `--set`, hence a subclass) and
`stoploss_grace_hours` / `stoploss_grace_level`, which set the worst-case trade size.

**Exit tags carry a TRAILING SPACE.** `BaseStrategy` builds them with
`exit_tag += "model_exit "` so several tags can concatenate, so the archive's `exit_reason` is
`"model_exit "`, not `"model_exit"`. A `Counter(...).get("model_exit")` therefore returns `None`
and the natural reading is "the model exit never fires". It fires 11 times in 99 trades on NNNC.
**Always `.strip()` an `exit_reason` before comparing it.**

**In backtest archives, `trailing_stop_loss` means the ATR stop *after it moved*.**
`trailing_stop` is `False` framework-wide, so attributing those exits to native trailing
mis-assigns a large share of trades. `stop_loss` is the initial/grace stop.

**Trade identity is `(pair, open_date, close_date)`, not `trades_hash`.** The record's
`trades_hash` also covers `profit_ratio`, so the same trades with a different fill (a held-pair
add, for one) hash differently. Compare trade sets on keys.

**The liquidity cap absorbs stake multipliers.** `BaseStrategy` caps a stake at what the candle
could fill, so a sizing change can be silently undone. Measure the realised stake ratio from the
trades, not the intended one.

---

## Walk-forward folds — how every experiment is run and judged

Every experiment since feature 007 runs on **anchored walk-forward folds**. The mechanics of the
harness are in `evaluation/README.md`; why 15 folds (the power measurement) is in
`regime/WALKFORWARD_POWER.md`; the early/late convention is in `evaluation/halves.py`. This
section collects the conventions that studies have converged on.

**Geometry.** Training always starts on 2021-01-01 and covers at least 730 days; each test fold
is the next 90 days. That gives 15 folds, `f00` (test 2023-01-01 → 2023-04-01) to `f14` (test
2026-06-14 → 2026-09-07). Each fold's model is trained on its own training window only, so
every test fold is out-of-sample. The universe is `config/config_wf.json` (SUI/USDT dropped: it
lists mid-history, and a pair missing from a training window silently prevents training).

**Plans.** `evaluation/folds/plan_<run_id>.json` holds the base strategy, a frozen params
snapshot (`params_snapshot`, with its SHA), and per fold the generated class name, the training
and test ranges and the `arm_tag`. An optional `thresholds: {fold_id: value}` sets
`prediction_threshold` per fold; `evaluation/pick_threshold.py` picks it on the fold's
**training** window at the incumbent's entry rate (0.014735 of bars), because a threshold is
not portable between architectures or labels (`evaluation/README.md`, last section).

**Running.** `evaluation/run_folds.py --plan <plan> [--folds f00,f01] [--seed N]` generates one
class per fold (`<Base>_fNN.py` plus its `.json`, in the family directory), trains it if its
model key has no weights, verifies the weights exist, and evaluates the fold through
`regime/run_arm.py --timerange`. Each fold writes an **arm record**
`regime/arms/<arm_tag>_<FNN>.json` (return, trades hash, model key, the archive it came from);
per-trade rows live only in the backtest archive (`user_data/backtest_results/*.zip`). Weights
are shared through the model key: variants of a model reuse its per-fold weights.

- `run_folds` is **resumable: it skips a fold whose record exists**. Move smoke-test records aside
  (rename to `*.smoke.json`) or the real run will silently reuse them.
- A regenerated fold class takes its threshold from the plan; a threshold that is only in an old
  fold class is lost.
- The first full NNNC run, training included, took about 2 hours; arms that reuse existing
  weights only backtest and are faster. Run one chain at a time.

**Judging.**
- `evaluation/pool.py` pools every fold's trades **before** any statistic.
- `evaluation/compare_pooled.py` gives the paired bootstrap verdict (pairs on
  `(fold_id, pair, open_date)`); `evaluation/risk.py` gives Sharpe and max-drawdown verdicts.
- `evaluation/halves.py` splits the folds into **early (f00–f07)** and **late (f08–f14)**. This
  split is the persistence test: there is no distant era in this data, so an effect must show in
  both halves, and any pass is PROVISIONAL.
- The usual per-arm rule is `judge_one` in `research/nnnc_switches/judge.py`: PASS = early gain
  > 0 AND late gain > 0 AND paired verdict not WORSENED AND risk verdicts not WORSENED;
  INCONCLUSIVE if the control has under 200 trades or the arm changed no fold's trades.
- Recent studies are **staged**: screen and choose on the early half, then read the late half
  once, for the winner only, so it stays unread confirmation. A margin under 1 pp needs a paired
  seed-2 repeat (`--seed 2`).
- 028 also judged model-free arms on three **2022 folds**, a window none of them had seen
  (`specs/028-recovery-aware-exits/`).

**Pre-registration.** A study commits `research/<study>/preregistration.json` (candidates,
parameters, cutoffs, decision rule) **before** any record it judges exists; its `judge.py`
refuses records older than that commit. Changes after the commit are listed as amendments, and
only before any P&L is read.

**Noise floor.** Pooled over 15 folds, effects of about 3 pp/yr are resolvable; 1–2 pp effects
are not and never will be on this much data. The harness is not licence to tune parameters.

---

## Single-window backtests (historical method)

Before the fold harness, changes were judged on pinned windows (`regime/regimedata.py`) with
`regime/run_arm.py`, training range `20220101-20250831`: W1 `20250901-20260831` (then the
acceptance window), W2 `20240601-20250531` and W3 `20230101-20231231` (in-sample). The
`regime/*.md` studies use this method. `run_arm.py` still evaluates every fold (now with
`--timerange`), and it remains useful for a quick look at a large effect, but do not judge a
change on it.

```zsh
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/run_arm.py \
  --strategy NNNC_MLX --window W1 --tag baseline
```

Rules that still apply to any result:
- Confirm `returncode == 0`. A failed or stale record is not evidence.
- **Identical trades to the control ⇒ INCONCLUSIVE**, not refuted — the change never reached
  the trades.
- **Zero trades ⇒ inconclusive**, never a zero return.
- **Never accept on a classifier metric.** val_mcc and P&L move independently here — it has been
  measured repeatedly (higher MCC with identical trades; +0.155 MCC for no P&L change).

### The single-window noise floor — why the folds exist

> **Method guide:** `docs/HYPERPARAMETER_STUDY.md` — how to examine a hyperparameter
> before searching it: measure its span against this floor, separate a mechanism from its
> knobs, calibrate a fast evaluator against one real backtest, and run your gates on the
> incumbent. Six of ten exit/guard parameters turn out to be inert.

Bootstrapping the acceptance window's own per-trade P&L (NNNC W1, 99 trades, +9.831%):

```
95% CI on the W1 return statistic : [2.383%, 17.649%]
half-width (the noise floor)      : 7.633pp     (6.1pp when arms are paired)
sampling SD                       : 3.893pp
compare.py's threshold            : 1.000pp     <- ~7.6x too lenient
```

**A whole parameter sweep fits inside that band** (the rvol sweep spans 4.93pp), so the "best" rung
of a sweep is a ranking of noise. Full sweeps of two parameters across all three windows disagreed
on the optimum every time, with contradictory rank correlations between windows (rvol W1~W2
**-0.46**, grace W1~W2 **+0.71**), none significant. **The optima are not trackable.**

Consequences:

- **Do not tune parameters on ~100-trade windows.** Structural changes (a mechanism fix, a label
  redefinition, an execution-model correction) can clear the floor. Parameter tweaks cannot.
- **Before believing any delta, bootstrap it**, paired on trade keys. A +1.86pp delta that
  `compare.py` called IMPROVED has a 95% CI of `[-4.29, +7.93]pp`, p=0.54.
- **A W1/W2 disagreement is not evidence of regime dependence** until the effect is resolvable.
- **Annualise before comparing floors.** A pooled CI spans ~3.7 years; a single window one.

---

## Operational rules

- **Run everything sequentially: one training or backtest chain at a time.** Two concurrent jobs
  put load 60 on 14 cores and finished slower than serialising; parallel trainers have caused
  out-of-memory kills and Metal faults; concurrent runs can cross-contaminate the arm archive.
  Check `ps` before starting one.
- **Never `pkill -9 -f freqtrade`** — that pattern matches the live `freqtrade trade` dry run.
  Use `pkill -9 -f 'freqtrade backtesting'`, or kill by PID after checking `ps`. `TaskStop` does
  not kill process trees.
- **Only the nested `user_data/strategies` repo is ours**; the outer freqtrade checkout is
  upstream. Commit directly to `master` there.
- **Never `git add -A` / `git add .`** — files are edited concurrently outside the agent session.
  Stage only the files your change touched, and check `git status` first.
- **`Framework/` stays strategy-agnostic.** Strategy-specific behaviour registers at runtime
  from the strategy; no pair-specific code anywhere.
- **New switches default OFF and byte-identical when off**; adopting one as a default is the
  owner's decision, not the experiment's.
- **Never run `setup.sh -i`** to upgrade freqtrade (it wipes `.venv`); use
  `scripts/upgrade_freqtrade.sh`.

---

## Directory Map

```
user_data/strategies/
├── Framework/           ← Universal base classes + mixins
│   ├── BaseStrategy.py  ← Root base class for ALL strategies (ROI, stoploss,
│   │                      bot_start lifecycle, custom_exit, guards)
│   ├── BaseNNStrategy.py← NN base (inherits the mixins below + BaseStrategy):
│   │                      lifecycle hooks, label generation, prediction,
│   │                      entry/exit wiring
│   ├── StrategyDiagnostics.py ← mixin: assessment/probability/correlation
│   │                      printing (pure diagnostics; mixed into BaseStrategy)
│   ├── FeatureNormalizer.py   ← mixin: feature lists (include_list /
│   │                      pre_normalized_columns), scaler + PCA state/methods
│   ├── TrainingEngine.py      ← mixin: training pipeline — prepare/class-weights/
│   │                      train_model, markov, GAN augmentation, maybe_train
│   ├── TrainingSignals.py  ← Future-aware label generation
│   └── CreateScalers.py ← Run once to generate normalization scalers
├── utils/               ← Shared utility code, incl. EntryFramework.py (the
│                          entry layer shared by NNNC and SimpleTrigger, feature 029)
│   ├── DataframeUtils.py
│   ├── DataframePopulator.py     ← Adds all technical indicators to a dataframe
│   ├── ClassifierKeras*.py        ← Keras classifier implementations
│   ├── ClassifierMLX*.py          ← Apple MLX classifier implementations
│   ├── ClassifierMLXMultiTask.py  ← MLX multi-task base (focal loss + grad clipping)
│   ├── ClassifierSklearn.py       ← sklearn classifier implementations
│   ├── Wavelets.py
│   ├── Forecasters.py
│   └── ...
├── NNNC/                ← N-ary (trinary) Classification strategies + NNNClassifier;
│                          also generated fold classes (<Base>_fNN.py/.json) and the
│                          width-book / sizer mixins (_width_book.py, _width_sizer.py)
├── SimpleTrigger/       ← the framework with no model (triggers only): study vehicle
├── Basket/              ← Momentum basket (the live dry-run)
├── NNMT/                ← Multi-Task strategies + NNMTClassifier (TF)
│                          + NNMTClassifierMLX (Apple Silicon)
├── NNPredict/           ← Regression strategies (continuous future_gain
│                          target → rolling-quantile signal). Keras / MLX /
│                          Ridge regressor backends.
├── Anomaly/             ← Anomaly Detection strategies (autoencoder + GANomaly)
├── Sklearn/             ← sklearn classifier strategies (RandomForest, XGBoost, …)
│                          inherit from BaseNNStrategy via SklearnStrategy
├── GANs/                ← GAN implementations + GANInterface + GANBackend ABC
│   ├── GANInterface.py  ← Thin facade: fit/generate/save/load
│   ├── GANBackend.py    ← Abstract base + registry (resolve_backend, fit/load_with_fallback)
│   ├── backends/        ← Concrete backends, one file per type/backend pair
│   ├── df_*_gp.py       ← TensorFlow trainer implementations
│   ├── df_*_mlx.py      ← MLX trainer implementations
│   ├── Create*GAN*.py   ← Strategy classes you run under freqtrade backtesting
│   │                      to train + save a GAN
│   └── tests/           ← Contract + robustness tests
├── MLX/                 ← Apple MLX neural net components (Mamba, etc.)
├── TSPredict/           ← Time-series/wavelet-based strategies
├── SimpleStrategies/    ← Single-indicator strategies (no ML)
├── Debug/               ← Debug/visualisation utilities
├── hyperopts/           ← Custom hyperopt loss functions
├── config/              ← Exchange-specific config files
├── saved_data/          ← Trained model files: models/<prefix>__<hash>/ by model
│                          key (feature 030); older ones by strategy name
│                          + GANs/<gan_type>/ subdirs for every GAN type
│                          (GANs_PCA/<gan_type>/ for PCA-reduced strategies)
├── scripts/             ← Shell scripts for all workflow tasks; model_index.py
├── evaluation/          ← The walk-forward harness: fold plans (folds/), run_folds,
│                          pool, compare_pooled, risk, halves, pick_threshold
├── research/            ← One directory per study: code, tests, pre-registration,
│                          report.md / findings.md
├── regime/              ← run_arm.py (evaluates one arm), arms/ (arm records), and the
│                          Sept 2026 single-window study write-ups (*.md)
├── specs/               ← spec-kit features NNN-*: spec, plan, research, tasks, report
├── docs/                ← Cross-cutting write-ups (mechanics, hyperparameter method, GANs)
├── .specify/memory/constitution.md ← research principles
├── archived/            ← Old/abandoned strategies (reference only)
└── reference/           ← External strategies for learning
```

> _**Historical note**_: an older `NeuralNets/` directory used to contain a
> separate git repo holding the NN base class (`NNStrategy`) and the scaler
> storage.  That directory is now deprecated — its contents have been folded
> into `Framework/` (base classes) and the top-level `saved_data/` (scalers).
> Older docs / older branches still reference `NeuralNets/`; those references
> are stale.

---

## Class Hierarchy

```
BaseStrategy(StrategyDiagnostics, IStrategy)            (Framework/BaseStrategy.py)
├── BaseNNStrategy(TrainingEngine, FeatureNormalizer, BaseStrategy)  ← composes the ML mixins
│   ├── NNNCStrategy (NNNC/NNNCStrategy.py)
│   │   └── NNNC_CGP, NNNC_CGP_LSTM2, NNNC_CGP_MLX_*, ... (concrete strategies)
│   ├── NNMTStrategy (NNMT/NNMTStrategy.py)
│   │   └── NNMT_WGAN, NNMT_WGAN_MLX, NNMT_CGP, ... (concrete strategies)
│   ├── NNPredictStrategy (NNPredict/NNPredictStrategy.py)  ← regression family
│   │   └── NNPredict_LSTM, NNPredict_MLX_LSTM, NNPredict_Ridge
│   ├── NNAnomalyStrategy (Anomaly/NNAnomalyStrategy.py)
│   └── SklearnStrategy (Sklearn/SklearnStrategy.py)
│       └── Skl_RandomForest, Skl_XGBoost, Skl_RandomForest_WGAN, ...
├── SimpleStrategy (SimpleStrategies/SimpleStrategy.py)
│   └── AO, BBBreakout, EMACross, ... (each in own file)
└── TSPredict (TSPredict/TSPredict.py)
    └── TS_Wavelet_DWT, TS_Coeff_FFT, ... (concrete strategies)
```

There is no separate `NNStrategy` class — `BaseNNStrategy` is the ML
pipeline base and the per-family bases (NNNC/NNMT/Anomaly/Sklearn)
inherit directly from it.  Older docs may still reference `NNStrategy`;
that's stale.

The ML pipeline is decomposed into **mixins** (composed in MRO order, mixin
first) rather than a single monolith — each is byte-identical-relocated, not
rewritten:
- **`StrategyDiagnostics`** (into `BaseStrategy`) — assessment / probability /
  correlation printing. Pure diagnostics; no effect on training or trading.
- **`FeatureNormalizer`** (into `BaseNNStrategy`) — feature-list selection
  (`include_list` / `pre_normalized_columns` / `one_hot_columns`) + scaler/PCA
  state and the methods that consume them (`rolling_dataframe_normalise`,
  `normalise_for_gan`, `clean_for_tensor`, `apply_pca`, `get_normalized_size`).
  Generic engine — list *contents* are overridable class attrs the families set.
- **`TrainingEngine`** (into `BaseNNStrategy`) — `prepare_training_data`,
  `get_training_class_weights`, `train_model`, markov helpers, GAN augmentation
  (`enhance_training_data` / `preprocess_training_data` + their helpers), and
  `maybe_train` (the training trigger `populate_indicators` delegates to).

The mixins assume composition (they call collaborators via `self`, e.g.
`TrainingEngine` calls `FeatureNormalizer.scale_dataframe`) and introduce no
literal references to their host class — consistent with the one-way dependency
rule below.

### BaseStrategy responsibilities
- ROI table, stoploss, trailing stop config
- `bot_start()` — freqtrade's one-time-init hook.  Handles environment
  setup, hyperopt-parameter printing, and shared utility instantiation
  (`DataframeUtils`, `DataframePopulator`).  Subclasses overriding this
  MUST call `super().bot_start(**kwargs)`.
- `iteration_init()` — runs at the start of each `populate_indicators()`
  cycle.  Now slim: just per-iteration scaler reset.
- `custom_exit()` — most actual sells happen here, not in `populate_exit_trend`
- `custom_stoploss()`
- `confirm_trade_entry()` / `confirm_trade_exit()`
- Guard conditions (disable trading in bad market conditions)
- Hyperopt parameters: guards, prediction threshold
- `populate_indicators()` calls `DataframePopulator` to add all technical indicators
- Assessment / distribution / correlation printing — via the `StrategyDiagnostics`
  mixin (pure diagnostics; safe to ignore when tracing trading behaviour)

### BaseNNStrategy responsibilities (in addition to BaseStrategy's)
`BaseNNStrategy` itself keeps the freqtrade lifecycle (`bot_start`/`iteration_init`/
`populate_indicators`), label generation (`get_training_labels` + `TrainingSignals`
+ MASTER thresholds), prediction, and entry/exit wiring. The rest is provided by
its composed mixins:
- **Classifier construction** via `get_classifier_type()` + `get_classifier()`
  (family-specific; called from `TrainingEngine.maybe_train`).
- **Normalization / feature lists** → `FeatureNormalizer`.
- **Training + GAN augmentation + class weights** → `TrainingEngine`. GAN
  augmentation is a single dispatcher: `enhance_training_data` inspects `gan_type`
  and the label shape (ndarray vs dict), validates the saved GAN's metadata
  against the strategy's current config, and routes to
  `GANs.balance.balance_single_task` / `balance_multi_task`. Concrete strategies
  declare `gan_type` (and optionally `gan_target_ratio`, `gan_run_diagnostics`,
  `gan_passthrough_columns`, `gan_augment_seed`) — they don't see GAN-type-specific
  code. Multi-task 3-D pipelines (e.g. `NNMT_WGAN`) turn off the 2-D dispatcher
  with `gan_augment = False` and run their own `preprocess_training_data`.
- **Train / save / load lifecycle** — `populate_indicators()` delegates to
  `TrainingEngine.maybe_train()` (classifier setup + multi-pair aggregation +
  training trigger).

> **Reproducibility:** GAN augmentation seeds its sampler from `gan_augment_seed`
> (default 42). Seeded runs are byte-reproducible **across separate processes**
> (each cold-starts identically), but NOT across repeated augmentations *within*
> one process — the AE-filter's lazy model load draws `mx.random` on first call,
> polluting the seeded stream. Validate GAN-path changes with a separate-process
> augmentation diff, not an in-process golden. Set `gan_augment_seed = None` for
> non-deterministic augmentation.

### Adding a new strategy (NN family)
1. Create a new `.py` file in the appropriate family directory (e.g., `NNNC/`, `NNMT/`, `NNPredict/`, `Anomaly/`, `Sklearn/`)
2. Inherit from the appropriate family base class (e.g., `NNNCStrategy`, `NNMTStrategy`, `NNPredictStrategy`, `SklearnStrategy`)
3. Override `get_classifier_type()` and `get_classifier()` to return your model (or regressor, for `NNPredictStrategy` subclasses)
4. Optionally override `add_strategy_indicators()`, `get_custom_training_data()`, etc.
5. Run backtest over a long period to train and save the model

### Adding a new SimpleStrategy
For framework studies (entries, exits, stops, guards with no model) use `SimpleTrigger/`
instead; it shares NNNC's entry framework. `SimpleStrategies/` holds older single-indicator
strategies.

1. Create a new `.py` file in `SimpleStrategies/`
2. Inherit from `SimpleStrategy`
3. Override `populate_entry_trend()` and (optionally) `populate_exit_trend()`
4. No training needed

---

## Config Files

All config files live in `user_data/strategies/config/`.

| File                              | Purpose                                         |
|-----------------------------------|-------------------------------------------------|
| `config_binanceus.json`           | Main backtest/hyperopt config (static pairlist) |
| `config_binanceus_short.json`     | Futures/short trading config                    |
| `config_binanceus_download.json`  | Download config (may use VolumePairlist)         |
| `config_binanceus_train.json`     | Long-range config for training NN models        |
| `config_binanceus_leveraged.json` | Leveraged trading config                        |
| `config_wf.json`                  | Walk-forward folds (SUI/USDT dropped)           |
| `config_mom_dryrun_15m.json`      | The live momentum-basket dry run                |

Config files used with scripts are referenced by exchange name. The scripts resolve the path automatically.

---

## Common Commands

### 1. Download data
```zsh
# Download last 180 days for all exchanges
zsh user_data/strategies/scripts/download.sh

# Specific exchange
zsh user_data/strategies/scripts/download.sh binanceus

# Futures/short data
zsh user_data/strategies/scripts/download.sh --short binanceus

# Manual command
freqtrade download-data --timerange=20230101- \
  -c user_data/strategies/config/config_binanceus.json \
  -t 5m 15m 1h 1d
```

### 2. Backtest a single strategy
```zsh
zsh user_data/strategies/scripts/test_strat.sh NNNC NNNC_CGP

# Manual equivalent
freqtrade backtesting \
  -c user_data/strategies/config/config_binanceus.json \
  --strategy-path user_data/strategies/NNNC \
  --strategy NNNC_CGP \
  --timerange=20230101-20231231
```

### 3. Backtest a group of strategies (with wildcards)
```zsh
zsh user_data/strategies/scripts/test_group.sh NNNC "NNNC_CGP_MLX*"
zsh user_data/strategies/scripts/test_group.sh TSPredict "TS_Wavelet*"
```

### 4. Hyperopt a strategy
```zsh
zsh user_data/strategies/scripts/hyp_strat.sh NNNC NNNC_CGP

# With custom loss and spaces
zsh user_data/strategies/scripts/hyp_strat.sh \
  -l ExpectancyHyperOptLoss \
  -s "buy sell roi" \
  binanceus NNNC_CGP

# Manual equivalent
freqtrade hyperopt \
  -c user_data/strategies/config/config_binanceus.json \
  --strategy-path user_data/strategies/NNNC \
  --strategy NNNC_CGP \
  --spaces buy sell roi \
  --hyperopt-loss ExpectancyHyperOptLoss \
  --timerange=20230101-20231231
```

### 5. Check for lookahead bias
```zsh
zsh user_data/strategies/scripts/check_bias.sh NNNC NNNC_CGP
```

### 6. Plot a strategy
```zsh
zsh user_data/strategies/scripts/plot_strat.sh NNNC NNNC_CGP BTC/USDT

# Output: user_data/plot/freqtrade-plot-BTC_USDT-5m.html
```

### 7. Create scalers (one-time setup for NN strategies)
```zsh
freqtrade backtesting \
  -c user_data/strategies/config/config_binanceus.json \
  --strategy-path user_data/strategies/Framework \
  --strategy CreateScalers \
  --timerange=20220101-
```
Scalers are saved to `user_data/strategies/saved_data/`.

### 8. Retrain a neural net model
Use `-f`: it deletes the model at the strategy's RESOLVED model-key path (see "Model keys" below),
retrains, and fails (exit 3) if no model was written:
```zsh
zsh user_data/strategies/scripts/test_strat.sh -f -t 20220101- NNNC NNNC_CGP
```

### Model keys (feature 030) — where weights live
NN weights are found by a **model key**, not by class name: a hash of everything that shapes the
weights (`Framework/ModelKey.py`):
- the declared training attributes (labels and targets, features, classifier, GAN settings);
- which class defines each training-path method;
- the scaler, PCA and GAN files, by content;
- the training window, whitelist, timeframe, startup candles and seed.

What follows:
- **Variants share their model's weights automatically.** A subclass that changes only exits,
  stops, guards, gates, triggers or the prediction threshold resolves to the same key and loads
  the same weights, with no redirect. `BorrowedModelMixin` is retired; importing it raises.
- **A changed weight-affecting input gets a new key.** A changed label target, feature list,
  classifier, scaler file, window or seed can never load another model's weights.
- **Walk-forward evaluation is strict.** `evaluate_fold` sets `FT_MODEL_EVAL_STRICT=1`, so a
  missing model halts with `ModelMissing` naming the key. Other runs may train, but only into
  the key's own location `saved_data/models/<prefix>__<hash>/`, with a `manifest.json`.
- **Generated fold classes pin their key.** They carry `MODEL_TRAIN_TIMERANGE` and `MODEL_SEED`,
  written by `run_folds`.
- **Training inputs your run changes.** If your run changes a training input that the weights
  did not see (e.g. a 4096-candle warm-up on weights trained with 64: warm-up IS a training input,
  because freqtrade trains before trimming startup bars), declare
  `MODEL_TRAINED_WITH = {"startup_candle_count": 64}`.
- **New class attributes must be classified.** Add a new class attribute or training-path method
  to the owning class's `MODEL_KEY_ATTRIBUTES` / `MODEL_KEY_METHODS` if it shapes the weights,
  else to `MODEL_KEY_IGNORED` / `MODEL_KEY_METHODS_IGNORED`. `Framework/tests/test_model_key_audit.py`
  fails on any unclassified name defined in more than one class.
- **Bump `MODEL_CODE_VERSION`** (in `BaseNNStrategy`) when an edit to training code changes what
  training produces. Otherwise old weights would be served to new code.
- **Pre-030 models are indexed, not moved.**
  - The walk-forward sets got keys rebuilt from their plans.
  - Single-model directories are `legacy:<Class>`: they load only for that class.
  - The ~96 classes with a class-level `get_model_path` override keep it. They are recorded in
    `specs/030-model-fingerprint-naming/legacy_redirects.json`, which a test keeps in sync.
- **Tools:**
  - `scripts/model_index.py key <Class> [--plan P --fold fNN --seed N]` prints the key, its
    location and whether weights exist;
  - `path <Class>` prints the weight path;
  - `migrate [--dry-run]` re-indexes;
  - `verify-bytes` snapshots or compares every weight file;
  - `audit` runs the classification test.
- **Arm records** carry `model_key` (hash, prefix, kind) for the model the backtest loaded.

### 9. Dry run
```zsh
zsh user_data/strategies/scripts/dryrun_strat.sh NNNC NNNC_CGP

# With port (for multiple simultaneous strategies)
zsh user_data/strategies/scripts/dryrun_strat.sh -p 8081 NNNC NNNC_CGP
```

### 10. Live run
```zsh
zsh user_data/strategies/scripts/run_strat.sh NNNC NNNC_CGP
```

---

## BaseNNStrategy Pipeline

Understanding the data flow is critical for debugging or extending NN strategies.  The pipeline lives in `Framework/BaseNNStrategy.py` and its mixins (`FeatureNormalizer` = normalization, `TrainingEngine` = training/GAN), and is shared across NNNC, NNMT, Anomaly, and Sklearn family bases.

### `bot_start(**kwargs)` (one-time, called once per backtest/dry-run/live)
1. Calls `super().bot_start()` (BaseStrategy: banner, environment, helpers).
2. Configures TF/MLX device visibility for `util_no_exchange` runs.
3. Loads MASTER thresholds from saved GAN metadata if present (so the strategy
   always uses the same `MIN_BUY_GAIN_THRESHOLD` / `MIN_SELL_LOSS_THRESHOLD` /
   `TRAINING_TYPE` the GAN was trained with).
4. Falls back to `buy_params` / `sell_params` overrides if no GAN metadata.

### `populate_indicators(dataframe, metadata)` (per pair, per iteration)
1. Checks that scalers exist in `saved_data/`.
2. Calls `DataframePopulator` to add all technical indicators.
3. Adds training labels via `get_training_labels` (`TrainingSignals`).
4. Delegates to `TrainingEngine.maybe_train()`: sets up the classifier, and
   (if no saved model) collects dataframes across the whitelist, trains via
   `train_model` — which normalizes (`FeatureNormalizer`), GAN-augments, computes
   class weights, and fits — then saves. If a model exists, training is skipped.

### `get_predictions(dataframe)`
1. Normalizes the dataframe
2. Converts to sequences of shape `[batch, seq_len, num_features]`
3. Returns probability predictions from the model
4. Applies threshold to get discrete class (sell/hold/buy)

### `populate_entry_trend` / `populate_exit_trend`
- Applies guard conditions
- Combines model predictions with additional technical filters

### Key configuration flags (class-level)
```python
aggregate_pairs = True    # train on all pairs combined
use_gan = False           # augment with GAN-generated data
seq_len = 8               # input sequence length
num_epochs = 100          # training epochs
batch_size = 1024
```

---

## Feature Engineering Rules (NN strategies)

NN strategies need **pair-agnostic** features — indicators that have consistent ranges across all pairs. The following are prohibited:
- Raw price (open, close, high, low)
- Raw volume
- Any indicator directly proportional to price

Use instead:
- Oscillators (RSI, MFI, CMF — already bounded 0-100 or -1 to 1)
- Z-score normalized indicators
- Percentage changes (returns)
- Ratio-based indicators

The `DataframePopulator` class already handles all of this — it adds a standard set of pre-approved indicators. Custom indicators must follow the same rules.

---

## Hyperopt Loss Functions

Copy from `user_data/strategies/hyperopts/` to `user_data/hyperopts/` before using.

| Function                      | Best for                                            |
|-------------------------------|-----------------------------------------------------|
| `ExpectancyHyperOptLoss`      | General use — robust across datasets (recommended)  |
| `OnlyExpectancyHyperOptLoss`  | When you want pure expectancy, nothing else         |
| `WeightedProfitHyperOptLoss`  | Maximising total profit                             |
| `QuickProfitHyperOptLoss`     | Maximising profit with short-duration trades        |
| `WinHyperOptLoss`             | Maximising win rate                                 |
| `MarketHyperOptLoss`          | Market-condition-adjusted win rate                  |
| `MedianProfitHyperOptLoss`    | Robust profit (less sensitive to outliers)          |
| `PEDHyperOptLoss`             | Balanced: Profit + Expectancy + Duration            |

---

## Troubleshooting

### Changed parameters aren't taking effect
Check the `.json` file beside the `.py` file (e.g., `NNNC_CGP.json`). Hyperopt results in this
file **override** Python defaults (`deep_merge_dicts(file_params, class_params)` — file wins).
Three further ways a parameter silently does nothing:
- It is a **class attribute**, not a `Parameter` (e.g. `atr_stoploss_multiplier`) — `--set` cannot
  reach it; subclass instead.
- It is **inert by construction** — `self.stoploss` and `trailing_*` under
  `use_atr_adaptive_stoploss`, or a guard whose threshold is looser than the label that produced
  the bars (`entry_guard_threshold` was inert for exactly this reason).
- The label and the guard filter **the same quantity**, so the stricter one binds and the other
  never fires.
See "Silent failure modes" at the top.

### Model not retraining
Use `test_strat.sh -f` (it deletes the strategy's resolved model-key path). To see where a
strategy's weights are: `../../.venv/bin/python scripts/model_index.py key <StrategyName>`.

### `ImportError` or `ModuleNotFoundError`
First check your **working directory** — `ModuleNotFoundError: No module named 'freqtrade'`
(usually alongside `config file not found: user_data/strategies/config/config.json`) when
running a `scripts/*.sh` almost always means you launched it from the wrong CWD. The scripts
use relative paths and a `.`-prefixed `PYTHONPATH`, so they only work from the project root:
```zsh
cd /path/to/freqtrade        # the directory containing user_data/
zsh user_data/strategies/scripts/test_strat.sh NNNC NNNC_CGP
```
If you're running Python/freqtrade manually, also make sure PYTHONPATH includes the
strategies directory:
```zsh
export PYTHONPATH="$PWD/user_data/strategies:$PYTHONPATH"   # from the repo root
```

### Suspiciously high backtest results (100%+)
Almost certainly lookahead bias. Check that:
- No global `mean()`/`min()`/`max()` applied to the full column
- All rolling operations use `min_periods` and only look backwards
- TA-lib functions are used instead of manual rolling where possible
- Run `check_bias.sh` to confirm

### Scalers missing
Run `CreateScalers` (see "Create scalers" above). NN strategies will fail at startup without them.

---

## Adding a New Hyperopt Loss Function

1. Create a new file in `user_data/strategies/hyperopts/` inheriting from `IHyperOptLoss`
2. Implement `hyperopt_loss_function(results, trade_count, min_date, max_date, config, processed, backtest_stats, *args, **kwargs) -> float`
3. Return a float where **lower is better**
4. Copy the file to `user_data/hyperopts/` to make it available to freqtrade
5. Reference it with `--hyperopt-loss <ClassName>` or `-l <ClassName>`

---

## Adding to / extending the GAN system

There are two distinct extension cases:

### Case A — new variant of an existing GAN type

Most additions are this case (e.g. a CTAB-GAN+ trained on a different feature
set, or a WGAN with a different augmentation ratio).

1. Create a new builder script in `GANs/` (e.g. `CreateMyGAN.py`).
2. Inherit from `CreateGAN` (single-task) or `CreateMTGAN` (multi-task) and
   set `gan_type = GANType.X` plus any per-class config overrides.  The
   pre-existing classes (`CreateWGAN`, `CreateCtabGanPlus`, etc.) are
   thin shims over these two unified bases — copy one of them as a template.
3. Run the new strategy via backtesting on a long timerange to train and save.
4. The saved model goes to `saved_data/<StrategyName>/GANs/<gan_type>/`
   (or `GANs_PCA/<gan_type>/` for PCA-reduced strategies) — the layout
   is centralised in `GANs/paths.py::gan_save_path`, so subclasses
   don't pick a directory name.
5. Strategies consume the GAN by setting `gan_type = GANType.X` on the
   class.  `BaseNNStrategy.enhance_training_data` then loads the model
   via `GANInterface`, validates its saved metadata against the
   strategy's current thresholds (raising
   `GANMetadataMismatchError` on drift), and dispatches class balancing
   through `GANs.balance.balance_single_task` /
   `balance_multi_task`.  Override `_gan_expected_metadata` if your
   strategy needs to validate extra keys on top of the default
   thresholds + training_type.

### Case B — genuinely new GAN type (new GANType enum value)

Rare.  Add when the existing types can't capture the new behaviour
(e.g. a different label modality or a fundamentally different conditioning).

> **Reference:** TabDDPM (`GANType.TAB_DDPM`) was added as a Case B
> follow-up — see `docs/superpowers/specs/2026-05-11-tabddpm-design.md`
> and `docs/superpowers/plans/2026-05-11-tabddpm-implementation.md` for
> a concrete worked example. It's MLX-only and continuous-only.

1. Add the new enum entry to `GANType` (`GANs/GANType.py`).
2. Create the trainer/model class(es) in `GANs/df_<name>_*.py` (TF and/or
   MLX, following the existing `df_wgan_*` and `df_ctab_*` patterns).
3. Create a backend class in `GANs/backends/<name>.py` subclassing
   `GANBackend` and decorate with `@register_backend`.  Implement
   `fit / generate / save / load` and `is_available()`.
4. Add `from . import <name>  # noqa: F401` to `GANs/backends/__init__.py`
   so it registers at import time.
5. Add a `_DEFAULTS` entry in `GANInterface` if your type has trainer-specific
   defaults that callers shouldn't be forced to know about.
6. Add the new type to `_BACKEND_MIGRATED` in `GANInterface.py`.
7. Cover it with the contract tests in `GANs/tests/`:
   `test_gan_metadata_roundtrip.py` (what gets persisted),
   `test_gan_output_contracts.py` (shape/dtype/finiteness),
   and (gated) `test_gan_robustness.py`.

---

## Python Conventions

These apply to all code under `user_data/strategies/`. When in doubt, match the
surrounding file; consistency within a module beats global purity.

- **Formatting / linting**: `ruff` is the source of truth (formatter + linter).
  100-character lines (the repo-root `pyproject.toml`). Run `ruff format` and `ruff check --fix` before declaring
  a change done. Don't hand-format what the formatter owns.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for
  classes, `UPPER_CASE` for module-level constants. Strategy class names are
  load-bearing — they're referenced by configs, `saved_data/` directories, and
  the `test_*.sh` scripts — so **never rename a strategy class as a side effect**
  of a refactor. If a rename is genuinely needed, list it explicitly and update
  the matching `saved_data/<Name>/` dir and any `<Name>.json`.
- **Type hints** on all new/edited function signatures (params and return).
  Prefer `from __future__ import annotations` at module top so forward
  references and `X | None` work without runtime cost.
- **Docstrings** on public classes and methods — one line on *what* and *why*,
  not a restatement of the signature. Document the contract (shapes, units,
  side effects), especially for anything touching the NN pipeline.
- **No secrets in code.** Exchange keys, API tokens, etc. live in config/env,
  never in `.py` files, and configs with secrets stay out of git.
- **Imports**: absolute within the strategies tree (the directory is on
  `PYTHONPATH`). No wildcard imports except the deliberate
  `from . import <name>  # noqa: F401` registry pattern in `GANs/backends/`.
- **Lookahead safety is a hard rule, not a style preference.** No global
  `mean/min/max` over a full column; rolling ops use `min_periods` and only look
  backwards (see Troubleshooting). A "cleaner" refactor that introduces lookahead
  bias is a regression — verify with `check_bias.sh`.

---

## Object-Oriented Design Rules

The hierarchy here grew over a long time and has drifted. These rules are the
target state; the refactor work should move code toward them.

### Dependency direction is one-way: bases never know their subclasses
- **A base class must never reference a subclass — not in code, not in type
  hints, not even in comments or docstrings.** No imports from a subclass module,
  no `isinstance(self, NNNC_CGP_MLX)`, no "used by NNMT_WGAN" notes in
  `BaseNNStrategy`. If a base needs subclass-specific behavior, that's a signal
  to invert the dependency: expose a hook/abstract method the subclass overrides,
  or a registry the subclass registers into (the `@register_backend` pattern in
  `GANs/` is the model to copy).
- Imports flow **upward only**: `NNNC_CGP → NNNCStrategy → BaseNNStrategy →
  BaseStrategy`. A lower layer importing from a higher/sibling family layer is a
  layering violation to flag.

### Prefer shallow hierarchies + composition over deep inheritance
- Inheritance is for genuine *is-a* with shared behavior. The current
  `Base → NNNCStrategy → NNNC_CGP → NNNC_CGP_MLX → NNNC_CGP_MLX_LSTM` depth is a
  smell — each level should earn its existence with real differentiated
  behavior, not just hold one overridden constant.
- When subclasses differ only in a **value** (model name, a threshold, a backend
  choice), that's configuration, not a class. Collapse them into one
  parameterized class or a small data-driven table rather than N near-identical
  files. (`DataframePopulator`, `GANInterface`, and the `CreateGAN`/`CreateMTGAN`
  shims already embody this — extend that style.)
- Reach for **mixins/composition** when behavior is shared across siblings in
  *different* branches (e.g. an MLX-device concern shared by NNNC and NNMT). A
  mixin shared by two families belongs in `Framework/` or `utils/`, not duplicated
  in each family dir.

### Keep the override contract honest
- A subclass override that is byte-identical to the parent should be deleted —
  it's dead weight and hides where behavior actually lives.
- An override that changes behavior must respect the parent contract: same shapes
  / return types / side effects. If it can't, the method is mis-placed in the
  hierarchy.
- Lifecycle hooks that the framework chains (`bot_start`, `iteration_init`) **must
  call `super()`** — overriding without `super()` silently drops base setup. Flag
  any override that doesn't.

### Single responsibility & explicit interfaces
- One class, one job. New behavior should go into a collaborator (`utils/`, a
  backend, a populator) the strategy *uses*, or one of the Framework mixins, not
  another `if`-branch in the base.
- The `BaseNNStrategy` decomposition into `StrategyDiagnostics` /
  `FeatureNormalizer` / `TrainingEngine` mixins is the realized example of this:
  each was a behavior-preserving (byte-identical) relocation, verified by the
  training-pipeline characterization fixtures + scaler-diff/plot/backtest. Add to
  the matching mixin rather than re-growing `BaseNNStrategy`.
- Cross-family shared logic lives in exactly one place. Duplicate helper methods
  across family bases are consolidation targets — lift them to `BaseNNStrategy`
  or a `utils/` helper.
- Abstract bases (`GANBackend`, the family bases' required overrides) should
  declare their required methods explicitly (`@abstractmethod`) so a half-built
  subclass fails loudly at construction, not deep in a backtest.
