# Agent Guide: Building and Running freqtrade Strategies

This file is intended to orient an AI agent (or a new developer) on how to build, test, and run trading strategies in this codebase. It covers the directory layout, class hierarchy, common tasks, and the commands to run them.

---

## Environment

- **Shell**: `zsh` (macOS default). All scripts use `zsh`, not `bash`.
- **Python**: Conda-managed environment. Activate with `source .venv/bin/activate` before running any Python or freqtrade commands.
- **Working directory**: Most freqtrade commands and scripts should be run from `~/freqtrade/` (the project root).
- **PYTHONPATH**: The scripts set this automatically. If running manually, export:
  ```
  export PYTHONPATH=~/freqtrade/user_data/strategies:$PYTHONPATH
  ```

---

## Directory Map

```
user_data/strategies/
├── Framework/           ← Universal base classes
│   ├── BaseStrategy.py  ← Root base class for ALL strategies (ROI, stoploss,
│   │                      bot_start lifecycle, custom_exit, guards)
│   ├── BaseNNStrategy.py← NN ML pipeline base (inherits BaseStrategy):
│   │                      classifier construction, training-signal labels,
│   │                      GAN augmentation hooks, class-weight handling
│   ├── TrainingSignals.py  ← Future-aware label generation
│   └── CreateScalers.py ← Run once to generate normalization scalers
├── utils/               ← Shared utility code
│   ├── DataframeUtils.py
│   ├── DataframePopulator.py     ← Adds all technical indicators to a dataframe
│   ├── ClassifierKeras*.py        ← Keras classifier implementations
│   ├── ClassifierMLX*.py          ← Apple MLX classifier implementations
│   ├── ClassifierMLXMultiTask.py  ← MLX multi-task base (focal loss + grad clipping)
│   ├── ClassifierSklearn.py       ← sklearn classifier implementations
│   ├── Wavelets.py
│   ├── Forecasters.py
│   └── ...
├── NNNC/                ← N-ary (trinary) Classification strategies + NNNClassifier
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
├── saved_data/          ← Trained model files (keyed by strategy name)
│                          + GANs/<gan_type>/ subdirs for every GAN type
│                          (GANs_PCA/<gan_type>/ for PCA-reduced strategies)
├── scripts/             ← Shell scripts for all workflow tasks
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
BaseStrategy (Framework/BaseStrategy.py)
├── BaseNNStrategy (Framework/BaseNNStrategy.py)  ← full ML pipeline
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

### BaseNNStrategy responsibilities (in addition to BaseStrategy's)
- Classifier construction via `get_classifier_type()` + `get_classifier()`
- Training-signal generation via `TrainingSignals` + MASTER thresholds
- GAN augmentation via a single dispatcher: `enhance_training_data`
  inspects `gan_type` and the label shape (ndarray vs dict), validates
  the saved GAN's metadata against the strategy's current config, and
  routes to `GANs.balance.balance_single_task` or `balance_multi_task`.
  Concrete strategies declare `gan_type` (and optionally
  `gan_target_ratio`, `gan_run_diagnostics`, `gan_passthrough_columns`)
  — they don't see GAN-type-specific code.  Multi-task 3-D pipelines
  (e.g. `NNMT_WGAN`) turn off the 2-D dispatcher with
  `gan_augment = False` and run their own `preprocess_training_data`
  that delegates to `balance_multi_task` on the 3-D tensor.
- Per-task class-weight computation
- Train / save / load lifecycle wired into `populate_indicators()`

### Adding a new strategy (NN family)
1. Create a new `.py` file in the appropriate family directory (e.g., `NNNC/`, `NNMT/`, `NNPredict/`, `Anomaly/`, `Sklearn/`)
2. Inherit from the appropriate family base class (e.g., `NNNCStrategy`, `NNMTStrategy`, `NNPredictStrategy`, `SklearnStrategy`)
3. Override `get_classifier_type()` and `get_classifier()` to return your model (or regressor, for `NNPredictStrategy` subclasses)
4. Optionally override `add_strategy_indicators()`, `get_custom_training_data()`, etc.
5. Run backtest over a long period to train and save the model

### Adding a new SimpleStrategy
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
Delete the existing model files and re-run backtest with a long timerange:
```zsh
rm -rf user_data/strategies/saved_data/NNNC_CGP/*
zsh user_data/strategies/scripts/test_strat.sh NNNC NNNC_CGP \
  --timerange 20220101-
```

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

Understanding the data flow is critical for debugging or extending NN strategies.  The pipeline lives in `Framework/BaseNNStrategy.py` and is shared across NNNC, NNMT, Anomaly, and Sklearn family bases.

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
3. Adds training labels via `TrainingSignals`.
4. Normalizes features using pre-computed scalers.
5. If no saved model: collects dataframes from all pairs, trains, saves.
6. If model exists: loads it.

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
Check the `.json` file beside the `.py` file (e.g., `NNNC_CGP.json`). Hyperopt results in this file **override** Python defaults.

### Model not retraining
Delete the saved model directory:
```zsh
rm -rf user_data/strategies/saved_data/<StrategyName>/*
```

### `ImportError` or `ModuleNotFoundError`
Make sure PYTHONPATH includes the strategies directory:
```zsh
export PYTHONPATH=~/freqtrade/user_data/strategies:$PYTHONPATH
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
