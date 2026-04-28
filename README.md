# Phil's Custom freqtrade Crypto Trading Strategies (Version 2)

_**The directory structure and code architecture have completely changed. The older code is available on branch v1, but I will not be maintaining that**_

Note that, like the rest of the software world, I have been playing with agent-assisted coding, so some of the code might look a little strange. I added a file AGENT_GUIDE.md that should be useful for agents (or humans) looking at the code.

All strategies share a common framework for entry/exit processing, custom exits, and stop-loss logic. There is also a consistent approach to normalization of data. The base framework lives in the _Framework_ directory.

Many of the strategies use Neural Networks (Machine Learning), which requires training the underlying models on lots of data. Even then, I had to implement Generative Adversarial Network (GAN) models to artificially enhance the data because there are simply not enough buy/sell signals to train the stategy models. Look in the GANs directory for more info.

## Quickstart

There are a bunch of scripts available to help, I recommend using them. Note that I use a Mac so I have to use the zsh shell, but I think most of these will run with bash.

*Download backtest data* (run daily):
```zsh
zsh user_data/strategies/scripts/download.sh -n 720 binanceus
```

*Create scalers for normalization* (only need to do this once for all strategies):
```zsh
zsh user_data/strategies/scripts/test_strat.sh -n 720 -o 30 Framework CreateScalers
```

*Train model* (if not already trained). 

Only need to do this once per strategy, and only if the strategy uses a saved model (which is all of the Neural Network-based strategies). If there is no existing model, the code will automatically train a new one, but you need to use a lot of data for this. 

Note - don't train on the same timeframe that are using for backtesting (use the -o argument)!:
```zsh
zsh user_data/strategies/scripts/test_strat.sh -n 720 -o 30 NNNC NNNCStrategy
```

*Run backtest*:
```zsh
zsh user_data/strategies/scripts/test_strat.sh -n 30 NNNC NNNCStrategy
```

*Plot results*:
This is very useful for assessing a strategy. It produces an html file that you can view in a browser. In the example below, that file would be:

file:///<freqtrade>/user_data/plot/freqtrade-plot-SOL_USDT-15m.html

```zsh
zsh user_data/strategies/scripts/plot_strat.sh -n 30 NNNC NNNCStrategy SOL/USDT
```

*Check Lookahead Bias*:
```zsh
zsh user_data/strategies/scripts/check_bias.sh -n 30 NNNC NNNCStrategy
```

More info below...

## Disclaimer

These strategies are for educational purposes only.

Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by backtesting, then dry-run for at least a week before going live. A strategy that looks great in backtesting often performs very differently with real, live data — market conditions, slippage, and timing cannot be fully reproduced from historical data.

Do not backtest only in bull markets (e.g., 2020). Include periods of poor performance (e.g., May, Nov, Dec 2021).

## Directories
```
tree -d -L 1
.
├── Anomaly
├── archived
├── config
├── Debug
├── Framework
├── GANs
├── hyperopts
├── MLX
├── NNMT
├── NNNC
├── reference
├── saved_data
├── scripts
├── SimpleStrategies
├── Sklearn
├── TSPredict
└── utils
```

- _Anomaly_ — anomaly-detection strategies (autoencoder + GANomaly variants)
- _archived_ — abandoned strategies kept for reference/cut & paste
- _config_ — exchange-specific config files (replaces old per-exchange subdirectories)
- _Debug_ — debug/visualisation strategies (all begin with `Debug`)
- _Framework_ — universal base classes (`BaseStrategy`, `BaseNNStrategy`, `TrainingSignals`, `CreateScalers`)
- _GANs_ — GAN implementations + the unified `GANInterface` / `GANBackend` system that wraps them.  See `GANs/README.md` for backend details.
- _hyperopts_ — custom hyperopt loss functions (copy to `user_data/hyperopts` to use)
- _MLX_ — Apple MLX neural network components (Mamba, etc.) shared across strategy families
- _NNMT_ — Neural Network Multi-Task classification strategies
- _NNNC_ — Neural Network N-ary (trinary) classification strategies
- _reference_ — example strategies from other authors (for learning purposes)
- _saved_data_ — saved model files, scalers, and GAN state; keyed by strategy name
- _scripts_ — shell scripts for backtesting, hyperopt, dry-run, and live trading
- _SimpleStrategies_ — indicator-driven strategies (no ML); each file is a standalone strategy
- _Sklearn_ — sklearn-based classifier strategies (RandomForest, XGBoost, etc.) sharing the NN training/augmentation pipeline
- _TSPredict_ — time-series prediction strategies using wavelets, FFTs, and DWT
- _utils_ — shared utility code (classifiers, data manipulation, indicators, wavelets, etc.)

> _**Note**: an older `NeuralNets/` directory used to hold a separate git repo with the NN base classes and scaler storage.  Its contents have been folded into `Framework/` (base classes) and `saved_data/` (scalers).  If you see references to `NeuralNets/` in older docs or branches, treat them as historical._

---

_NOTES_:

- _**BinanceUS**_: I live in the USA, and most exchanges have blocked API access from here. All strategies should work, but you will need to run hyperopt to get good parameters for your exchange.

- _**Mac M1**_: My development machine is a Mac M1 laptop. All scripts are written for _zsh_ (the default macOS shell). See [README_MACM1.md](README_MACM1.md) for package notes.

- _**MLX**_: Several strategies use neural network models implemented with Apple's MLX library (similar to PyTorch). Where possible, non-MLX alternatives are provided, but these are harder for me to test consistently.


## Strategy Class Hierarchy

```
BaseStrategy (Framework/)
├── BaseNNStrategy (Framework/)
│   ├── NNNCStrategy (NNNC/)        – N-ary (trinary) classifiers
│   ├── NNMTStrategy (NNMT/)        – Multi-task classifiers
│   ├── NNAnomalyStrategy (Anomaly/) – Anomaly detection
│   └── SklearnStrategy (Sklearn/)  – sklearn classifiers (RF, XGBoost, …)
├── SimpleStrategy (SimpleStrategies/)
└── TSPredict (TSPredict/)          – Wavelet/FFT/DWT regression
```

`BaseStrategy` provides the universal boilerplate: ROI tables, stop-loss, trailing stops, `custom_exit`, guard conditions, and `DataframePopulator` integration. `bot_start()` (freqtrade's one-time-init hook) handles environment setup, hyperopt-parameter printing, and shared utility instantiation; `iteration_init()` runs per-iteration scaler reset.

`BaseNNStrategy` adds the full ML pipeline on top: classifier construction, training-signal generation, GAN augmentation hooks (`wgan_enhance_training_data`, `ctab_gan_enhance_training_data`), and per-task class-weight computation.  Family bases (`NNNCStrategy`, `NNMTStrategy`, etc.) wire in the family-specific classifier factory and label format.

Subclasses add family-specific logic and need only override a small number of methods.

## Intro

This folder contains a variety of custom trading strategies for use with the [freqtrade](https://www.freqtrade.io/) framework.

Please read the freqtrade documentation at https://www.freqtrade.io before using this software.

I currently focus on strategies that revolve around one of several approaches:

1. **Time-series prediction** — model expected price behaviour and compare to actual. Buy when the model projects a higher price (above a threshold), sell when it projects lower. Variants use Discrete Wavelet Transforms (DWT), FFTs, and Kalman filters. The DWT variants tend to perform best.

2. **Neural network classification** — trinary classifiers (sell/hold/buy) based on technical indicators. Base class is `BaseNNStrategy` (`Framework/`); per-family subclasses live in `NNNC/` (N-ary), `NNMT/` (multi-task), `Anomaly/`, and `Sklearn/`. Models are trained over long periods and saved to `saved_data/<StrategyName>/`.

3. **Anomaly detection** — train on historical "normal" (hold) data, then flag anomalous points (unusually high reconstruction error) as buy/sell candidates. Variants use `NNAnomalyClassifier` (autoencoder) and `NNGANomalyClassifier` (GANomaly).

4. **Simple indicator strategies** — single-indicator or small-combination strategies in `SimpleStrategies/`. Each file is self-contained. I implemented these mostly to get a baseline for expected performance, but also to be able to visualise (through plots) how the various indicators behave so that I can use them in more complex strategies


## Neural Network Strategies

### Model Management

Neural net strategies train a model on first run and save it to `saved_data/<StrategyName>/`.  Subsequent runs load the saved model.

To retrain: delete the files in `saved_data/<StrategyName>/`.

When training, use a long timerange (at least 1 year). Use the training config:

```
zsh user_data/strategies/scripts/test_strat.sh NNNC NNNC_CGP -c config/config_binanceus_train.json
```

### Scalers

Before running NN strategies for the first time, create the scalers:

```
freqtrade backtesting -c config/config_binanceus.json --strategy-path user_data/strategies/Framework --strategy CreateScalers --timerange=20230101-
```

or:

```zsh
zsh user_data/strategies/scripts/test_strat.sh -n 720 -o 30 Framework CreateScalers
```

Scaler files are stored in `saved_data/`.

It turns out that normalization of data is very, very important for any kind of model-based strategy. The data passed to the models cannot have pair-specific data (like price or volume), and the data must be consistently scaled across pairs. Because of this, I spent a lot of time tweaking which indicators are passed into the models. Also, using several indicators which essentially have the same information confuses the models, so I have trimmed down redundant indicators where possible. If you are interested, look at DebugAnalyseDf to see how I assess the indicators.

So, do *not* just blindly throw indicators into the dataframe, it will mess up the models. Also, if you do add anytghing, you will need to update the notrmalization code, regenerate the scalers and retrain the GAN and strategy models.

### GAN Data Augmentation

Neural net strategies suffer from severe class imbalance (many more holds than buys/sells). GANs generate synthetic minority-class samples to improve training.

GAN models are created by running one of the `Create*GAN*` strategies in `GANs/` (e.g. `CreateWGAN`, `CreateMTWGAN`, `CreateCtabGanPlus`) under freqtrade backtesting; the trained model is stored in `saved_data/GANs/` (or `saved_data/CTABGANs/`, `saved_data/MTCTABGANs/` for the CTAB-GAN+ variants).  Strategies that consume the GAN load it via `GANInterface(GANType.X, save_path=...)`.

Internally, every GAN type is wrapped by a `GANBackend` subclass (in `GANs/backends/`) so the `fit / generate / save / load` lifecycle is uniform across types and across MLX/TF backends.  Adding a new variant of an existing type is a one-class change in `GANs/`; adding a genuinely new GAN type means a new `GANBackend` in `GANs/backends/`.  See `GANs/README.md` for backend details and `GANs/tests/` for the contract tests.

### NNNC Family

`NNNC/` contains N-ary (trinary: sell/hold/buy) classifiers. Variants differ by model architecture:

| Variant             | Architecture            |
|---------------------|-------------------------|
| `NNNC_Strategy`     | Base (MLP)              |
| `NNNC_CGP`          | Base, with CTAB-GAN+ enhancement |
| `NNNC_CGP_LSTM2`    | LSTM                    |
| `NNNC_CGP_GRU`      | GRU                     |
| `NNNC_CGP_CNN`      | 1D CNN                  |
| `NNNC_CGP_Transformer` | Transformer          |
| `NNNC_CGP_Attention`| Multi-head attention    |
| `NNNC_CGP_TCN`      | Temporal Convolutional  |
| `NNNC_CGP_VAE`      | Variational Autoencoder |
| `NNNC_CGP_Wavenet`  | WaveNet                 |
| `NNNC_CGP_MLX_*`    | Apple MLX variants      |

### NNMT Family

`NNMT/` uses multi-task classification — the model predicts several variables simultaneously, which reduces overfitting. Variants follow the same naming pattern as NNNC.

### Anomaly Family

`Anomaly/` trains on "normal" (hold) data and treats high-reconstruction-error points as buy/sell candidates. `NNGANomalyStrategy` uses GANomaly for better anomaly generation.

## Reference repositories

Strategies I either used or learned from:

- https://github.com/freqtrade/freqtrade-strategies
- https://github.com/i1ya/freqtrade-strategies
- https://github.com/ntsd/freqtrade-configs
- https://github.com/froggleston/cryptofrog-strategies
- https://github.com/werkkrew/freqtrade-strategies
- https://github.com/brookmiles/freqtrade-stuff
- https://github.com/hansen1015/freqtrade_strategy/blob/main/heikin.py
- https://github.com/Foxel05/freqtrade-stuff

## Setting Up Your Configuration

See the [freqtrade docs](https://www.freqtrade.io/en/stable/configuration/) for generic instructions.

Config files are now kept in the `config/` directory:

- `config/config_<directory>.json` — static pairlist config for backtesting/hyperopt
- `config/config_<directory>_short.json` — futures/short config
- `config/config_<directory>_download.json` — download-only config
- `config/config_<directory>_train.json` — long-range training config for NN models

Do **not** put API keys or passwords in any of these files. Keep those in a separate config at the project root that is not committed to git.

### Configuration for Short Strategies

Short trading requires futures mode. Add to your config:

```json
"trading_mode": "futures",
"margin_mode": "isolated",
```

To list available pairs for futures trading:

```
freqtrade list-pairs --exchange <directory> --trading-mode futures
```

Filter to USDT pairs with leverage > 1:

```
freqtrade list-pairs --exchange binanceus --trading-mode futures | grep USDT | awk '$16>1 {print "\""$4"\","}'
```

## Downloading Test Data

Most strategies need 5m, 15m, and 1h data. Some also use 1d and BTC/USDT:

```
freqtrade download-data --timerange=<timerange> -c <config> -t 5m 15m 1h 1d
```

Or use the script (defaults to last 180 days):

```
zsh user_data/strategies/scripts/download.sh [<directory>]
```

For futures/short data:

```
zsh user_data/strategies/scripts/download.sh --short [<directory>]
```

## Generating a Pairlist

VolumePairlist does not work for backtesting or hyperopt. Generate a static pairlist with:

```
freqtrade test-pairlist -c <real_config>
```

Copy the output into your test config's `pair_whitelist`. Change single quotes to double quotes.

## Backtesting

```
freqtrade backtesting -c <config> --strategy-path <path> --strategy <strategy> --timerange=<timerange>
```

Or:

```
zsh user_data/strategies/scripts/test_strat.sh <directory> <strategy>
```

To test a whole group of strategies matching a pattern:

```
zsh user_data/strategies/scripts/test_group.sh <directory> <pattern>
```

e.g., `test_group.sh NNNC "NNNC_CGP_MLX*"` — supports shell wildcards.

**Lookahead bias**: If you see suspiciously high backtest results (100%+), your strategy is likely using future data. Use rolling operations or TA-lib functions to avoid this. See [freqtrade docs on common mistakes](https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies).

**Bias check**: Use the `check_bias.sh` script to run the freqtrade lookahead-bias detector on a strategy:

```
zsh user_data/strategies/scripts/check_bias.sh <directory> <strategy>
```

## Plotting Results

```
freqtrade plot-dataframe --strategy-path <path> --strategy <strategy> -p BCH/USD --timerange=<timerange> --indicators1 ema5 ema20 --indicators2 mfi
```

Or use the script:

```
zsh user_data/strategies/scripts/plot_strat.sh <directory> <strategy> [<pair>]
```

Output goes to `user_data/plot/`. Open the HTML file in a browser — it's interactive.

## Hyper-Parameter Optimisation

```
freqtrade hyperopt -c <config> --strategy-path <path> --strategy <strategy> --spaces <space> --hyperopt-loss <loss> --timerange=<timerange>
```

Or:

```
zsh user_data/strategies/scripts/hyp_strat.sh -s "buy sell roi" <directory> <strategy>
```

To hyperopt a group of strategies:

```
zsh user_data/strategies/scripts/hyp_group.sh <directory> <pattern>
```

**Optimised parameters** are written to a `.json` file matching the strategy (e.g. `NNNC_CGP.py` → `NNNC_CGP.json`). These override the Python defaults, so if parameter changes aren't taking effect, check the json file.

### Stoploss

I typically do _not_ optimise for stoploss — I set it manually to 10% (`-0.1`). Optimising stoploss tends to give better backtest numbers but worse real-world results: one losing trade with a large stoploss wipes out many winners.

## Hyperopt Loss Functions

Custom loss functions are in the `hyperopts/` directory. Copy them to `<freqtrade>/user_data/hyperopts/` to use them.

| Loss Function                 | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| `ExpectancyHyperOptLoss`      | Optimises primarily on Expectancy (projected profit per trade)               |
| `OnlyExpectancyHyperOptLoss`  | Optimises purely on Expectancy, ignoring other metrics                       |
| `PEDHyperOptLoss`             | Optimises equally on Profit, Expectancy, and Duration                        |
| `WeightedProfitHyperOptLoss`  | Optimises primarily on profit, with secondary metrics                        |
| `QuickProfitHyperOptLoss`     | Like WeightedProfit but prioritises short trade duration                     |
| `WinHyperOptLoss`             | Optimises primarily on Win/Loss ratio                                        |
| `MarketHyperOptLoss`          | Win/Loss-based loss adjusted for market conditions                           |
| `MedianProfitHyperOptLoss`    | Optimises on median profit per trade (more robust to outliers)               |

All functions require a minimum number of trades and a minimum win/loss ratio. I generally use `ExpectancyHyperOptLoss`.

Example:

```
zsh user_data/strategies/scripts/hyp_strat.sh -l ExpectancyHyperOptLoss binanceus NNNC_CGP
```

## Scripts Reference

See `scripts/README.md` for the canonical list. The most commonly used scripts are:

| Script                      | Description                                                                                    |
|-----------------------------|------------------------------------------------------------------------------------------------|
| `download.sh`               | Downloads candle data for an exchange (defaults to all exchanges, last 180 days)               |
| `safe_download.sh`          | Like download.sh but skips pairs that already have recent data                                 |
| `test_strat.sh`             | Tests an individual strategy for the specified exchange                                        |
| `test_group.sh`             | Tests all strategies matching a glob pattern; summarises results                               |
| `test_monthly.sh`           | Runs test_group.sh over the past 6 months; shows average performance and ranks strategies      |
| `hyp_strat.sh`              | Runs hyperopt on an individual strategy                                                        |
| `hyp_group.sh`              | Runs hyperopt on a group of strategies matching a pattern                                      |
| `hyp_leveraged.sh`          | Runs hyperopt for leveraged/futures strategies                                                 |
| `check_bias.sh`             | Runs the freqtrade lookahead-bias detector on a strategy                                       |
| `plot_strat.sh`             | Generates an interactive plot for a strategy and pair                                          |
| `dryrun_strat.sh`           | Dry-runs a strategy on the specified exchange                                                  |
| `run_strat.sh`              | Runs a strategy live on the specified exchange                                                 |
| `cleanup.sh`                | Removes old files from user_data subdirectories (default: older than 30 days)                  |
| `install_packages.sh`       | Installs required Python packages for the strategies                                           |
| `update_python.sh`          | Updates Python packages to latest compatible versions                                          |

All scripts accept `-h` for help.

## Dry Runs

```
freqtrade trade --dry-run --strategy-path <path> --strategy <strategy>
```

Or:

```
zsh user_data/strategies/scripts/dryrun_strat.sh -p <port> <directory> <strategy>
```

The `-p` flag is needed when running multiple strategies on the same exchange (requires a matching `config_<directory>_<port>.json`).

If you have freqUI installed, monitor trades at http://127.0.0.1:8080/.

## Live Trading

**Never go live without at least a week of dry-run on real data.**

```
freqtrade trade --strategy <strategy>
```

Or:

```
zsh user_data/strategies/scripts/run_strat.sh -p <port> <directory> <strategy>
```

_Your computer must be synced to an NTP time source for live trading._
