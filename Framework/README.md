# Framework — Universal Base Classes

Every strategy in this repo (NN, sklearn, simple, time-series) ultimately
inherits from one of the bases here.  The framework owns the
freqtrade-facing contract — ROI, stoploss, trailing stops, custom_exit,
guard conditions, hyperopt parameters, lifecycle hooks — so leaf
strategies stay small and only override what's specific to them.

## Main files

| File | What it does |
|---|---|
| `BaseStrategy.py` | Root class for **every** strategy.  ROI/stoploss/trailing config; `bot_start()` for one-time setup (banner, environment, helper instantiation); `iteration_init()` for per-iteration setup; `populate_entry_trend` / `populate_exit_trend` templates; `custom_exit`, `custom_stoploss`, `confirm_trade_entry/exit`; classification assessment helpers; the `TradingAction`, `MarketRegime`, `RiskLevel`, `FlowDirection`, `MomentumDirection` enums; the `StrategyConfig` dataclass and `NormalizationType` / `ModelType` enums.  Re-exports `GANType` for backwards-compat imports. |
| `BaseNNStrategy.py` | NN-specific extension of `BaseStrategy`.  Adds the full ML pipeline: classifier construction (`get_classifier_type` / `get_classifier`), training-signal generation, normalization, the `enhance_training_data` GAN dispatcher (routes to `GANs.balance.balance_single_task` or `balance_multi_task` based on `gan_type` and label shape, with strict metadata validation via `GANInterface.load(expected=…)`), per-task class weights, train/save/load lifecycle.  All NN-family bases (NNNC, NNMT, Anomaly, Sklearn) inherit from this. |
| `TrainingSignals.py` | Future-aware label generation.  Given a price series, computes peak-detection-based buy/sell labels for training, parameterised by minimum-gain threshold, lookahead window, and label method (`TRAINING_TYPE`). |
| `CreateScalers.py` | One-shot strategy: run it under freqtrade backtesting once to fit and save the scalers used for feature normalisation across every NN strategy.  See top-level `README.md` for the command. |

## Class hierarchy

```
BaseStrategy
├── BaseNNStrategy
│   ├── NNNCStrategy   (NNNC/)
│   ├── NNMTStrategy   (NNMT/)
│   ├── NNAnomalyStrategy (Anomaly/)
│   └── SklearnStrategy (Sklearn/)
├── SimpleStrategy     (SimpleStrategies/)
└── TSPredict          (TSPredict/)
```

## Lifecycle (NN strategies)

`bot_start()` runs **once** when the bot starts: prints banner, sets
device visibility, applies any hyperopt threshold overrides
(`buy_params` / `sell_params`), and instantiates `DataframeUtils` /
`DataframePopulator`.  Threshold and `training_type` consistency
between the strategy and any saved GAN is enforced *at GAN load time*
via `GANInterface.load(expected=…)` rather than by silently overriding
the strategy's values from the GAN's metadata — drift raises
`GANMetadataMismatchError` so the operator can decide whether to
retrain the GAN or update the strategy.

`iteration_init()` runs at the start of every `populate_indicators()`
call: resets the per-iteration scaler state and updates
`self.training_needed = not self.model_exists()`.

If a saved model is present, it's loaded; otherwise the strategy
collects dataframes from every pair, optionally augments with the
configured GAN (set `gan_type` on the class — the dispatcher does the
rest), trains the classifier, and saves it to
`saved_data/<StrategyName>/`.
