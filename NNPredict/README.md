# NNPredict — Neural Network Regression Strategies

Parallel to `NNNC` (n-ary classification) but the model predicts a
**continuous** future-gain target rather than a class label.  Entry / exit
actions are derived from a **rolling-quantile** threshold on the predicted
gains, so the buy/sell band adapts per pair and per regime.

Inherits from `BaseNNStrategy` (`Framework/BaseNNStrategy.py`) via
`NNPredictStrategy`.

## Why regression?

Classification compresses the target into 3 discrete buckets up front,
which throws away information about *how strong* the predicted move is.
A regressor keeps the magnitude, and the rolling-quantile gate decides
which predictions are "unusually high/low for this pair right now."

This frees the model from having to pick a global cutoff that works
across pairs with very different volatility profiles.

## Target

`get_training_labels` returns the H-bar-forward shift of `current_gain`:

```
labels[i] = (close[i+H] - close[i]) / close[i] / atr_pct[i+H]
```

i.e. the H-bar forward close-to-close return in ATR-units, clipped to
`±max(target_max_gain, target_max_loss)` and pre-normalized to `[-1, +1]`.

`H = HORIZON = 4` by default — deliberately small, picked to break the
"predict current state" shortcut a model finds when `H ≈ fisher_ss
period / 2`.

A matching backward feature, `recent_gain`, plus the per-bar
`current_gain` are added in `add_additional_indicators` and registered
as `pre_normalized_columns` so the shared scaler doesn't refit them.

## Signal logic

`get_predictions` runs the regressor, then for each pair computes a
rolling quantile over the last `rolling_window=200` predictions:

| Condition                                | Action |
|------------------------------------------|--------|
| `pred_gain > rolling_quantile(0.90)`     | BUY    |
| `pred_gain < rolling_quantile(0.10)`     | SELL   |
| otherwise                                | HOLD   |

There is **no** `> 0 / < 0` sign filter on top — the quantile alone is
what captures "unusually high/low" and the sign filter wrongly clips
predictions that sit predominantly on one side of zero for a given pair.

## What's bypassed vs. BaseNNStrategy

The classification-only paths in `BaseNNStrategy` are short-circuited:

| Method                          | Why overridden                                                                 |
|---------------------------------|--------------------------------------------------------------------------------|
| `get_training_labels`           | Returns 1D float array (continuous future_gain), not `TradingAction` indices.  |
| `prepare_training_data`         | Skips `one_hot_encode(labels, 3)` — float targets pass through.                |
| `get_training_class_weights`    | Returns `None` — `np.bincount` on negative continuous values would crash.      |
| `get_predictions`               | Consumes continuous regressor output, applies rolling quantile → `TradingAction`. |

Also disabled at the class level:

- `augment_training_data = False` — signal augmentation is binary-only,
  not meaningful for continuous targets.
- `use_markov_smoothing = False` — the Markov transition matrix is for
  discrete states.

## Main files

| File | What it does |
|---|---|
| `NNPredictStrategy.py` | Family base class.  Holds the regression pipeline overrides above, the rolling-quantile signal logic, target caps (`target_max_gain` / `target_max_loss = 8.0` ATR-units), and the `HORIZON` constant. |
| `NNPredictRegressor.py` | TF/Keras regressor factory.  `RegressorType` enum + `create_regressor()`.  Currently wraps `utils.NNPredictors.predictor_lstm`. |
| `NNPredictRegressorMLX.py` | Apple-MLX regressor factory.  `RegressorTypeMLX` enum + `create_regressor_mlx()`.  Hybrid LSTM: parallel `Linear(resize)` + `Conv1d(k=2)` branches concatenated, fed into an LSTM (last-timestep) and a linear output head.  Concat (not add) keeps the global and local views in separate channels. |
| `NNPredictRegressorRidge.py` | sklearn Ridge regressor factory.  `RegressorTypeRidge` enum + `create_regressor_ridge()`.  Linear-in-flattened-feature baseline. |
| `NNPredict_LSTM.py` | Concrete strategy: TF/Keras LSTM regressor. |
| `NNPredict_MLX_LSTM.py` | Concrete strategy: MLX LSTM regressor.  Sets `max_epochs = 300` and passes `HORIZON` to the trainer for the "predict-current-state" shortcut diagnostic. |
| `NNPredict_Ridge.py` | Concrete strategy: Ridge baseline.  Outperformed the LSTM variants on Spearman rank correlation in early diagnostics — kept as a floor any nonlinear model should beat. |

## Adding a new variant

1. Create `NNPredict_<Name>.py` here.
2. Inherit from `NNPredictStrategy`.
3. Override `get_classifier_type()` to return a `RegressorType` /
   `RegressorTypeMLX` / `RegressorTypeRidge` value.  Override
   `get_classifier()` if the backend needs custom wiring (see
   `NNPredict_MLX_LSTM.py` for the MLX guard + `max_epochs` /
   `horizon` plumbing).
4. Run a long-timerange backtest to train and save the model.

See top-level `README.md` and `AGENT_GUIDE.md` for build / test commands.
