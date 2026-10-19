# NNPredict — Neural Network Regression Strategies

The regression counterpart of `NNNC`: the model predicts a **continuous** future-gain value rather
than a class, and a per-pair gate turns unusually strong predictions into entries. For setup, data,
backtesting and the shared framework, see the [main README](../README.md).

> **Status (2026-10-19):** NNPredict had never been evaluated on the project's 15-fold walk-forward
> harness. Feature 039 (`specs/039-nnpredict-improvement/`, `research/nnpredict/`) puts it there, adds
> plain-LSTM regressors, ports the NNNC entry framework and tests NNPredict as a width-book sizer.
> Results are in `research/nnpredict/report.md`. Feature 038 found that forecasting short-horizon
> returns from price and indicators has about zero edge after costs, so expect gains, if any, from
> NNPredict as a **ranker or sizer**, not from more forecast accuracy.

## Target (`target_mode`)

All targets are in ATR units, clipped to ±8 and scaled to [−1, +1]. H = `HORIZON` (4 by default).

| Mode | Label at candle i |
|---|---|
| `"point"` (default) | close-to-close return from i to i+H |
| `"excursion"` | the dominant signed move within the next H candles (the larger of the rise and the fall), i.e. what a trade could capture |
| `"stop_aware"` | the point return, unless the exit stack's ATR stop (with grace) would have been hit first, in which case the realised stop |

`add_additional_indicators` also adds two pre-normalised inputs: `current_gain` (the backward H-candle
return) and `recent_gain` (the backward excursion).

## Signal gate (`get_predictions`)

Over the last `rolling_window` (200) predictions of each pair:

| Condition | Action |
|---|---|
| z-score of the prediction ≥ `entry_z` (1.0) **and** prediction > magnitude floor | BUY |
| z-score ≤ −`entry_z` **and** prediction < −floor | SELL |
| otherwise | HOLD |

The floor is `max(min_magnitude, mag_std_mult × rolling std)` (`magnitude_floor_mode = "absolute"`,
`min_magnitude` 0.10, `mag_std_mult` 0), or a rolling percentile of |prediction| (`"percentile"`).

This replaced the original rolling-quantile gate (q90/q10), which fired constantly when predictions
were smooth (see memory: label smoothing backfires). The gate settings act after the model, so they
do not change the model key, and changing them reuses the trained weights (039 fix).

## Backends

| Class | Model |
|---|---|
| `NNPredict_Ridge` | sklearn Ridge on the flattened feature window (saves `.joblib`). The cheap floor every model must beat |
| `NNPredict_LSTM` | Keras LSTM regressor |
| `NNPredict_MLX_LSTM` | MLX hybrid LSTM (parallel resize and conv branches into an LSTM) |
| `NNPredict_MLX_MultiHorizon_LSTM` | Joint multi-horizon heads. Failed: the shared backbone fits the easiest horizon (memory) |
| `NNPredict_Coeff*` | Ridge on wavelet-coefficient features (`utils/Wavelets.py`). The `_Exc` config was the best pre-harness result (8/8 windows on its own 8-window walk-forward); excluded from 039 while `Wavelets.py` has uncommitted edits |

## What earlier work established (memory)

- **Fit metrics are decoupled from P&L.** Regularisation, PCA, horizon sweeps and feature smoothing
  moved ρ or R² but not walk-forward P&L. Only cutting turnover (`min_magnitude`, a cliff between 0.10
  and 0.20) and the excursion target helped.
- **Prediction spread is not a quality measure.** An adaptive floor based on it removed winning pairs.
- **Joint multi-horizon training fails,** and label smoothing backfires with z-score or quantile gates.

## Adding a variant

1. Create `NNPredict_<Name>.py`, inheriting `NNPredictStrategy` (or a concrete class).
2. Choose the backend via `get_classifier_type()`, or `classifier_type` for MLX.
3. Merge any parameter dict from the parent (`{**Parent.buy_params, ...}`).
4. Evaluate it on the 15-fold harness (`evaluation/run_folds.py`). See 039's plans.
