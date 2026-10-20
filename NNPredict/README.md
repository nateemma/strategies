# NNPredict — Neural Network Regression Strategies

The regression counterpart of `NNNC`: the model predicts a **continuous** future-gain value rather
than a class, and a per-pair gate turns unusually strong predictions into entries. For setup, data,
backtesting and the shared framework, see the [main README](../README.md).

> **Status (2026-10-21, feature 039, `research/nnpredict/report.md`):** on the 15-fold harness every
> NNPredict model loses money standalone (best: Ridge on the excursion target at H = 32, −11.0 pp).
> `NNPredictStrategy` carries the NNNC entry framework.
>
> **Defaults since 2026-10-21 (039 adopted):** entry = the model AND the gbb trigger
> (`ENTRY_MODE = "trigger_and_model"`, `TRIGGER_NAME = "gbb"`), guards open, percentile magnitude
> floor. On `NNPredict_LSTM_RawInd` that is −12.7 → +29.7 pp (IMPROVED, both halves), and the plain
> class reproduces the 039 gate arm's trades on 15/15 folds. On `st_g24_rev`'s exits the gbb trigger
> alone still beats it, so the gain is the trigger's.
>
> **Where NNPredict earns its keep is as a width-book sizer.** The excursion-H32 Ridge's predicted move
> sizes `w_aro`'s book to 57.0 vs 51.4 pp (IMPROVED), still below `cs_aro` 59.9.
>
> The 039 record classes pin what they ran: `_OP` arms are model-only (`_operating_point.py`), and the
> as-shipped models stage runs `<Class>_Pre039` (`_pre039.py`).

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

The floor is a rolling `magnitude_pctile` (0.60) percentile of |prediction| (`magnitude_floor_mode =
"percentile"`, the default since 039), or `max(min_magnitude, mag_std_mult × rolling std)` (`"absolute"`,
`min_magnitude` 0.10, `mag_std_mult` 0; the default before 039). A BUY becomes an entry only on a bar
where the gbb trigger also fires (the 039 default gate) and the entry guards pass (open by default).

This replaced the original rolling-quantile gate (q90/q10), which fired constantly when predictions
were smooth (see memory: label smoothing backfires). The gate settings act after the model, so they
do not change the model key, and changing them reuses the trained weights (039 fix).

## Backends

| Class | Model |
|---|---|
| `NNPredict_Ridge` | sklearn Ridge on the flattened feature window (saves `.joblib`). The cheap floor every model must beat |
| `NNPredict_LSTM_Raw*` | 039 plain-LSTM regressors (MLX `LSTM_RAW`) on raw8 + gains, + indicators (`RawInd`), or + lagged returns (`RawRet`); `_OP` twins run at the NNNC operating point |
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
