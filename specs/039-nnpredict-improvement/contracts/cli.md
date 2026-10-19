# Contract: 039 entry points

| Command | Reads | Writes | Refuses when |
|---|---|---|---|
| `research.nnpredict.judge models` | model-stage fold records | `verdicts_models.json` (+ the R7 selection) | `preregistration_models.json` is uncommitted or dirty, or a record predates its commit |
| `research.nnpredict.build_arms framework` | `verdicts_models.json` (the R7 selection) | arm classes in `NNPredict/`, `evaluation/folds/plan_<M>_*.json` | `preregistration_framework.json` is uncommitted |
| `research.nnpredict.judge framework` | framework fold records | `verdicts_framework.json` (identity first) | uncommitted pre-registration; identity failed (feature verdicts are not read) |
| `research.nnnc_framework.signals --models NNP` | NNP fold classes (strict mode) | `research/nnnc_framework/signals/NNP_fNN.parquet` with a `predict_gain` column | a fold model is missing (strict mode) |
| `research.nnpredict.screen` | the signal tables, `research.new_inputs.trades` | `screen.json` | uncommitted `preregistration_sizer.json`; trade count or gap not reproduced |
| `research.nnpredict.build_arms sizer` | `screen.json` (must PASS), training windows | the `nnp_cs` class and plan (per-fold thresholds) | the screen did not pass |
| `research.nnpredict.judge sizer` | `nnp_cs` records | `verdicts_sizer.json` | uncommitted pre-registration |

## Library contracts

- `NNPredictStrategy.MODEL_KEY_ATTRIBUTES` holds only the names that shape the weights. Post-model
  gate names are in `MODEL_KEY_IGNORED`.
- `RegressorTypeMLX.LSTM_RAW`: features → LSTM (last time step) → `Linear(1)`. The same training
  loop as the existing MLX regressor.
- Every input-mixin column is causal (a truncation test) and listed in `pre_normalized_columns`.
- `NNPredictStrategy` with nothing declared: the entry is exactly the model's `predict_buy` and the
  tags are BaseStrategy's (identity).
