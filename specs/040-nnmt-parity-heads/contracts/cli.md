# Contract: 040 entry points

Every command below refuses to write a verdict or screen when its pre-registration is uncommitted
or dirty, or when a record it reads predates that commit (the `research.nnnc_switches.judge`
`refusals()` rule).

| Command | Reads | Writes | Refuses when |
|---|---|---|---|
| `research.nnmt.build_arms models` | — | `evaluation/folds/plan_{mt,mt_fw0,mt_op,mt_raw,mt_lr,mt_lr_raw,mt_lr_ri}.json` and the arm classes | `preregistration_models.json` is uncommitted |
| `research.nnmt.judge identity` | `mt`, `mt_fw0` records | `verdicts_models.json["identity"]` | the record count is not 15 per arm |
| `research.nnmt.judge models` | model-stage records, signal tables (model share) | `verdicts_models.json` (+ the R7 selection) | identity failed |
| `research.nnmt.build_arms framework` | the R7 selection | `<M>_g24/_hpa/_gate/_tp/_tr` classes and plans | `preregistration_framework.json` is uncommitted |
| `research.nnmt.judge framework` | framework records | `verdicts_framework.json` | an uncommitted pre-registration |
| `research.nnnc_framework.signals --models <M>` | the fold classes (strict mode) | `signals/<M>_fNN.parquet` with `p_<task>_<class>` | a fold model is missing |
| `research.nnmt.screen sizer` | the signal tables, `research.new_inputs.trades` | `screens_sizer.json` | `preregistration_screens.json` is uncommitted; 412 trades or the 0.1004 gap is not reproduced |
| `research.nnmt.build_arms sizer` | `screens_sizer.json` (must pass) | the `<M>_cs` class, plan and thresholds | the screen did not pass |
| `research.nnmt.head_screen` | `<M>_ah` signal tables, the auxiliary labels (recomputed) | `screens_heads.json` | an uncommitted pre-registration; any head column all-NaN |
| `research.nnmt.build_arms heads` | `screens_heads.json` | `hd_<head>` classes and plans (passing heads only) | no head passed (writes nothing, exits 0) |
| `research.nnmt.transition labels` | OHLCV (config_wf pairs + BTC) | `user_data/data/nnmt_transition/labels_<H>.parquet`, base rates | `preregistration_transition.json` is uncommitted |
| `research.nnmt.transition causality` | the label builder and inputs | `screens_transition.json["truncation"]` | — (exit 1 on any mismatch) |
| `research.nnmt.transition screen --fold fNN` | labels, inputs | stand-in predictions `user_data/data/nnmt_transition/pred_<cell>_fNN.parquet` | the truncation test has not passed |
| `research.nnmt.transition judge` | stand-in predictions | `screens_transition.json` (pass, selected cells) | — |
| `research.nnmt.build_arms transition` | `screens_transition.json` | `tr_*` and `<M>_trh` classes and plans (passing cells only) | no cell passed |
| `research.nnmt.judge {sizer,heads,transition}` | their records | `verdicts_<stage>.json` | an uncommitted pre-registration |

## Library contracts

- **`NNMTStrategy`** with nothing declared (`ENTRY_MODE = "model"`, `MODEL_ENTRY_FAMILY = None`,
  `HELD_PAIR_ADD = None`): the entry is exactly NNMT's current mask and the tags are BaseStrategy's
  (identity).
- **`NNMTStrategy.MODEL_KEY_ATTRIBUTES`** holds only the names that shape the weights. Every
  post-model name is in `MODEL_KEY_IGNORED`.
- **`ClassifierTypeMLX.LSTM_RAW`** (multi-task): features → LSTM (last step) → one `Linear(n_classes)`
  per task. No resize or convolution. The same training loop and task weights as the other types.
- **`NNMTStrategy._head_probs`**: after `get_predictions`, a dict `task → (n_bars, n_classes)`
  probabilities aligned with the dataframe offset. Its argmax equals the `%<task>` column where the
  threshold does not force neutral.
- **Transition labels** are a pure function of OHLCV up to t+H. `S_H(t)` is a pure function of OHLCV
  up to t.
