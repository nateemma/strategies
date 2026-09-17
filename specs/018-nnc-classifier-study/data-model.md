# Data Model: NNC Classifier Study

## Entities

### ExperimentConfig
| field | type | notes |
|---|---|---|
| `arch` | str | a `ClassifierTypeMLX` member name (existing or new) |
| `hp` | dict | hyperparameters injected as `clf.hp`; `{}` = the type's defaults |
| `input_set` | str | `ind24` \| `raw8` \| `both32` \| `ind24-minus:<group>` |
| `seq_len` | int | window length; production 16 |
| `split_set` | str | `v1` |
| `train` | dict | `{max_epochs: 40, patience: 10, batch_size: 256, lr: 1e-4, weight_decay: 0.0, stride: 1}` |
| `phase` | int | 2..7 (not in id) |
| `seeds` | list[int] | not in id |
| `tags` | list[str] | free text (not in id) |
| `config_id` | str | derived: sha1(canonical json of the id fields)[:12] |

**Validation**: `arch` must resolve in the registry; `input_set` must parse; `seq_len ≥ 2`;
`train.max_epochs ≥ 1`; `stride ≥ 1`. Baseline pseudo-archs `MAJORITY` and `PERSISTENCE`
are accepted by `run.py` and bypass training.

### SplitSet / Window
| field | type | notes |
|---|---|---|
| `split_set` | str | `v1` |
| `window_id` | str | `A`, `B`, `C` |
| `train_start`, `train_end` | ISO datetime | anchored; `train_end = test_start − EMBARGO_BARS` |
| `val_start`, `val_end` | ISO datetime | `val_end = train_end`; `val_start` = 10% of the train span from the end; rows in `[val_start − EMBARGO, val_start)` belong to neither train nor val |
| `test_start`, `test_end` | ISO datetime | evaluation window |
| `distant` | bool | Constitution II marker |
| `min_history_days` | int | 365; pair inclusion rule |

**Invariants** (tested): windows are disjoint and ordered; `train_end + EMBARGO ≤ test_start`;
`val_start − EMBARGO ≥` last train-only row; no row is in two roles; every window has ≥ 1
included pair with all three classes present in its training span.

### RunResult (one JSON line)
| group | fields |
|---|---|
| key | `config_id`, `split_set`, `window_id`, `seed` |
| echo | `phase`, `arch`, `hp` (json), `input_set`, `seq_len`, `train` (json), `tags` |
| status | `status` ∈ {`ok`, `failed`, `collapsed`, `class_absent`}, `error` (str or null) |
| data | `pairs_included`, `pairs_excluded`, `n_train`, `n_val`, `n_test`, `class_counts_train` [3], `class_counts_test` [3], `pred_counts` [3] |
| headline | `bal_acc`, `macro_f1`, `mcc` |
| per-class | `prec_sell`, `prec_hold`, `prec_buy`, `rec_sell`, `rec_hold`, `rec_buy` (NaN when undefined) |
| probabilistic | `logloss`, `auroc_sell`, `auroc_hold`, `auroc_buy`, `auroc_macro`, `ece`, `reliability_bins` (list of {lo, hi, conf, acc, n}) |
| matrix | `confusion` [[3×3]] rows = true, cols = pred, order SELL, HOLD, BUY |
| secondary | `buy_prec_at_08`, `buy_cov_at_08` |
| cost | `n_params`, `train_seconds`, `epochs_run`, `best_epoch`, `infer_ms_per_window_b256`, `infer_ms_per_window_b1` |
| provenance | `harness_version`, `git_sha` (nested strategies repo), `started_at`, `finished_at`, `weights_path` (or null) |

**Status rules**: `failed` = exception or non-finite loss; `collapsed` = a single predicted
class covers ≥ 99.5% of test rows; `class_absent` = a true class has zero test rows (metrics
still computed where defined). `collapsed` and `class_absent` rows keep their metrics; a
config with any `collapsed` or `failed` window is never viable.

### Phase file
A YAML list of ExperimentConfig objects plus a header `{phase: N, description, gate_from:
N−1 or null}`. `phase.py` expands `seeds` × windows into run keys.

### GateDecision (`gates/phase_N.json` + `.md`)
| field | notes |
|---|---|
| `phase` | the phase whose results were judged |
| `rule` | `{margin_floor: 0.02, margin_used: float, std_bound: 0.03, majority_of: 3}` |
| `ridge_reference` | per-window seed-mean MCC / macro-F1 of `RIDGE`, and its pooled seed-std of MCC |
| `per_config` | for each config: per-window pass/fail on each clause, overall `viable` bool |
| `viable_set` | list of config_ids carried forward |
| `decided_at`, `decided_by` | timestamp; "gate.py + author review at checkpoint" |

**Rule**: `margin_used = max(0.02, 2 × ridge_pooled_seed_std_mcc)` computed from the phase-2
rows at the time of the decision and frozen into the file.

### Leaderboard
Aggregation of RunResult: for each `config_id` × `window_id`, mean and std over seeds of
every numeric metric and `n_seeds`; plus `pooled_*` = mean over windows of the per-window
means. Sorted by pooled MCC; baselines always shown.

### Report
`report/report.md` sections: Scope & methods (incl. label-fidelity output and leakage
controls), Leaderboard (all phases), Phase findings 2–7, Gate decisions (verbatim from
`gates/`), Recommendation (candidates for later P&L evaluation), Label concerns, Open
questions, Reproduction commands. Plots under `report/plots/`.
