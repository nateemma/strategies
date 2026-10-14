# Data Model: Candidate-Only Training (032)

## TrainRowFilter (declared on the strategy)

`TRAIN_ROW_FILTER: tuple[tuple[str, str, float], ...] | absent`

- Each element is `(column, op, value)`. `op` is one of `"lt"`, `"le"`, `"gt"`, `"ge"`. The
  conditions are ANDed.
- An empty tuple is invalid (raise). An attribute that is absent or `None` means no filter.
- An unknown column or op raises. NaN in a column fails the condition.
- NNNC value (gbb rule): `(("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))`.
- It is part of the model key (it shapes weights) only on classes whose MRO declares it in
  `MODEL_KEY_ATTRIBUTES`.

## Filtered training set (in memory, per training run)

Unfiltered, per pair `i`:

| array | rows | meaning |
|---|---|---|
| `tsr_train[j]` | `train_end - offset` | window ending at bar `j + offset` |
| `train_labels[j]` | same | one-hot label of that bar |
| `train_weights[j]` | same, optional | P&L weight |

The `test` (validation) arrays are the same shape over `[test_start:]`.

Filtered: `keep = row_filter_mask(raw_pair_frame)[slice][offset:]` and `idx = flatnonzero(keep)`.
Every array of that split is indexed by `idx`. Invariants:
- `len(tensor) == len(labels) (== len(weights))`, both before and after filtering.
- A kept sample's window is bit-identical to the same row of the unfiltered tensor.
- An empty `idx` across all pairs raises before training starts.

## RowFilterReport (sidecar, written per training run)

`row_filter.json` sits next to the weights at the model key's location:

```json
{"spec": [["guard_metric","lt",0.3],["bb_width","gt",0.035]],
 "train": {"rows": 0, "kept": 0, "kept_share": 0.0, "class_counts": [0, 0, 0]},
 "validation": {"rows": 0, "kept": 0, "kept_share": 0.0, "class_counts": [0, 0, 0]}}
```

`class_counts` is ordered SELL, HOLD, BUY. This report is the liveness evidence (R9).

## Arm (per model)

| field | ARO | GBR |
|---|---|---|
| base class | `NNNC_CO_ARO` = `CandidateTrainingMixin` + `NNNC_A_Raw_Open` | `NNNC_CO_GBR` = mixin + `NNNC_FS_GBR_GR` |
| fold classes | `NNNC_CO_ARO_f00..f14` (generated) | `NNNC_CO_GBR_f00..f14` |
| plan / run_id | `plan_co_aro.json` / `co_aro` | `plan_co_gbr.json` / `co_gbr` |
| params snapshot | the control's (`params_20261013-g24_NNNC_A_Raw_Open.json`) | `params_fs_NNNC_FS_GBR_GR.json` |
| thresholds | `thresholds_co_aro.json` (rate 0.014735) | `thresholds_co_gbr.json` |
| control | `wf_aro_g24` | `fs_gbr_gr` |
| width book | `w_aro` | `w_gbr` |

Both arms set `HELD_PAIR_ADD = None` and use 24 h grace (from the snapshot).

## Pre-registration (`research/candidate_training/preregistration.json`)

```json
{"label": "...", "target_rate": 0.014735,
 "books": {"ARO": {"arm": "co_aro", "control": "wf_aro_g24", "width": "w_aro", "folds": ["f00", "...", "f14"]},
           "GBR": {"arm": "co_gbr", "control": "fs_gbr_gr", "width": "w_gbr", "folds": ["..."]}},
 "early": ["f00", "...", "f07"], "late": ["f08", "...", "f14"],
 "stages": {"S1": "...", "S2": "...", "S3": "..."},
 "decision": "...", "exempt_records": []}
```

It is committed before the first training run. The judge refuses any arm record older than its
commit.

## Verdict (`research/candidate_training/verdicts.json`)

Per model:
- `S1` holds the AUC by half for the retrained and the current model, and pass or fail.
- `S2` holds early pp for the arm, control and width, and pass, fail or not_reached.
- `S3` holds `judge_one` vs the control, `judge_one` vs the width book, late pp, and pass, fail or
  not_reached.
- Also `fidelity` (simulator recall and precision on the arm's entries), `entry_violations` (must
  be 0) and `liveness` (per-fold RowFilterReport).
- `decision` is one of ADOPTED, NOT ADOPTED or INCONCLUSIVE, with the stopping stage.
