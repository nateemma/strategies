# Contract C1: pooled width-relative relabel (Framework, strategy-agnostic)

## `TrainingSignals.path_outcome(df, target, stop, horizon, side="buy") -> (np.ndarray, np.ndarray)`

The `side="sell"` mirror uses `target = −exit_loss` (a fall) and `stop = +exit_take_profit`. It
matches `labels_gbb_path_sell` in the same way the buy side matches `labels_gbb_path`.

- **Input:** a frame with a `close` column and `horizon >= 1`. For `side="buy"`, `target > 0` and
  `stop < 0`; for `side="sell"`, `target < 0` and `stop > 0`. Otherwise `ValueError`.
- **Output:** `outcome` (float) and `bars_to_exit` (int, −1 when NaN). Near the frame end (fewer
  than `horizon` bars left), the outcome is valid if a barrier is hit within the remaining bars,
  else NaN.
- **Close-based, matching `labels_gbb_path`:** a bar has `outcome == target` if and only if
  `labels_gbb_path`, with its rule gates disabled and the same target, stop and horizon, labels it
  1. The equivalence is asserted on bars with a full horizon. The near-end rule is tested
  directly.
- **Pure:** no I/O, and it does not modify its inputs.

## `PooledLabel.relabel(frames, labels, spec, fit_ends=None) -> (labels_out, report)`

- **Input:**
  - `frames`: the raw per-pair frames, as buffered by `maybe_train`;
  - `labels`: the per-pair labels in the format `prepare_training_data` receives;
  - `spec`: a PooledLabelSpec.
- **Effects:**
  - writes `%wr_keep` (1.0 for an entry- or exit-eligible bar with a valid outcome, else NaN) into
    each frame;
  - returns new label arrays: entry-eligible valid bars become BUY or HOLD, exit-eligible valid bars
    become SELL or HOLD, and every other bar keeps its label.
- **Buckets:** computed once per side, over the pooled eligible valid bars of the fit rows: the
  rows `[:fit_ends[i]]` of each frame (`None` = all rows). The engine passes exactly the training
  rows `prepare_training_data` uses: `int(TRAIN_DATA_SPLIT × n) − (seq_len − 1)` per frame. The edges and cuts are applied unchanged to
  the remaining (validation) rows.
- **Determinism:** the same inputs give the same output. The ranking key is
  `(outcome, tie-break, pair index, row)`:
  - descending on the buy side, where the top `share` is BUY;
  - ascending on the sell side, where the bottom `share` is SELL.
- **Errors:**
  - `ValueError` on a bad spec, or when no eligible valid bars exist;
  - `KeyError` when a declared column is missing.
- **Absent spec:** the caller does not call it, so behaviour is byte-identical.

## TrainingEngine hook

In `_train_keyed` (passing `fit_ends` per frame, as above):
- when `getattr(self, "POOLED_LABEL", None)` is not None, `relabel` runs on the buffered frames and
  labels before `train_model`;
- `self._pooled_label_report` is set and written as `width_relative.json` beside the weights (as
  032's `row_filter.json`);
- with no spec declared, no code path changes.

**Tests (Framework/tests/test_pooled_label.py):**
- `path_outcome` against `labels_gbb_path` on a scripted and a real frame;
- hand-computed buckets and labels on a scripted 2-pair frame;
- every bucket is about 50% after the split, including ties at the target;
- with `fit_ends`, the bucket edges and cuts are unchanged when only validation rows' outcomes
  change, and validation rows are labelled by those cuts;
- NaN outcomes give `%wr_keep` NaN and are excluded by the row filter;
- the sell side: `path_outcome(side="sell")` against `labels_gbb_path_sell`; SELL about 50% per
  exit bucket; entry and exit eligibility disjoint (an overlap raises);
- the spec-absent build is byte-identical to the 032 fixture;
- the model key changes only for classes declaring the spec;
- `ValueError` and `KeyError` paths.
