# Contract: `NewInputMixin` (`NNNC/_new_inputs.py`)

```text
class NewInputMixin:
    NEW_INPUTS: tuple[str, ...] = ()          # default: no effect, byte-identical
    MODEL_KEY_ATTRIBUTES = ("NEW_INPUTS",)
    def add_additional_indicators(self, dataframe, ...) -> DataFrame
```

**Guarantees:**
- **With `NEW_INPUTS == ()`:** `add_additional_indicators` returns `super()`'s frame unchanged,
  and include_list, pre_normalized_columns and the model key are those of the parent.
- **Otherwise:** after `super()`, it left-merges the declared `ni_*` columns (and flags) from
  `user_data/data/new_inputs/<PAIR>.parquet` on `date`.
  - The row count and order are unchanged, and no other column changes.
  - Absent rows give NaN, which the framework fills with 0.
- **Refusal:** it raises when the cache is missing, or when its `build_meta.json` input-table sha
  differs from the declared one. It never trains on silent zeros.
- **Generated survivor classes** set `include_list` and `pre_normalized_columns` as new lists
  (parent + columns). The mixin never mutates a parent list.
- **No Framework/ change:** the mixin lives in `NNNC/`, and no pair-specific code is allowed.

**Tests:**
- byte identity: the mixin with empty `NEW_INPUTS` over `NNNC_A_Raw_Open` gives the same model key
  and the same populated frame on a sample pair;
- the survivor key differs from the control's;
- the main scaler's 6 columns are unchanged;
- a merge on a frame with gaps keeps the row count;
- a stale-cache refusal;
- end-to-end causality (constitution V): the merged column at a cut equals the value from a
  cache built on sources truncated at that cut.
