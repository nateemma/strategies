# Contract: `NewInputMixin` (`NNNC/_new_inputs.py`)

```text
class NewInputMixin:
    NEW_INPUTS: tuple[str, ...] = ()          # default: no effect on the frame
    NEW_INPUTS_SHA: str | None = None          # the input table the cache must match
    NEW_INPUT_COLUMNS: tuple[str, ...] = ()    # the ni_* columns (value + flag)
    MODEL_KEY_ATTRIBUTES = ("NEW_INPUTS", "NEW_INPUTS_SHA")
    def add_additional_indicators(self, dataframe, ...) -> DataFrame
```

**Guarantees:**
- **With `NEW_INPUTS == ()`:** `add_additional_indicators` returns `super()`'s frame unchanged,
  and include_list and pre_normalized_columns are those of the parent.
- **Model key:** a class with the mixin never shares the parent's key. `NEW_INPUTS` and
  `NEW_INPUTS_SHA` are key attributes, and the key also hashes the class that defines
  `add_additional_indicators`. "Default off" therefore means that every class without the mixin
  is unaffected, which is tested on the existing-key fixture. (Amended in implementation: the
  earlier "same key when empty" guarantee cannot hold under feature 030's method hashing.)
- **`NEW_INPUTS_SHA`:** the input table the class was built against. The mixin refuses a cache
  whose `build_meta.json` table sha differs.
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
- off: the mixin with empty `NEW_INPUTS` over `NNNC_A_Raw_Open` returns the parent's frame
  unchanged with the parent's include_list; every pre-034 model key is unchanged;
- the survivor key differs from the control's;
- the main scaler's 6 columns are unchanged;
- a merge on a frame with gaps keeps the row count;
- a stale-cache refusal;
- end-to-end causality (constitution V): the merged column at a cut equals the value from a
  cache built on sources truncated at that cut.
