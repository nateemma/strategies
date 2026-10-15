"""New model inputs from a precomputed per-pair cache (feature 035, contract new_input_mixin).

The inputs (research/new_inputs/inputs.py) are built offline, causally, into
`user_data/data/new_inputs/<PAIR>.parquet` and are already normalised to [-1, 1] with no fitted
artefact (035 research R1). A class lists the tags it uses in NEW_INPUTS and declares the cache
definition it was built against in NEW_INPUTS_SHA; generated survivor classes also extend
`include_list` and `pre_normalized_columns` with the input's columns, so the control's scaler (6
robust-scaled columns) is read unchanged.

  * NEW_INPUTS == () -- add_additional_indicators returns the parent's frame unchanged;
  * otherwise the declared columns are left-merged on `date` after the parent's indicators; row
    count and order are unchanged and absent rows are NaN (the framework fills 0).
  * refuses a missing cache, a cache built from another input table, or absent columns -- never
    trains on silent zeros.

The model key: NEW_INPUTS and NEW_INPUTS_SHA are key attributes, and a class with this mixin
defines add_additional_indicators, so it never shares a key with a class without it. Classes that
do not use the mixin are unaffected.
"""

import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "new_inputs"


class NewInputMixin:
    NEW_INPUTS = ()
    NEW_INPUTS_SHA = None
    NEW_INPUT_COLUMNS = ()
    MODEL_KEY_ATTRIBUTES = ("NEW_INPUTS", "NEW_INPUTS_SHA")
    # derived from NEW_INPUTS, and already in the key through include_list
    MODEL_KEY_IGNORED = ("NEW_INPUT_COLUMNS",)

    _new_input_cache = None

    def _new_input_frame(self, pair: str) -> DataFrame:
        cache = type(self)._new_input_cache
        if cache is None:
            cache = type(self)._new_input_cache = {}
        if pair not in cache:
            meta_p = CACHE_DIR / "build_meta.json"
            if not meta_p.is_file():
                raise ValueError(f"{type(self).__name__}: no new-input cache at {CACHE_DIR}")
            meta = json.loads(meta_p.read_text())
            if meta.get("table_sha") != self.NEW_INPUTS_SHA:
                raise ValueError(f"{type(self).__name__}: cache built from input table "
                                 f"{meta.get('table_sha')}, class declares {self.NEW_INPUTS_SHA}")
            p = CACHE_DIR / f"{pair.replace('/', '_')}.parquet"
            if not p.is_file():
                raise ValueError(f"{type(self).__name__}: no new-input cache for {pair}")
            df = pd.read_parquet(p, columns=["date", *self.NEW_INPUT_COLUMNS])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            cache[pair] = df
        return cache[pair]

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super().add_additional_indicators(dataframe)
        if not self.NEW_INPUTS:
            return dataframe
        cols = list(self.NEW_INPUT_COLUMNS)
        if not cols:
            raise ValueError(f"{type(self).__name__}: NEW_INPUTS set but NEW_INPUT_COLUMNS empty")
        src = self._new_input_frame(self.curr_pair)
        n = len(dataframe)
        dates = pd.to_datetime(dataframe["date"], utc=True)
        merged = pd.DataFrame({"date": dates}).merge(src, on="date", how="left")
        if len(merged) != n:
            raise ValueError(f"{type(self).__name__}: cache has duplicate dates")
        for col in cols:
            dataframe[col] = merged[col].to_numpy(dtype=float)
        return dataframe
