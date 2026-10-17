"""Extra model inputs for the ATR-barrier study (research/tb_model/): the 037 composite's member
columns, precomputed causally into user_data/data/tb_inputs/<PAIR>.parquet by
research/tb_model/inputs.py and normalised to [-1, 1] without a fitted artefact (trailing z-score,
as 035's inputs). Same contract as `_new_inputs.NewInputMixin`, own cache: TB_INPUTS_SHA must equal
the cache's build_meta table_sha; absent rows are NaN (the framework fills 0); a missing cache or
column raises. TB_INPUTS_SHA is a model-key attribute."""

import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame


CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "tb_inputs"


class TbInputMixin:
    TB_INPUTS_SHA = None
    TB_INPUT_COLUMNS = ()
    MODEL_KEY_ATTRIBUTES = ("TB_INPUTS_SHA", "TB_INPUT_COLUMNS")

    _tb_cache = None

    def _tb_frame(self, pair: str) -> DataFrame:
        cache = type(self)._tb_cache
        if cache is None:
            cache = type(self)._tb_cache = {}
        if pair not in cache:
            meta = json.loads((CACHE_DIR / "build_meta.json").read_text())
            if meta.get("table_sha") != self.TB_INPUTS_SHA:
                raise ValueError(f"{type(self).__name__}: tb_inputs cache {meta.get('table_sha')}, "
                                 f"class declares {self.TB_INPUTS_SHA}")
            p = CACHE_DIR / f"{pair.replace('/', '_')}.parquet"
            if not p.is_file():
                raise ValueError(f"{type(self).__name__}: no tb_inputs cache for {pair}")
            df = pd.read_parquet(p, columns=["date", *self.TB_INPUT_COLUMNS])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            cache[pair] = df
        return cache[pair]

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super().add_additional_indicators(dataframe)
        cols = list(self.TB_INPUT_COLUMNS)
        if not cols:
            raise ValueError(f"{type(self).__name__}: TB_INPUT_COLUMNS empty")
        merged = pd.DataFrame({"date": pd.to_datetime(dataframe["date"], utc=True)}).merge(
            self._tb_frame(self.curr_pair), on="date", how="left")
        if len(merged) != len(dataframe):
            raise ValueError(f"{type(self).__name__}: tb_inputs cache has duplicate dates")
        for col in cols:
            dataframe[col] = merged[col].to_numpy(dtype=float)
        return dataframe
