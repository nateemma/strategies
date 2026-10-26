"""Mixed sizer score as a mixin (research/xmodel_sizer/): agree score = mean of the model's p_buy
percentile and the 037 'dip' composite percentile, both ranked against the fold's validation-tail
width-book run starts (MIX_KNOTS/MIX_fNN.json). The composite comes from the model-free tables in
user_data/data/cmp_sizer (research/cmp_sizer/signals.py). Put it before WidthSizerMixin in the MRO.
A bar missing either score reads 0 (disagree). Fold id from the class name (_fNN)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def pct(x, knots) -> np.ndarray:
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), np.searchsorted(k, x, side="right") / len(k), np.nan)


class MixSizerMixin:
    MODEL_KEY_IGNORED = ("CMP_TABLES", "MIX_KNOTS", "_knots")  # the sizer's tables / knots: post-model
    CMP_TABLES = DATA / "cmp_sizer"
    MIX_KNOTS: Path = None          # per class: DATA / "mix_sizer_<model>"
    _cmp_table = None
    _knots = None

    def _fold_id(self) -> str:
        m = re.search(r"_(f\d{2})(?:_|$)", type(self).__name__)
        if not m:
            raise ValueError(f"{type(self).__name__}: no fold id in the class name")
        return m.group(1)

    def _composite(self, dataframe) -> np.ndarray:
        if self._cmp_table is None:
            t = pd.read_parquet(self.CMP_TABLES / f"HGB_{self._fold_id()}.parquet")
            t["date"] = pd.to_datetime(t["date"], utc=True)
            self._cmp_table = t.set_index(["pair", "date"])["p_buy"].sort_index()
        t = self._cmp_table
        s = t.loc[self.curr_pair] if self.curr_pair in t.index.get_level_values(0) else pd.Series(dtype=float)
        return pd.to_datetime(dataframe["date"], utc=True).map(s).to_numpy(float)

    def sizer_score(self, dataframe):
        if self._knots is None:
            self._knots = json.loads((self.MIX_KNOTS / f"MIX_{self._fold_id()}.json").read_text())
        k = self._knots
        p = 0.5 * (pct(dataframe["%predict_prob_buy"].to_numpy(float), k["model"])
                   + pct(self._composite(dataframe), k["cmp"]))
        dataframe["sizer_score"] = p
        return np.nan_to_num(p, nan=0.0)
