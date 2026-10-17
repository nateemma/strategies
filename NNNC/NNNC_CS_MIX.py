# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Mixed sizer study (research/mix_sizer/): agree score = mean of ARO's p_buy percentile and the 037 'dip'
composite percentile, both ranked against the fold's validation-tail width-book run starts
(user_data/data/mix_sizer/MIX_fNN.json, research/mix_sizer/knots.py). Composite from cs_cmp's tables.
Entries, weights (model key = cs_aro's), stakes and the 0.5-share threshold rule are cs_aro's.
A bar missing either score reads 0 (disagree)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_CMP import NNNC_CS_CMP


def _pct(x, knots):
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), np.searchsorted(k, x, side="right") / len(k), np.nan)


class NNNC_CS_MIX(NNNC_CS_CMP):
    MIX_KNOTS = Path(__file__).resolve().parents[2] / "data" / "mix_sizer"
    _knots = None

    def _mix_knots(self) -> dict:
        if self._knots is None:
            m = re.search(r"_(f\d{2})(?:_|$)", type(self).__name__)
            if not m:
                raise ValueError(f"{type(self).__name__}: no fold id in the class name")
            self._knots = json.loads((self.MIX_KNOTS / f"MIX_{m.group(1)}.json").read_text())
        return self._knots

    def sizer_score(self, dataframe):
        t = self._hgb()
        s = t.loc[self.curr_pair] if self.curr_pair in t.index.get_level_values(0) else pd.Series(dtype=float)
        c = pd.to_datetime(dataframe["date"], utc=True).map(s).to_numpy(float)
        k = self._mix_knots()
        p = 0.5 * (_pct(dataframe["%predict_prob_buy"].to_numpy(float), k["aro"]) + _pct(c, k["cmp"]))
        dataframe["sizer_score"] = p
        return np.nan_to_num(p, nan=0.0)
