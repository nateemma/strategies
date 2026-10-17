# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""HGB sizer study (research/hgb_sizer/): `NNNC_CS_ARO` with the agree score taken from the 037
label:buy HGB instead of ARO's p_buy. Entries, weights (model key = cs_aro's), stakes and the
0.5-share threshold rule are cs_aro's; only `sizer_score` differs.

The fold's HGB p_buy comes from user_data/data/hgb_sizer/HGB_fNN.parquet (research/hgb_sizer/
signals.py), fold id from the generated class name. A bar without a row scores 0 (disagree)."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent))

from NNNC_CS_ARO import NNNC_CS_ARO


class NNNC_CS_HGB(NNNC_CS_ARO):
    HGB_SIGNALS = Path(__file__).resolve().parents[2] / "data" / "hgb_sizer"
    _hgb_table = None

    def _hgb(self) -> pd.DataFrame:
        if self._hgb_table is None:
            m = re.search(r"_(f\d{2})(?:_|$)", type(self).__name__)
            if not m:
                raise ValueError(f"{type(self).__name__}: no fold id in the class name")
            t = pd.read_parquet(self.HGB_SIGNALS / f"HGB_{m.group(1)}.parquet")
            t["date"] = pd.to_datetime(t["date"], utc=True)
            self._hgb_table = t.set_index(["pair", "date"])["p_buy"].sort_index()
        return self._hgb_table

    def sizer_score(self, dataframe):
        t = self._hgb()
        s = t.loc[self.curr_pair] if self.curr_pair in t.index.get_level_values(0) else pd.Series(dtype=float)
        p = pd.to_datetime(dataframe["date"], utc=True).map(s).to_numpy(float)
        dataframe["sizer_score"] = p
        return np.nan_to_num(p, nan=0.0)
