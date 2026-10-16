"""037 pooled panel: the 10 scored pairs' cache rows concatenated, read a few columns at a time
(research R13: screens read only the columns they need). BTC columns join by timestamp as `btc:`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H

#: always loaded: what defines the scored rows and the label targets
CORE = ["meta:data_ok", "base:gbb_candidate", "label:buy", "label:sell", "close"]


def screen_pairs() -> list[str]:
    return [p for p in json.loads(H.MANIFEST.read_text())["pairs"] if p != H.BTC]


def scored_columns(include_btc: bool = False) -> list[str]:
    """Every harvested value / raw-signal column (causality FAILs were never cached)."""
    m = json.loads(H.MANIFEST.read_text())
    cols = [c["name"] for c in m["columns"]
            if c["kind"] in ("value", "raw_buy", "raw_sell", "candidate") and c["causality"] != "FAIL"]
    return cols + ([f"btc:{c}" for c in cols if c != "base:gbb_candidate"] if include_btc else [])


class Panel:
    def __init__(self, pairs: list[str] | None = None, cache=H.CACHE):
        self.pairs = pairs or screen_pairs()
        self.cache = cache
        frames = [H.read_cache(p, CORE, cache).assign(pair=i) for i, p in enumerate(self.pairs)]
        self.lengths = [len(f) for f in frames]
        core = pd.concat(frames, ignore_index=True)
        self.date = pd.DatetimeIndex(core["date"])
        self.pair = core["pair"].to_numpy(np.int16)
        self.data_ok = core["meta:data_ok"].to_numpy() > 0
        self.candidate = core["base:gbb_candidate"].to_numpy() > 0
        self.core = {c: core[c].to_numpy(np.float64) for c in CORE if c in core}
        self._btc_pos = None

    def __len__(self) -> int:
        return len(self.date)

    def columns(self, names: list[str]) -> dict[str, np.ndarray]:
        """{name: pooled float32 array}; a column absent for a pair is NaN there."""
        own = [n for n in names if not n.startswith("btc:")]
        btc = [n[4:] for n in names if n.startswith("btc:")]
        out = {n: np.full(len(self), np.nan, np.float32) for n in own}
        start = 0
        for p, n in zip(self.pairs, self.lengths):
            f = H.read_cache(p, own, self.cache)
            for c in own:
                if c in f:
                    out[c][start: start + n] = f[c].to_numpy(np.float32)
            start += n
        if btc:
            out.update({f"btc:{k}": v for k, v in self.btc(btc).items()})
        return out

    def btc(self, names: list[str]) -> dict[str, np.ndarray]:
        """BTC/USDT's columns at each panel row's timestamp (NaN where BTC has no bar)."""
        f = H.read_cache(H.BTC, ["meta:data_ok", *names], self.cache)
        if self._btc_pos is None:
            bd = pd.DatetimeIndex(f["date"])
            pos = bd.get_indexer(self.date)
            self._btc_pos = pos
        pos = self._btc_pos
        ok = (pos >= 0)
        okb = f["meta:data_ok"].to_numpy() > 0
        out = {}
        for c in names:
            v = np.full(len(self), np.nan, np.float32)
            if c in f:
                src = f[c].to_numpy(np.float32)
                good = ok.copy()
                good[ok] &= okb[pos[ok]]
                v[good] = src[pos[good]]
            out[c] = v
        return out
