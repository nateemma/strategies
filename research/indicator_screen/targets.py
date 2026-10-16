"""037 targets (research R4, R11): the gbb_path labels and the regime targets A1/A2/B/C.

Labels (`--labels`): the ARO control class's own `get_training_labels` on each pair's full-range
frame (the classes the ARO model trains on), stored into the harvest cache as `label:buy` /
`label:sell` (NaN for the last 48 bars, which have no future).

Regime targets are built on demand by `regime_target` (they depend on the fold's cut points).

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.targets --labels
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W


# ---------------------------------------------------------------------------- labels


def labels_for(aro, ohlcv: pd.DataFrame, pair: str) -> tuple[np.ndarray, np.ndarray]:
    """(buy, sell) float arrays in {0, 1, NaN}: the ARO control's trained classes."""
    from Framework.BaseStrategy import TradingAction

    d = H.populate_path(aro, ohlcv, pair)
    k = np.asarray(aro.get_training_labels(d.copy()))[-len(d):]
    buy = (k == int(TradingAction.BUY)).astype(float)
    sell = (k == int(TradingAction.SELL)).astype(float)
    buy[-W.LABEL_HORIZON:] = np.nan
    sell[-W.LABEL_HORIZON:] = np.nan
    return buy, sell


def add_labels(pairs: list[str]) -> None:
    H._quiet()
    cfg = H.base_config()
    aro = H.load_aro()
    for pair in pairs:
        frame = H.read_cache(pair)
        buy, sell = labels_for(aro, H.load_ohlcv(cfg, pair), pair)
        if len(buy) != len(frame):
            raise SystemExit(f"{pair}: label rows {len(buy)} != cache rows {len(frame)}")
        frame["label:buy"] = buy.astype(np.float32)
        frame["label:sell"] = sell.astype(np.float32)
        sc = json.loads(H.sidecar_path(pair).read_text())
        frame.to_feather(H.cache_path(pair))
        sc["columns"] = [c for c in frame.columns if c not in ("date", *H.OHLCV)]
        sc["labels"] = {"class": H.ARO_CLASS, "buy_share": float(np.nanmean(buy)),
                        "sell_share": float(np.nanmean(sell))}
        H.sidecar_path(pair).write_text(json.dumps(sc, indent=1))
        print(f"{pair}: buy {np.nanmean(buy):.4f} sell {np.nanmean(sell):.4f}", flush=True)


# ---------------------------------------------------------------------------- regime (R11)

HORIZONS = (48, 288, 672)
FAMILIES = ("A1", "A2", "B-btc", "B-pair", "C-btc", "C-pair")
MARKET_WIDE = ("A1", "A2", "B-btc", "C-btc")


def regime_ids(horizons=HORIZONS) -> tuple[str, ...]:
    return tuple(f"{f}:{n}" for n in horizons for f in FAMILIES)


def _prefix(v: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(v)])


def forward_mean(s: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Per grid index t: sum(s[t+1..t+n]) / count(c[t+1..t+n]); NaN when no event or past the end."""
    P, Q = _prefix(s), _prefix(c)
    g = len(s)
    t = np.arange(g)
    hi = np.minimum(t + n + 1, g)
    num, den = P[hi] - P[t + 1], Q[hi] - Q[t + 1]
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    out[t + n >= g] = np.nan
    return out


def trailing_mean(s: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Per grid index t: sum over events at indices [t-n, t-1] / count; NaN when none."""
    P, Q = _prefix(s), _prefix(c)
    t = np.arange(len(s))
    lo = np.maximum(t - n, 0)
    num, den = P[t] - P[lo], Q[t] - Q[lo]
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    out[t < n] = np.nan
    return out


def forward_return(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[:-n] = np.log(close[n:] / close[:-n])
    return out


def trailing_return(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[n:] = np.log(close[n:] / close[:-n])
    return out


def forward_vol(close: np.ndarray, n: int) -> np.ndarray:
    """std of the 15m log returns r[t+1..t+n] (r[i] = log(close[i] / close[i-1]))."""
    r = pd.Series(np.r_[np.nan, np.diff(np.log(close))])
    back = r.rolling(n, min_periods=n).std(ddof=0).to_numpy()    # std of r[i-n+1..i]
    out = np.full(len(close), np.nan)
    out[:-n] = back[n:]
    return out


def trailing_vol(close: np.ndarray, n: int) -> np.ndarray:
    """std of r[t-n+1..t]: the returns known at t."""
    r = pd.Series(np.r_[np.nan, np.diff(np.log(close))])
    return r.rolling(n, min_periods=n).std(ddof=0).to_numpy()


class RegimeTargets:
    """Regime targets A1/A2/B/C at each horizon (R11). `y(target, window)` = top tercile 1, bottom
    0, middle and not-computable NaN, with cut points from the window's training span (E0: its own
    rows; A2: the early test windows, pre-registered). `baseline(target)` = the same quantity over
    the trailing N bars (closed trades only for A1/A2)."""
    kind = "regime"

    def __init__(self, panel, only=None, a1_events=None, a2_trades=None, horizons=HORIZONS):
        self.p = panel
        self.ids = tuple(only) if only else regime_ids(horizons)
        self.market_wide = tuple(t for t in self.ids if t.split(":")[0] in MARKET_WIDE)
        self.t0 = pd.Timestamp(W.DATA_START, tz="UTC")
        self.grid_n = int((pd.Timestamp(W.DATA_END, tz="UTC") - self.t0) / W.BAR)
        self.gi = np.asarray((panel.date - self.t0) / W.BAR).astype(np.int64)
        self._a1_events = a1_events or globals()["a1_events"](panel.pairs)
        self._a2_trades = a2_trades or globals()["a2_trades"]()
        self._q, self._b, self._cuts = {}, {}, {}

    # -- forward quantity and baseline, per panel row
    def _grid_events(self, fam):
        g = self.grid_n
        so, co, sc, cc = (np.zeros(g) for _ in range(4))
        ev = self._a1_events() if fam == "A1" else self._a2_trades()
        for oi, ci, prof in ev:              # open grid index, close grid index, profit
            if 0 <= oi < g:
                so[oi] += prof
                co[oi] += 1
            if 0 <= ci < g:
                sc[ci] += prof
                cc[ci] += 1
        return so, co, sc, cc

    def _btc_close_grid(self):
        f = H.read_cache(H.BTC, ["close"], self.p.cache)
        gi = np.asarray((pd.DatetimeIndex(f["date"]) - self.t0) / W.BAR).astype(np.int64)
        c = np.full(self.grid_n, np.nan)
        c[gi] = f["close"].to_numpy(float)
        return pd.Series(c).ffill().to_numpy()

    def _compute(self, target):
        fam, n = target.split(":")[0], int(target.split(":")[1])
        if fam in ("A1", "A2"):
            so, co, sc, cc = self._grid_events(fam)
            q, b = forward_mean(so, co, n)[self.gi], trailing_mean(sc, cc, n)[self.gi]
            if fam == "A2":                  # width-book trades exist only inside fold test windows
                inside = np.zeros(len(self.p), bool)
                for f in W.folds():
                    inside |= f.mask(self.p.date)
                q[~inside] = np.nan
        elif fam.endswith("-btc"):
            c = self._btc_close_grid()
            fw, bw = (forward_return, trailing_return) if fam == "B-btc" else (forward_vol, trailing_vol)
            q, b = fw(c, n)[self.gi], bw(c, n)[self.gi]
        else:
            close = self.p.core["close"]
            q, b = np.full(len(self.p), np.nan), np.full(len(self.p), np.nan)
            fw, bw = (forward_return, trailing_return) if fam == "B-pair" else (forward_vol, trailing_vol)
            start = 0
            for ln in self.p.lengths:          # each pair's rows are a contiguous 15m grid
                sl = slice(start, start + ln)
                q[sl], b[sl] = fw(close[sl], n), bw(close[sl], n)
                start += ln
        self._q[target], self._b[target] = q, b

    def quantity(self, target):
        if target not in self._q:
            self._compute(target)
        return self._q[target]

    def baseline(self, target, window=None):
        if target not in self._b:
            self._compute(target)
        return self._b[target]

    def cuts(self, target, window):
        key = (target, window.id if not target.startswith("A2:") else "A2")
        if key not in self._cuts:
            q = self.quantity(target)
            n = W.purge(target)
            if target.startswith("A2:"):
                m = np.zeros(len(self.p), bool)
                for f in W.folds():
                    if f.half == "early":
                        m |= f.mask(self.p.date)
            elif window.train_start is None:          # E0: its own rows (descriptive)
                m = window.mask(self.p.date)
            else:
                m = np.asarray((self.p.date >= window.train_start) & (self.p.date < window.start - n * W.BAR))
            v = q[m & self.p.data_ok & np.isfinite(q)]
            self._cuts[key] = (np.quantile(v, 1 / 3), np.quantile(v, 2 / 3)) if len(v) else (np.nan, np.nan)
        return self._cuts[key]

    def y(self, target, window):
        q = self.quantity(target)
        lo, hi = self.cuts(target, window)
        out = np.full(len(q), np.nan)
        out[q >= hi] = 1.0
        out[q <= lo] = 0.0
        out[~np.isfinite(q)] = np.nan
        return out

    def columns_for(self, target, cols):
        return cols

    def baseline_scores(self, panel, wins, pr, out_dir=None) -> pd.DataFrame:
        from research.indicator_screen import scores as S
        from research.indicator_screen.screen_single import POPS, rows

        recs = []
        for t in self.ids:
            b = self.baseline(t)
            for w in wins:
                y = self.y(t, w)
                for pop in POPS:
                    idx = rows(panel, w, pop)
                    recs.append({"target": t, "population": pop, "window": w.id,
                                 "auc": S.auc(b[idx], y[idx])})
        return pd.DataFrame(recs)


def a1_events(pairs, t0=None):
    """Outcome-table rows on gbb candidate bars (Open operating point): (open idx, close idx, profit)."""
    def load():
        from research.outcomes.leakage import OUTCOME_COLUMN_HOLD_BARS, OUTCOME_COLUMN_PROFIT
        from research.outcomes.table import ExitConfig, load_table

        ec = ExitConfig.from_operating_point()
        start = t0 or pd.Timestamp(W.DATA_START, tz="UTC")
        out = []
        for p in pairs:
            t = load_table(p, ec)
            c = H.read_cache(p, ["base:gbb_candidate", "meta:data_ok"])
            m = t.merge(c, on="date", how="inner")
            m = m[(m["base:gbb_candidate"] > 0) & (m["meta:data_ok"] > 0) & m[OUTCOME_COLUMN_PROFIT].notna()]
            oi = np.asarray((pd.DatetimeIndex(m["date"]) - start) / W.BAR).astype(np.int64)
            ci = oi + m[OUTCOME_COLUMN_HOLD_BARS].fillna(0).to_numpy(np.int64)
            out.extend(zip(oi, ci, m[OUTCOME_COLUMN_PROFIT].to_numpy(float)))
        return out
    return load


def a2_trades(t0=None):
    """The width book's (w_aro) recorded trades: (open idx, close idx, profit_ratio)."""
    def load():
        from research.live_pnl.shared_wallet import _fold_trades

        start = t0 or pd.Timestamp(W.DATA_START, tz="UTC")
        out = []
        for trades in _fold_trades("w_aro").values():
            for tr in trades:
                oi = int((pd.Timestamp(tr["open_date"]) - start) / W.BAR)
                ci = int((pd.Timestamp(tr["close_date"]) - start) / W.BAR)
                out.append((oi, ci, float(tr["profit_ratio"])))
        return out
    return load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", action="store_true", required=True)
    ap.add_argument("--pair", default="all")
    a = ap.parse_args(argv)
    cfg = H.base_config()
    pl = [p for p in H.pairs(cfg) if p != H.BTC] if a.pair == "all" else [a.pair]
    add_labels(pl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
