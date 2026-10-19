"""038 scorer (research R8, R10; data-model § Trade, § Cell score).

A method's forecasts become long-only trades: signal at t when f > 0 and f >= the causal 0.9 quantile
of the pair's previous 30 days of forecasts; enter open[t+1], exit open[t+1+n]; one position per pair;
only eligible candles. Trades are scored net of fee + slippage against pair-matched random entry, per
half, with a week-block bootstrap lower bound. Skill, IC and hit rate are descriptive only.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.score [--method M]
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from research.forecast import common as C
from research.forecast import frames as F

# a fold where the forecast is constant (hist_mean) has no rank correlation; reported as NaN
warnings.filterwarnings("ignore", category=ConstantInputWarning)

QUANTILE = 0.9
LOOKBACK_DAYS = 30
N_BOOT = 2000
HALVES = ("early", "late", "fwd")


# ---------------------------------------------------------------------------- primitives


def causal_threshold(f: np.ndarray, lookback: int, q: float = QUANTILE) -> np.ndarray:
    """thr[i] = q-quantile of f[i-lookback:i] (strictly before i); NaN until a full lookback."""
    s = pd.Series(f, dtype=float)
    return s.rolling(lookback, min_periods=lookback).quantile(q, interpolation="linear").shift(1).to_numpy()


def long_signals(f: np.ndarray, lookback: int, q: float = QUANTILE) -> np.ndarray:
    thr = causal_threshold(f, lookback, q)
    with np.errstate(invalid="ignore"):
        return (f > 0) & (f >= thr)


def short_signals(f: np.ndarray, lookback: int, q: float = QUANTILE) -> np.ndarray:
    return long_signals(-f, lookback, q)


def simulate(rows: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    """Signal rows (frame row index, ascending) actually traded: eligible and not inside a trade.
    A trade at t holds open[t+1] -> open[t+1+n], so the next entry signal must be at t' >= t + n."""
    out, nxt = [], -1
    for t in rows:
        if t >= nxt and eligible[t]:
            out.append(t)
            nxt = t + n
    return np.asarray(out, dtype=int)


def net(gross: np.ndarray, bps: float) -> np.ndarray:
    return gross - C.cost_rt(bps)


def matched_benchmark(trades: pd.DataFrame, rand: dict) -> float:
    """Mean random-entry gross return, matched to each trade's pair and half."""
    if trades.empty:
        return float("nan")
    return float(np.mean([rand[(p, h)] for p, h in zip(trades["pair"], trades["half"])]))


def bootstrap_lb(x: np.ndarray, dates: pd.Series, n_boot: int = N_BOOT, seed: int = 0,
                 alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) lower bound of mean(x), resampling calendar weeks with replacement."""
    if len(x) == 0:
        return float("nan")
    wk = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).to_period("W").astype(str)
    g = pd.DataFrame({"w": wk, "x": x}).groupby("w")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(np.quantile(means, alpha))


def r2_oos(f: np.ndarray, r: np.ndarray) -> float:
    return float(1.0 - np.sum((r - f) ** 2) / np.sum(r ** 2))


def hit_rate(f: np.ndarray, r: np.ndarray) -> float:
    m = f != 0
    return float(np.mean(np.sign(f[m]) == np.sign(r[m]))) if m.any() else float("nan")


# ---------------------------------------------------------------------------- per method


def load_pred(method: str) -> pd.DataFrame:
    return pd.read_feather(C.PRED / f"{method}.feather")


def _frames(columns: list[str]) -> dict[str, pd.DataFrame]:
    return {p: F.read(p, ["date", "fold", "half", "has_data", *columns]) for p in C.pairs()}


def score_method(method: str, pred: pd.DataFrame | None = None, frames: dict | None = None) -> dict:
    pred = load_pred(method) if pred is None else pred
    cols = [c for n in C.HORIZONS for c in (f"target_{n}", f"trade_ret_{n}", f"eligible_{n}")]
    frames = _frames(cols) if frames is None else frames
    stride = int(pred.attrs.get("stride", 0)) or _infer_stride(pred)
    lookback = LOOKBACK_DAYS * C.BARS_PER_DAY // stride
    out = {"stride": stride, "n": {}}
    for n in C.HORIZONS:
        trades, shorts, desc_rows, rand_rows, years = [], [], [], [], {}
        for pair, fr in frames.items():
            p = pred[pred["pair"] == pair]
            if p.empty:
                continue
            pos = pd.Series(fr.index, index=fr["date"]).reindex(p["date"]).to_numpy()
            p = p.assign(row=pos).dropna(subset=["row"]).sort_values("row")
            f = p[f"f_{n}"].to_numpy(float)
            r_idx = p["row"].to_numpy(int)
            ok = np.isfinite(f)
            f_ok, r_ok = f[ok], r_idx[ok]
            el = fr[f"eligible_{n}"].to_numpy()
            gross_all = fr[f"trade_ret_{n}"].to_numpy()
            half = fr["half"].to_numpy()
            if f"sig_{n}" in p.columns:  # a pre-registered signal of its own (the N-of-M votes)
                sig_rows = (r_idx[p[f"sig_{n}"].to_numpy(bool)], r_idx[p[f"sigs_{n}"].to_numpy(bool)])
            else:
                sig_rows = (r_ok[long_signals(f_ok, lookback)], r_ok[short_signals(f_ok, lookback)])
            for rows_, sink, sign in ((sig_rows[0], trades, 1.0), (sig_rows[1], shorts, -1.0)):
                t = simulate(rows_, el, n)
                sink.append(pd.DataFrame({"pair": pair, "row": t, "date": fr["date"].to_numpy()[t],
                                          "half": half[t], "fold": fr["fold"].to_numpy()[t],
                                          "gross": sign * gross_all[t]}))
            tgt = fr[f"target_{n}"].to_numpy()[r_ok]
            desc_rows.append(pd.DataFrame({"pair": pair, "half": half[r_ok], "fold": fr["fold"].to_numpy()[r_ok],
                                           "f": f_ok, "r": tgt, "traded": el[r_ok]}))
            rr = r_idx[el[r_idx]]
            rand_rows.append(pd.DataFrame({"pair": pair, "half": half[rr], "gross": gross_all[rr]}))
            hd = fr["has_data"].to_numpy()
            for h in HALVES:
                years[(pair, h)] = float(np.sum(hd & (half == h))) / (C.BARS_PER_DAY * 365.0)
        tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        sh = pd.concat(shorts, ignore_index=True) if shorts else pd.DataFrame()
        de = pd.concat(desc_rows, ignore_index=True)
        ra = pd.concat(rand_rows, ignore_index=True)
        rand = ra.groupby(["pair", "half"])["gross"].mean().to_dict()
        fail = _fail_share(pred)
        out["n"][str(n)] = {h: _cell(tr, sh, de, rand, years, h, fail) for h in HALVES}
    return out


def _infer_stride(pred: pd.DataFrame) -> int:
    d = pred.sort_values(["pair", "date"]).groupby("pair")["date"].diff().dropna()
    step = d.dt.total_seconds().median() / 900.0 if len(d) else 1.0
    return max(1, int(round(step)))


def _fail_share(pred: pd.DataFrame) -> float:
    tried = ~pred.get("skipped_gap", pd.Series(False, index=pred.index)).astype(bool)
    failed = pred.get("failed", pd.Series(False, index=pred.index)).astype(bool)
    return float(failed[tried].mean()) if tried.any() else 0.0


def _stats(tr: pd.DataFrame, sh: pd.DataFrame, de: pd.DataFrame, rand: dict, yrs: float) -> dict:
    g = tr["gross"].to_numpy() if len(tr) else np.array([])
    out = {"n_trades": int(len(g)), "mean_gross": float(g.mean()) if len(g) else None}
    for s in C.SLIPPAGE_BPS:
        out[f"mean_net_{s}"] = float(net(g, s).mean()) if len(g) else None
    rb = matched_benchmark(tr, rand) if len(tr) else float("nan")
    out["rand_gross"] = None if np.isnan(rb) else rb
    out["edge_vs_rand"] = (out["mean_gross"] - rb) if len(g) else None
    out["net_lb95"] = bootstrap_lb(net(g, C.VERDICT_BPS), tr["date"]) if len(g) else None
    out["net_per_pair_year"] = float(net(g, C.VERDICT_BPS).sum() / yrs) if len(g) and yrs > 0 else None
    sg = sh["gross"].to_numpy() if len(sh) else np.array([])
    out["short_n"] = int(len(sg))
    out["short_mean_net_5.0"] = float(net(sg, C.VERDICT_BPS).mean()) if len(sg) else None
    for tag, d in (("all", de), ("traded", de[de["traded"]])):
        d = d[np.isfinite(d["r"])]
        if len(d) < 10:
            continue
        ics = [spearmanr(x["f"], x["r"]).statistic for _, x in d.groupby("fold") if len(x) > 10 and x["f"].std() > 0]
        out[f"r2_oos_{tag}"] = r2_oos(d["f"].to_numpy(), d["r"].to_numpy())
        out[f"ic_median_{tag}"] = float(np.nanmedian(ics)) if ics else None
        out[f"hit_rate_{tag}"] = hit_rate(d["f"].to_numpy(), d["r"].to_numpy())
        out[f"n_rows_{tag}"] = int(len(d))
    return out


def _cell(tr, sh, de, rand, years, h, fail) -> dict:
    sel = lambda x: x[x["half"] == h] if len(x) else x  # noqa: E731
    t, s, d = sel(tr), sel(sh), sel(de)
    pairs_ = sorted(set(d["pair"]))
    yrs = sum(v for (p, hh), v in years.items() if hh == h)
    cell = _stats(t, s, d, rand, yrs) | {"fail_share": fail}
    cell["pairs"] = {p: _stats(t[t["pair"] == p] if len(t) else t, s[s["pair"] == p] if len(s) else s,
                               d[d["pair"] == p], rand, years.get((p, h), 0.0)) for p in pairs_}
    return cell


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append")
    a = ap.parse_args(argv)
    methods = a.method or sorted(p.stem for p in C.PRED.glob("*.feather"))
    path = C.HERE / "scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    cols = [c for n in C.HORIZONS for c in (f"target_{n}", f"trade_ret_{n}", f"eligible_{n}")]
    frames = _frames(cols)
    for m in methods:
        scores[m] = score_method(m, frames=frames)
        for n in C.HORIZONS:
            e, l = scores[m]["n"][str(n)]["early"], scores[m]["n"][str(n)]["late"]
            print(f"{m:12s} n={n:3d} trades {e['n_trades']:6d}/{l['n_trades']:6d} "
                  f"net5 {_bp(e['mean_net_5.0'])}/{_bp(l['mean_net_5.0'])} edge {_bp(e['edge_vs_rand'])}/"
                  f"{_bp(l['edge_vs_rand'])} lb95(late) {_bp(l['net_lb95'])} r2(late) {l.get('r2_oos_traded')}")
        C.write_json(path, scores)
    return 0


def _bp(x) -> str:
    """A log return in basis points, for the summary line."""
    return "    -   " if x is None else f"{x * 1e4:+7.1f}bp"


if __name__ == "__main__":
    raise SystemExit(main())
