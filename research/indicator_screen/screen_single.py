"""037 US2 + US4 single-column screens (research R6, R7, R8, R11).

Every harvested column is scored against each target, per window, on two populations (all bars;
gbb candidate bars). Pooled across the 10 pairs per window. Per window: rank AUC (strength,
direction), binned normalised MI (edges from the fold's training span; E0 from itself), lift and
fire rate for raw signals, and on candidate bars the stratified AUC within the bb_width x
guard_metric grid (edges from training-span candidate bars) beside the width sort.

`--targets labels` runs the INSTRUMENT CHECK first (label:buy, all bars, gate halves E0 + early):
guard_metric (down) and bb_width (up) must each have the expected direction and strength >= 0.15
in every gate window (amended 2026-10-16), or the screen writes {"instrument": "FAIL"} and exits 1. Late windows need the committed pre-registration (R15).

Window scores: user_data/data/indicator_screen/out/single_<targets>_<half>.feather (not committed).
Summary: research/indicator_screen/single_<targets>.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.screen_single --targets labels --halves e0,early
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from research.indicator_screen import harvest as H
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns

HERE = Path(__file__).resolve().parent
OUT = H.CACHE / "out"
WIDTH, GUARD = "base:bb_width", "base:guard_metric"
POPS = ("all", "candidate")
CHUNK = 48


def prereg() -> dict:
    return json.loads(W.PREREG.read_text())


# ---------------------------------------------------------------------------- targets


class LabelTargets:
    kind = "labels"
    ids = ("label:buy", "label:sell")
    market_wide = ()

    def __init__(self, panel: Panel):
        self.p = panel

    def y(self, target: str, window: W.Window) -> np.ndarray:
        return self.p.core[target]

    def baseline(self, target: str, window: W.Window):
        return None

    def columns_for(self, target: str, cols: list[str]) -> list[str]:
        return [c for c in cols if not c.startswith("btc:")]


# ---------------------------------------------------------------------------- rows


def rows(panel: Panel, window: W.Window, pop: str) -> np.ndarray:
    m = window.mask(panel.date) & panel.data_ok
    if pop == "candidate":
        m &= panel.candidate
    return np.flatnonzero(m)


def train_sample(panel: Panel, window: W.Window, pop: str, n: int, seed: int) -> np.ndarray:
    """Seed-fixed sample of the window's TRAINING rows (E0: its own rows), for edges (R6, R7)."""
    if window.train_start is None:
        idx = rows(panel, window, pop)
    else:
        m = (panel.date >= window.train_start) & (panel.date < window.start) & panel.data_ok
        if pop == "candidate":
            m &= panel.candidate
        idx = np.flatnonzero(m)
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return idx


# ---------------------------------------------------------------------------- scoring


def score_windows(panel, targets, windows, cols, pr) -> pd.DataFrame:
    """One record per column x target x population x window."""
    min_fires, max_rate = pr["min_fires"], pr["max_fire_rate"]
    bins, ngrid = pr["mi_bins"], pr["residual_grid"]
    ctx = {}
    wg = panel.columns([WIDTH, GUARD])
    for w in windows:
        for pop in POPS:
            idx = rows(panel, w, pop)
            tr = train_sample(panel, w, pop, pr["sample_rows"], pr["seed"])
            c = {"idx": idx, "train": tr}
            if pop == "candidate":
                c["ea"] = S.quantile_edges(wg[WIDTH][tr], ngrid)
                c["eb"] = S.quantile_edges(wg[GUARD][tr], ngrid)
                c["a"], c["b"] = wg[WIDTH][idx], wg[GUARD][idx]
            yfull = {t: targets.y(t, w) for t in targets.ids}
            c["y"] = {t: v[idx] for t, v in yfull.items()}
            c["width_auc"] = dict(zip(targets.ids, S.auc_multi(wg[WIDTH][idx], list(c["y"].values()))))
            if targets.market_wide:          # market-wide target x btc: column -> one row per timestamp
                _, first = np.unique(panel.date[idx].asi8, return_index=True)
                c["idx_d"] = idx[np.sort(first)]
                c["y_d"] = {t: yfull[t][c["idx_d"]] for t in targets.market_wide}
            ctx[(w.id, pop)] = c
    recs = []
    for k in range(0, len(cols), CHUNK):
        chunk = cols[k: k + CHUNK]
        data = panel.columns(chunk)
        for col in chunk:
            x_all = data[col]
            sig = col.startswith("raw_") or col == "base:gbb_candidate"
            for w in windows:
                for pop in POPS:
                    c = ctx[(w.id, pop)]
                    tids = [t for t in targets.ids if col in targets.columns_for(t, [col])]
                    if not tids:
                        continue
                    edges = None if sig else S.quantile_edges(x_all[c["train"]], bins)
                    dedup = set(targets.market_wide) if col.startswith("btc:") else set()
                    groups = [([t for t in tids if t not in dedup], c["idx"], c["y"], False),
                              ([t for t in tids if t in dedup], c.get("idx_d"), c.get("y_d"), True)]
                    for g_tids, g_idx, g_y, g_dedup in groups:
                        if g_tids:
                            _score_group(recs, col, sig, x_all[g_idx], g_tids, [g_y[t] for t in g_tids],
                                         w, pop, c, edges, g_dedup, targets.kind, min_fires, max_rate)
        print(f"  scored {min(k + CHUNK, len(cols))}/{len(cols)} columns", flush=True)
    df = pd.DataFrame(recs)
    df["strength"] = (df["auc"] - 0.5).abs()
    df["width_strength"] = (df["width_auc"] - 0.5).abs()
    df["mi_edges_from"] = np.where(df["window"] == "E0", "self", "train")
    return df


def _score_group(recs, col, sig, x, tids, ys, w, pop, c, edges, dedup, kind, min_fires, max_rate):
    aucs = S.auc_multi(x, ys)
    for t, y, a in zip(tids, ys, aucs):
        ok = np.isfinite(x) & np.isfinite(y)
        r = {"column": col, "target": t, "population": pop, "window": w.id,
             "half": w.half, "n_rows": int(ok.sum()), "n_pos": int((y[ok] > 0.5).sum()),
             "auc": a, "width_auc": c["width_auc"][t], "dedup": dedup}
        if sig:
            lf, rate, _ = S.lift(x, y, min_fires, max_rate)
            r.update(lift=lf, fire_rate=rate, mi_norm=S.mi_norm(x, y, [0.5]))
        else:
            r.update(lift=np.nan, fire_rate=np.nan, mi_norm=S.mi_norm(x, y, edges))
        # residual beyond bb_width + guard_metric (FR-015): label targets, candidate bars
        if pop == "candidate" and kind == "labels":
            r["strat_auc"] = S.stratified_auc(x, y, c["a"], c["b"], c["ea"], c["eb"])
        else:
            r["strat_auc"] = np.nan
        recs.append(r)


# ---------------------------------------------------------------------------- instrument


def instrument(df: pd.DataFrame, pr: dict, halves) -> dict:
    """The screen must recover the label's known inputs with the right sign: each instrument column
    has its expected direction and strength >= min_strength in EVERY window of `halves`
    (pre-registration amendment, 2026-10-16). The top-ranked columns are listed for the report."""
    ins = pr["instrument"]
    g = df[(df["target"] == ins["target"]) & (df["population"] == ins["population"])
           & df["half"].isin(halves)]
    med = g[g["n_rows"] >= pr["min_rows"]].groupby("column")["strength"].median().sort_values(ascending=False)
    rank = {c: int(i) + 1 for i, c in enumerate(med.index)}
    wins = sorted(g["window"].unique())
    cols = {}
    for c, want in ins["expected_direction"].items():
        r = g[g["column"] == c].set_index("window")
        per = {w: (round(float(r.at[w, "auc"]), 4) if w in r.index else None) for w in wins}
        ok = all(v is not None and S.direction(v) == want and abs(v - 0.5) >= ins["min_strength"]
                 and r.at[w, "n_rows"] >= pr["min_rows"] for w, v in per.items())
        cols[c] = {"pass": bool(ok), "expected_direction": want, "rank": rank.get(c), "auc_by_window": per}
    return {"verdict": "PASS" if wins and all(v["pass"] for v in cols.values()) else "FAIL",
            "rule": f"expected direction and strength >= {ins['min_strength']} in every window",
            "halves": list(halves), "windows": wins, "columns": cols,
            "top": [{"column": c, "median_strength": round(float(med[c]), 6)} for c in med.index[:10]]}


# ---------------------------------------------------------------------------- summary


def _med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else None


def _dir(v):
    return None if v is None else S.direction(v)


def summarise(df: pd.DataFrame, pr: dict, baseline: pd.DataFrame | None = None) -> list[dict]:
    """ScreenResult rows (data-model) from all window scores available."""
    mr = pr["min_rows"]
    df = df[df["n_rows"] >= mr]
    out = []
    late_done = (df["half"] == "late").any()
    bl = None
    if baseline is not None and len(baseline):
        bl = baseline.set_index(["target", "population", "window"])["auc"]
    for (col, t, pop), g in df.groupby(["column", "target", "population"], sort=True):
        tests = g[g["half"] != "e0"]
        recs = g.to_dict("records")
        agg = S.aggregate(recs, mr, "auc")
        halves = {}
        for h in ("e0", "early", "late"):
            hh = g[g["half"] == h]
            halves[h] = {"median_auc": _med(hh["auc"]), "median_strength": _med(hh["strength"]),
                         "median_strat_auc": _med(hh["strat_auc"]),
                         "median_beat_width": _med(hh["strength"] - hh["width_strength"]),
                         "n_windows": int(len(hh))}
            if bl is not None:
                b = np.array([bl.get((t, pop, wid), np.nan) for wid in hh["window"]], float)
                halves[h]["median_beat_baseline"] = _med(hh["strength"].to_numpy() - np.abs(b - 0.5))
        e0d = _dir(halves["e0"]["median_auc"])
        ed, ld = _dir(halves["early"]["median_auc"]), _dir(halves["late"]["median_auc"])
        e0_agrees = None if (e0d is None or not late_done or ld is None) else bool(e0d == ed == ld != 0)

        def both(key, test):
            if not late_done:
                return None
            e, l = halves["early"].get(key), halves["late"].get(key)
            return bool(e is not None and l is not None and test(e) and test(l))

        flags = {"e0_agrees": e0_agrees}
        if pop == "candidate":
            flags["beats_width_both_halves"] = both("median_beat_width", lambda v: v > 0)
            sd = _dir(halves["early"]["median_strat_auc"])
            flags["residual_both_halves"] = None if not late_done else bool(
                halves["early"]["median_strat_auc"] is not None and halves["late"]["median_strat_auc"] is not None
                and abs(halves["early"]["median_strat_auc"] - 0.5) >= pr["residual_min_strength"]
                and abs(halves["late"]["median_strat_auc"] - 0.5) >= pr["residual_min_strength"]
                and sd == _dir(halves["late"]["median_strat_auc"]))
        if bl is not None:
            flags["beats_baseline_both_halves"] = both("median_beat_baseline", lambda v: v > 0)
        out.append({
            "column": col, "target": t, "population": pop,
            "median_auc": agg["median"], "median_strength": _med(tests["strength"]),
            "median_direction": agg["median_direction"], "same_direction_share": agg["same_direction_share"],
            "halves": halves, "median_mi_norm": _med(tests["mi_norm"]), "median_lift": _med(tests["lift"]),
            "median_fire_rate": _med(tests["fire_rate"]), "median_strat_auc": _med(tests["strat_auc"]),
            "width_sort_auc": _med(tests["width_auc"]),
            "baseline_auc": None if bl is None else _med([bl.get((t, pop, wid), np.nan) for wid in tests["window"]]),
            "dedup": bool(g["dedup"].any()) if "dedup" in g else False, "candidate_flags": flags})
    return out


# ---------------------------------------------------------------------------- redundancy (R8)


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.full(len(x), np.nan)
    fin = np.isfinite(x)
    if fin.sum() > 1:
        r[fin] = rankdata(x[fin])
        r = (r - np.nanmean(r)) / (np.nanstd(r) or 1.0)
    return np.nan_to_num(r, nan=0.0).astype(np.float32)


def redundancy(panel: Panel, cols: list[str], pr: dict, early: list[W.Window]) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    rng = np.random.default_rng(pr["seed"])
    m_all = np.zeros(len(panel), bool)
    for w in early:
        m_all |= w.mask(panel.date)
    m_all &= panel.data_ok
    pools = {"candidate": np.flatnonzero(m_all & panel.candidate), "all": np.flatnonzero(m_all)}
    samp = {k: np.sort(rng.choice(v, min(pr["sample_rows"], len(v)), replace=False)) for k, v in pools.items()}
    R = {k: np.zeros((len(v), len(cols)), np.float32) for k, v in samp.items()}
    for k in range(0, len(cols), CHUNK):
        data = panel.columns(cols[k: k + CHUNK])
        for j, c in enumerate(cols[k: k + CHUNK]):
            for pop, s in samp.items():
                R[pop][:, k + j] = _ranks(data[c][s].astype(float))
    out = {"sample_rows": {k: int(len(v)) for k, v in samp.items()}, "cut": pr["redundancy_cut"]}
    for pop in ("candidate", "all"):
        n = len(samp[pop])
        rho = (R[pop].T @ R[pop]) / n
        np.fill_diagonal(rho, 1.0)
        rho = np.clip(np.nan_to_num(rho), -1, 1)
        gi, wi = cols.index(GUARD), cols.index(WIDTH)
        out[f"rho_guard_{pop}"] = {c: round(float(rho[i, gi]), 4) for i, c in enumerate(cols)}
        out[f"rho_width_{pop}"] = {c: round(float(rho[i, wi]), 4) for i, c in enumerate(cols)}
        if pop == "candidate":
            dist = np.clip(1 - np.abs(rho), 0, None)
            dist = (dist + dist.T) / 2
            np.fill_diagonal(dist, 0)
            lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                           t=1 - pr["redundancy_cut"], criterion="distance")
            out["groups"] = {c: int(g) for c, g in zip(cols, lab)}
    return out


# ---------------------------------------------------------------------------- driver


def run(kind: str, halves: list[str], panel: Panel | None = None, out_dir: Path = OUT,
        summary_path: Path | None = None, cols: list[str] | None = None, targets=None,
        windows_fn=W.all_windows, pr: dict | None = None, red_path: Path | None = None) -> int:
    pr = pr or prereg()
    wins = windows_fn(tuple(halves))
    lock = W.require_lock_for(wins)
    panel = panel or Panel()
    targets = targets or (LabelTargets(panel) if kind == "labels" else _regime_targets(panel))
    cols = cols or scored_columns(include_btc=(kind == "regime"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_path or HERE / f"single_{kind}.json"
    t0 = time.perf_counter()

    if kind == "labels":
        gate = pr["instrument_gate_halves"]
        first = targets.ids[:1] if set(halves) >= set(gate) else ()
        if first:                                          # instrument first: label:buy, all bars
            ins_t = _Only(targets, pr["instrument"]["target"])
            dfi = score_windows(panel, ins_t, [w for w in wins if w.half in gate], cols, pr)
            dfi = dfi[dfi["population"] == pr["instrument"]["population"]]
            verdict = instrument(dfi, pr, gate)
            print(f"instrument ({'+'.join(gate)}): {verdict['verdict']} "
                  f"{ {c: v['pass'] for c, v in verdict['columns'].items()} }", flush=True)
            if verdict["verdict"] == "FAIL":
                summary_path.write_text(json.dumps({"instrument": "FAIL", "detail": verdict}, indent=1))
                return 1

    if halves:                                         # [] = summarise the stored window scores only
        df = score_windows(panel, targets, wins, cols, pr)
        for h in halves:
            df[df["half"] == h].reset_index(drop=True).to_feather(out_dir / f"single_{kind}_{h}.feather")
    full = pd.concat([pd.read_feather(f) for f in sorted(out_dir.glob(f"single_{kind}_*.feather"))],
                     ignore_index=True)
    all_wins = windows_fn(tuple(sorted(full["half"].unique())))   # baselines for EVERY scored window
    baseline = targets.baseline_scores(panel, all_wins, pr, out_dir) if hasattr(targets, "baseline_scores") else None
    res = {"screen": f"037 single-column screen: {kind}",
           "label": "SCREEN (descriptive): association with a label or regime decides nothing; "
                    "a candidate needs a backtest in a later feature before any P&L claim (FR-025)",
           "prereg_lock": lock, "halves_scored": sorted(full["half"].unique().tolist()),
           "windows_scored": sorted(full["window"].unique().tolist()),
           "n_columns": int(full["column"].nunique()), "targets": list(targets.ids)}
    if kind == "labels":
        gate = pr["instrument_gate_halves"]
        res["instrument"] = instrument(full, pr, gate)
        if "late" in res["halves_scored"]:
            res["instrument_confirm"] = instrument(full, pr, ("early", "late"))
    red_path = red_path or HERE / "redundancy.json"
    if kind == "labels" and "early" in halves:          # R8: early candidate bars only
        own = [c for c in cols if not c.startswith("btc:")]
        red_path.write_text(json.dumps(redundancy(panel, own, pr, windows_fn(("early",))), indent=1))
    red = json.loads(red_path.read_text()) if red_path.exists() else None
    results = summarise(full, pr, baseline)
    if red:
        for r in results:
            c = r["column"][4:] if r["column"].startswith("btc:") else r["column"]
            r["redundancy_group"] = red["groups"].get(c)
            r["rho_guard"] = {p: red[f"rho_guard_{p}"].get(c) for p in POPS}
            r["rho_width"] = {p: red[f"rho_width_{p}"].get(c) for p in POPS}
    if kind == "regime":       # the full summary (~60 MB) stays in the cache; commit the slim one
        (out_dir / f"single_{kind}_full.json").write_text(json.dumps({**res, "results": results}, default=_json))
        results = slim(results, REGIME_KEEP_TOP)
        res["slim"] = (f"top {REGIME_KEEP_TOP} columns by median strength per target x population, plus every "
                       f"column with a candidate flag; full summary: {out_dir / f'single_{kind}_full.json'}")
    res["results"] = results
    summary_path.write_text(json.dumps(res, indent=1, default=_json))
    print(f"{kind}: {len(results)} results, {time.perf_counter() - t0:.0f}s -> {summary_path.name}")
    return 0


REGIME_KEEP_TOP = 30


def slim(results: list[dict], top: int) -> list[dict]:
    keep = set()
    by = {}
    for i, r in enumerate(results):
        by.setdefault((r["target"], r["population"]), []).append(i)
        if any(v is True for k, v in r["candidate_flags"].items() if k != "e0_agrees"):
            keep.add(i)
    for idx in by.values():
        idx = sorted(idx, key=lambda i: (-(results[i]["median_strength"] or 0), results[i]["column"]))
        keep.update(idx[:top])
    return [results[i] for i in sorted(keep)]


def _json(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


class _Only:
    """A target provider restricted to one target id."""

    def __init__(self, inner, target):
        self.inner, self.ids, self.kind = inner, (target,), inner.kind
        self.market_wide = tuple(t for t in getattr(inner, "market_wide", ()) if t == target)

    def y(self, t, w):
        return self.inner.y(t, w)

    def columns_for(self, t, cols):
        return self.inner.columns_for(t, cols)


def _regime_targets(panel):
    from research.indicator_screen.targets import RegimeTargets

    return RegimeTargets(panel)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", choices=("labels", "regime"), required=True)
    ap.add_argument("--halves", default="e0,early", help="comma list; 'none' = re-summarise stored scores")
    a = ap.parse_args(argv)
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    try:
        return run(a.targets, [] if a.halves == "none" else a.halves.split(","))
    except W.LateLockError as e:
        print(f"REFUSED: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
