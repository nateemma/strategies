"""037 US3/US4 per-fold multivariate models (research R10, FR-017, FR-018). One fold per process.

Per fold and target: a depth-3 rule tree and a gradient-boosted model, each fitted on that fold's
TRAINING rows only (purge = the target's look-ahead in bars) on (a) every harvested column and
(b) the restricted inputs [bb_width, guard_metric]; scored by test-window AUC beside the width
sort and, for label:buy, the ARO model's p_buy (stored signal tables). Settings are fixed in the
pre-registration.

Rows: label targets -> candidate bars. Regime targets -> all bars (primary; hourly subsample, the
purge applied in bars BEFORE subsampling) and candidate bars (secondary; not subsampled). A2 ->
training rows are A2 bars of earlier test windows ending >= N bars before test start; f00 is not
computable. Every model candidate is labelled "distant era not tested" (no data before E0 to train).

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.screen_models --target label:buy --fold f00
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.screen_models --merge --target label:buy
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns
from research.indicator_screen.screen_single import GUARD, HERE, WIDTH, prereg

MODELS_DIR = HERE / "models"
ARO_SIGNALS = HERE.parent / "nnnc_framework" / "signals"
LABELS = ("label:buy", "label:sell")
CHUNK = 64


def slug(target: str) -> str:
    return target.replace(":", "_")


def populations(target: str) -> tuple[str, ...]:
    return ("candidate",) if target in LABELS else ("all", "candidate")


def hourly(panel: Panel) -> np.ndarray:
    return np.asarray(panel.date.minute == 0)


def train_rows(panel, targets, target, pop, fold: W.Window, all_folds: list[W.Window]) -> np.ndarray:
    purge = W.purge(target) * W.BAR
    base = panel.data_ok.copy()
    if pop == "candidate":
        base &= panel.candidate
    if target.startswith("A2:"):
        m = np.zeros(len(panel), bool)
        for f in all_folds:
            if f.end <= fold.start - purge:       # an earlier test window, ending a purge before
                m |= f.mask(panel.date) & np.isfinite(targets.y(target, f))
        m &= base
    else:
        m = base & np.asarray((panel.date >= fold.train_start) & (panel.date < fold.start - purge))
        m &= np.isfinite(targets.y(target, fold))
    if pop == "all" and target not in LABELS:
        m &= hourly(panel)                        # after the purge (R10)
    return np.flatnonzero(m)


def test_rows(panel, targets, target, pop, fold) -> np.ndarray:
    m = fold.mask(panel.date) & panel.data_ok & np.isfinite(targets.y(target, fold))
    if pop == "candidate":
        m &= panel.candidate
    if pop == "all" and target not in LABELS:
        m &= hourly(panel)
    return np.flatnonzero(m)


def matrix(panel, cols, idx) -> np.ndarray:
    X = np.empty((len(idx), len(cols)), np.float32)
    for k in range(0, len(cols), CHUNK):
        d = panel.columns(cols[k: k + CHUNK])
        for j, c in enumerate(cols[k: k + CHUNK]):
            X[:, k + j] = d[c][idx]
    return X


def tree_rules(tree, names) -> list[str]:
    """Every root-to-leaf path as 'a <= x AND b > y -> p(class 1)'."""
    t = tree.tree_
    out = []

    def walk(node, conds):
        if t.feature[node] < 0:
            v = t.value[node][0]
            p = float(v[1] / v.sum()) if v.sum() else float("nan")
            out.append(" AND ".join(conds) + f" -> p1={p:.3f} n={int(t.n_node_samples[node])}")
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.6g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.6g}"])

    walk(0, [])
    return out


def aro_p_buy(panel, idx, fold_id: str) -> np.ndarray | None:
    p = ARO_SIGNALS / f"ARO_{fold_id}.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p, columns=["date", "pair", "p_buy"])
    t["date"] = pd.to_datetime(t["date"], utc=True)
    key = pd.DataFrame({"date": panel.date[idx], "pair": np.asarray(panel.pairs)[panel.pair[idx]]})
    m = key.merge(t, on=["date", "pair"], how="left")
    return m["p_buy"].to_numpy(float)


def fit_fold(panel, targets, target, fold, all_folds, cols, pr) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    restricted = pr["models"]["restricted_inputs"]
    recs = []
    for pop in populations(target):
        tr = train_rows(panel, targets, target, pop, fold, all_folds)
        te = test_rows(panel, targets, target, pop, fold)
        base = {"fold": fold.id, "half": fold.half, "target": target, "population": pop,
                "subsample": 4 if (pop == "all" and target not in LABELS) else 1,
                "n_train": int(len(tr)), "n_test": int(len(te))}
        # the fold's cut points applied to its training rows (A2: fixed early-window cuts, R11)
        ytr = targets.y(target, fold)[tr] > 0.5
        yte = targets.y(target, fold)[te]
        if len(tr) == 0 or len(te) == 0 or ytr.all() or not ytr.any():
            reason = "A2: no prior test windows" if target.startswith("A2:") and len(tr) == 0 \
                else "no training or test rows with both classes"
            recs.append({**base, "computable": False, "reason": reason})
            continue
        Xtr, Xte = matrix(panel, cols, tr), matrix(panel, cols, te)
        ri = [cols.index(c) for c in restricted]
        width_te = Xte[:, cols.index(WIDTH)]
        aro = aro_p_buy(panel, te, fold.id) if target == "label:buy" else None
        for model in ("tree", "hgb"):
            for inputs, Xa, Xb, names in (("all", Xtr, Xte, cols),
                                          ("restricted", Xtr[:, ri], Xte[:, ri], restricted)):
                m = (DecisionTreeClassifier(**pr["models"]["tree"]) if model == "tree"
                     else HistGradientBoostingClassifier(**pr["models"]["hgb"]))
                m.fit(Xa, ytr)
                p = m.predict_proba(Xb)[:, 1]
                r = {**base, "computable": True, "model": model, "inputs": inputs,
                     "auc": S.auc(p, yte), "width_sort_auc": S.auc(width_te, yte),
                     "aro_auc": S.auc(aro, yte) if aro is not None else None}
                if model == "tree":
                    r["rules"] = tree_rules(m, names)
                    used = [names[i] for i in m.tree_.feature if i >= 0]
                    r["top_splits"] = [c for c, _ in Counter(used).most_common()]
                recs.append(r)
    return recs


def merge(target: str, models_dir: Path = MODELS_DIR) -> dict:
    files = sorted((models_dir / slug(target)).glob("f*.json"))
    recs = [r for f in files for r in json.loads(f.read_text())["results"]]
    out = {"target": target, "folds": [f.stem for f in files],
           "label": "SCREEN (descriptive); persistence: distant era not tested (models cannot be "
                    "scored on E0) -- a later backtest must cover 2022", "populations": {}}
    for pop in populations(target):
        rp = [r for r in recs if r["population"] == pop]
        comp = [r for r in rp if r.get("computable")]
        block = {"computable_folds": sorted({r["fold"] for r in comp}),
                 "not_computable": sorted({(r["fold"], r.get("reason")) for r in rp if not r.get("computable")})}
        for model in ("tree", "hgb"):
            halves = {}
            for h in ("early", "late"):
                full = [r["auc"] for r in comp if r["model"] == model and r["inputs"] == "all" and r["half"] == h]
                rest = [r["auc"] for r in comp if r["model"] == model and r["inputs"] == "restricted" and r["half"] == h]
                wid = [r["width_sort_auc"] for r in comp if r["model"] == model and r["inputs"] == "all" and r["half"] == h]
                aro = [r["aro_auc"] for r in comp if r["model"] == model and r["inputs"] == "all"
                       and r["half"] == h and r.get("aro_auc") is not None]
                halves[h] = {"median_auc_all": _med(full), "median_auc_restricted": _med(rest),
                             "median_width_sort_auc": _med(wid), "median_aro_auc": _med(aro),
                             "n_folds": len(full)}
            cand = all(halves[h]["median_auc_all"] is not None and halves[h]["median_auc_restricted"] is not None
                       and halves[h]["median_auc_all"] > halves[h]["median_auc_restricted"] for h in ("early", "late"))
            splits = Counter(c for r in comp if r["model"] == "tree" and r["inputs"] == "all"
                             for c in r.get("top_splits", [])[:3])
            block[model] = {"halves": halves, "candidate": bool(cand) if halves["late"]["n_folds"] else None,
                            "persistence": "distant era not tested",
                            "recurring_top_splits": splits.most_common(10)}
        out["populations"][pop] = block
    return out


def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def provider(panel, target):
    from research.indicator_screen.screen_pairs import provider as P

    return P(panel, target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold")
    g.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.merge:
        res = merge(a.target)
        (HERE / f"models_{slug(a.target)}.json").write_text(json.dumps(res, indent=1))
        for pop, b in res["populations"].items():
            print(f"{a.target} [{pop}]: " + ", ".join(
                f"{m} candidate={b[m]['candidate']}" for m in ("tree", "hgb")))
        return 0
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    folds = W.folds()
    fold = next(f for f in folds if f.id == a.fold)
    try:
        lock = W.require_lock_for([fold])
    except W.LateLockError as e:
        print(f"REFUSED: {e}")
        return 1
    pr = prereg()
    t0 = time.perf_counter()
    panel = Panel()
    cols = scored_columns(include_btc=a.target not in LABELS)
    recs = fit_fold(panel, provider(panel, a.target), a.target, fold, folds, cols, pr)
    d = MODELS_DIR / slug(a.target)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.fold}.json").write_text(json.dumps(
        {"target": a.target, "fold": a.fold, "prereg_lock": lock, "results": recs}, indent=1))
    print(f"{a.target} {a.fold}: " + "; ".join(
        f"{r['population']}/{r['model']}/{r['inputs']} auc={r['auc']:.4f}" for r in recs if r.get("computable"))
        + f" ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
