"""037 addendum 3 (after the lock, user request 2026-10-17): the M-of-N vote screen. Rules in
preregistration_mofn.json, committed before any M-of-N score.

1. Pool: every raw_buy signal in a tbatr pair SURVIVOR (selected on early + E0 only; confirmation
   status is not used), either population.
2. Members (early windows, all bars, label:tbatr_buy only): cluster the pool by the phi correlation
   of their fires (average linkage, cut at phi >= PHI_CUT); one member per cluster, the one with
   the highest early median single lift; keep the N_MAX clusters whose member lifts most.
3. Score: count of members firing on the bar (0..N). Signal "M-of-N" = count >= M.
4. Choice of M on the early windows: M_best = the M with the highest early median lift (computable
   in >= min_windows early windows); M_match = the M whose early median fire rate is closest to the
   width book's (`cmp:width_book`).
5. Confirmation (late + E0), per M in {M_best, M_match} and population: "yes" if the late median lift
   exceeds the width book's AND the best single member's late medians, AND the E0 lift >= the width
   book's E0 lift and >= 1; "early+late only" if only E0 fails; else "no". On label:tbatr_buy
   (primary) and label:tb_buy (secondary: what the current fixed-% exits pay).
6. Benchmark: per fold, a logistic regression on the N member signals (training rows to 48 bars
   before the fold, hourly, data_ok) vs the count score; test-window AUC.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.mofn
"""
from __future__ import annotations

import json

import numpy as np

from research.indicator_screen import harvest as H
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel
from research.indicator_screen.screen_single import HERE, prereg

PRIMARY, SECONDARY = "label:tbatr_buy", "label:tb_buy"
POOL_FROM = HERE / "pairs_label_tbatr_buy.json"
PHI_CUT = 0.3
N_MAX = 8
OUT = HERE / "mofn.json"
PANEL_KW: dict = {}    # vol.py sets tradeable_only (and POOL_FROM / OUT to vol/)
POPS = ("all", "candidate")


def pool() -> list[str]:
    pp = json.loads(POOL_FROM.read_text())
    return sorted({x for b in pp["populations"].values() for s in b["survivors"] for x in (s["a"], s["b"])})


def _rows(panel, w, pop):
    m = w.mask(panel.date) & panel.data_ok
    return np.flatnonzero(m & panel.candidate if pop == "candidate" else m)


def _lift(sig, y, pr):
    return S.lift(sig, y, pr["min_fires"], pr["max_fire_rate"])


def _med(v):
    v = np.asarray([x for x in v if x is not None], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def members(panel, pool_cols, y, pr) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    early = W.all_windows(("early",))
    idx = np.concatenate([_rows(panel, w, "all") for w in early])
    d = panel.columns(pool_cols)
    F = np.column_stack([np.nan_to_num(d[c][idx]) > 0.5 for c in pool_cols]).astype(float)
    phi = np.nan_to_num(np.corrcoef(F.T))
    np.fill_diagonal(phi, 1.0)
    dist = np.clip(1 - phi, 0, None)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), t=1 - PHI_CUT, criterion="distance")
    lift = {c: _med([_lift(d[c][_rows(panel, w, "all")], y[_rows(panel, w, "all")], pr)[0] for w in early])
            for c in pool_cols}
    best = {}
    for c, g in zip(pool_cols, lab):
        if lift[c] is not None and (g not in best or lift[c] > lift[best[g]]):
            best[g] = c
    chosen = sorted(best.values(), key=lambda c: -lift[c])[:N_MAX]
    return {"pool": pool_cols, "clusters": {c: int(g) for c, g in zip(pool_cols, lab)},
            "early_lift": lift, "members": chosen}


def curves(panel, mem, targets: dict, comps: dict, pr) -> dict:
    """{target: {pop: {window: {"M": [lift per M], "fire": [...], "wb": lift, "wb_fire": rate, "single": {...}}}}}"""
    d = panel.columns(mem + ["cmp:width_book"])
    count = np.sum([np.nan_to_num(d[c]) > 0.5 for c in mem], axis=0)
    out = {}
    for t, y in targets.items():
        out[t] = {}
        for pop in POPS:
            per = {}
            for w in W.all_windows(("e0", "early", "late")):
                i = _rows(panel, w, pop)
                yy = y[i]
                ms = [_lift((count[i] >= m).astype(float), yy, pr) for m in range(1, len(mem) + 1)]
                wb = _lift(d["cmp:width_book"][i], yy, pr)
                per[w.id] = {"half": w.half, "lift": [m[0] for m in ms], "fire": [m[1] for m in ms],
                             "wb": wb[0], "wb_fire": wb[1],
                             "single": {c: _lift(d[c][i], yy, pr)[0] for c in mem},
                             "count_auc": S.auc(count[i].astype(float), yy)}
            out[t][pop] = per
    return out


def choose_m(per: dict, n: int, pr) -> dict:
    early = [v for v in per.values() if v["half"] == "early"]
    med = []
    for k in range(n):
        vals = [v["lift"][k] for v in early if np.isfinite(v["lift"][k])]
        med.append(float(np.median(vals)) if len(vals) >= pr["pairs"]["min_windows"] else None)
    fire = [_med([v["fire"][k] for v in early]) for k in range(n)]
    wbf = _med([v["wb_fire"] for v in early])
    ok = [k for k in range(n) if med[k] is not None]
    if not ok:
        return {"M_best": None, "M_match": None, "early_median_lift": med, "early_median_fire": fire}
    kb = max(ok, key=lambda k: med[k])
    km = min(ok, key=lambda k: abs((fire[k] or 0) - (wbf or 0)))
    return {"M_best": kb + 1, "M_match": km + 1, "early_median_lift": med, "early_median_fire": fire,
            "width_book_early_fire": wbf}


def confirm(per: dict, m: int, mem: list[str]) -> dict:
    k = m - 1
    h = lambda half, key: _med([v[key][k] if isinstance(v[key], list) else v[key]
                                for v in per.values() if v["half"] == half])
    late, wb_late = h("late", "lift"), _med([v["wb"] for v in per.values() if v["half"] == "late"])
    singles = {c: _med([v["single"][c] for v in per.values() if v["half"] == "late"]) for c in mem}
    best_single = max([s for s in singles.values() if s is not None], default=None)
    e0 = next((v for v in per.values() if v["half"] == "e0"), None)
    e0l, e0wb = (e0["lift"][k], e0["wb"]) if e0 else (None, None)
    late_ok = late is not None and late > (wb_late or 0) and late > (best_single or 0)
    e0_ok = e0l is not None and np.isfinite(e0l) and e0l >= 1 and (e0wb is None or not np.isfinite(e0wb) or e0l >= e0wb)
    return {"M": m, "early_lift": h("early", "lift"), "late_lift": late, "e0_lift": _f(e0l),
            "early_fire": h("early", "fire"), "late_fire": h("late", "fire"),
            "width_book": {"early": _med([v["wb"] for v in per.values() if v["half"] == "early"]),
                           "late": wb_late, "e0": _f(e0wb)},
            "best_single_late": best_single,
            "confirmed": "no" if not late_ok else ("yes" if e0_ok else "early+late only")}


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else None


def logistic_benchmark(panel, mem, y, pr) -> dict:
    from sklearn.linear_model import LogisticRegression

    d = panel.columns(mem)
    X = np.column_stack([np.nan_to_num(d[c]) > 0.5 for c in mem]).astype(float)
    count = X.sum(1)
    hourly = np.asarray(panel.date.minute == 0)
    out = {}
    for f in W.folds():
        tr = np.flatnonzero(panel.data_ok & hourly & np.isfinite(y)
                            & np.asarray((panel.date >= f.train_start) & (panel.date < f.start - W.purge(PRIMARY) * W.BAR)))
        te = _rows(panel, f, "all")
        te = te[np.isfinite(y[te])]
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr] > 0.5)
        out[f.id] = {"half": f.half, "logit_auc": S.auc(m.predict_proba(X[te])[:, 1], y[te]),
                     "count_auc": S.auc(count[te], y[te]), "coef": dict(zip(mem, np.round(m.coef_[0], 4).tolist()))}
    return out


def run() -> dict:
    pr = prereg()
    panel = Panel(**PANEL_KW)
    ys = {t: panel.columns([t])[t].astype(float) for t in (PRIMARY, SECONDARY)}
    W.require_lock_for(W.all_windows(("late",)))
    mem = members(panel, pool(), ys[PRIMARY], pr)
    n = len(mem["members"])
    cv = curves(panel, mem["members"], ys, {}, pr)
    res = {"label": "SCREEN (descriptive), post-lock addendum; preregistration_mofn.json", "members": mem,
           "choice": {}, "confirm": {}, "logistic": logistic_benchmark(panel, mem["members"], ys[PRIMARY], pr)}
    for pop in POPS:
        ch = choose_m(cv[PRIMARY][pop], n, pr)          # M chosen on the primary label, early only
        res["choice"][pop] = ch
        for t in (PRIMARY, SECONDARY):
            res["confirm"].setdefault(t, {})[pop] = [confirm(cv[t][pop], m, mem["members"])
                                                     for m in sorted({ch["M_best"], ch["M_match"]} - {None})]
    res["count_auc"] = {t: {pop: {h: _med([v["count_auc"] for v in per.values() if v["half"] == h])
                                  for h in ("e0", "early", "late")} for pop, per in by.items()}
                        for t, by in cv.items()}
    res["curves"] = cv
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return res


def main() -> int:
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    r = run()
    print("members:", r["members"]["members"])
    for pop, ch in r["choice"].items():
        print(pop, {k: ch[k] for k in ("M_best", "M_match")})
    for t, by in r["confirm"].items():
        for pop, rows in by.items():
            for c in rows:
                print(t, pop, json.dumps({k: c[k] for k in ("M", "early_lift", "late_lift", "e0_lift",
                                                            "early_fire", "width_book", "best_single_late", "confirmed")},
                                         default=float))
    lg = r["logistic"]
    for h in ("early", "late"):
        print(h, "logit AUC", _med([v["logit_auc"] for v in lg.values() if v["half"] == h]),
              "count AUC", _med([v["count_auc"] for v in lg.values() if v["half"] == h]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
