"""037 addendum 4 (after the lock, user request 2026-10-17): a continuous composite of the
tbatr-direction columns and an 'extreme within k bars, then turns' event, on tradeable bars. Rules
in preregistration_composite.json, committed before any composite score.

    PYTHONPATH=.:../../evaluation ../../.venv/bin/python -m research.indicator_screen.composite
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import mofn as MN
from research.indicator_screen import scores as S
from research.indicator_screen import tags as TG
from research.indicator_screen import tb as TB
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel
from research.indicator_screen.screen_single import HERE, prereg

PRIMARY, SECONDARY = MN.PRIMARY, MN.SECONDARY
SINGLE = HERE / "vol" / "single_tbatr.json"
OUT = HERE / "vol" / "composite.json"
N_MAX, RHO_MAX = 8, 0.7
MIN_EARLY, MIN_E0 = 0.02, 0.01
KS = (4, 12, 24)
Q_EXTREME = 0.90
POPS = MN.POPS
VOTE_M = 4                                         # the 4-of-8 vote (vol/mofn.json, M_best on all bars)


def vote_cols() -> list[str]:
    return json.loads((HERE / "vol" / "mofn.json").read_text())["members"]["members"]
_med = MN._med


def pool() -> list[dict]:
    tags = TG.load()
    out = []
    for r in json.loads(SINGLE.read_text())["results"]:
        c, h = r["column"], r["halves"]
        if r["target"] != PRIMARY or r["population"] != "all" or c.startswith(("raw_", "cmp:")) \
                or c == "base:gbb_candidate" or TG.tag_of(c, tags) != "indicator":
            continue
        e, z = h.get("early") or {}, h.get("e0") or {}
        if e.get("median_strength") is None or z.get("median_auc") is None:
            continue
        d = np.sign(e["median_auc"] - 0.5)
        if e["median_strength"] >= MIN_EARLY and np.sign(z["median_auc"] - 0.5) == d \
                and z["median_strength"] >= MIN_E0:
            out.append({"column": c, "dir": int(d), "early": e["median_strength"],
                        "group": r["redundancy_group"]})
    return out


def members(panel, cands: list[dict]) -> list[dict]:
    best = {}
    for c in cands:
        if c["group"] not in best or c["early"] > best[c["group"]]["early"]:
            best[c["group"]] = c
    order = sorted(best.values(), key=lambda c: -c["early"])
    idx = np.concatenate([MN._rows(panel, w, "all") for w in W.all_windows(("early",))])[::16]
    d = panel.columns([c["column"] for c in order])
    rk = {c["column"]: pd.Series(d[c["column"]][idx]).rank().to_numpy() for c in order}
    kept = []
    for c in order:
        x = rk[c["column"]]
        if all(abs(pd.Series(x).corr(pd.Series(rk[k["column"]]))) < RHO_MAX for k in kept):
            kept.append(c)
        if len(kept) == N_MAX:
            break
    return kept


def _train_rows(panel, f) -> np.ndarray:
    ok = panel.data_ok & np.asarray((panel.date >= f.train_start)
                                    & (panel.date < f.start - W.purge(PRIMARY) * W.BAR))
    start = 0
    for n in panel.lengths:                                 # per-pair warm-up
        ok[start: start + W.WARMUP_BARS] = False
        start += n
    return ok


def _shift(x: np.ndarray, lengths, k: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan)
    s = 0
    for n in lengths:
        out[s + k: s + n] = x[s: s + n - k]
        s += n
    return out


def _recent_max(x: np.ndarray, lengths, k: int) -> np.ndarray:
    """max over bars t-k..t-1 within the pair."""
    out = np.full(len(x), np.nan)
    s = 0
    for n in lengths:
        out[s: s + n] = pd.Series(x[s: s + n]).shift(1).rolling(k, min_periods=1).max().to_numpy()
        s += n
    return out


def run() -> dict:
    pr = prereg()
    panel = Panel(tradeable_only=True)
    W.require_lock_for(W.all_windows(("late",)))
    mem = members(panel, pool())
    cols = [m["column"] for m in mem]
    sign = np.array([m["dir"] for m in mem], float)
    d = panel.columns(cols + ["cmp:width_book"] + vote_cols())
    X = np.column_stack([d[c] for c in cols]).astype(float)
    ys = {t: panel.columns([t])[t].astype(float) for t in (PRIMARY, SECONDARY)}
    vote = np.sum([np.nan_to_num(d[c]) > 0.5 for c in vote_cols()], axis=0) >= VOTE_M
    pa = TB.aro_p(panel)
    folds = {f.id: f for f in W.folds()}
    hourly = np.asarray(panel.date.minute == 0)

    comp = np.full(len(panel), np.nan)
    events = {k: np.zeros(len(panel), bool) for k in KS}
    Z_by_fold, logit = {}, {}
    from sklearn.linear_model import LogisticRegression
    for w in W.all_windows(("e0", "early", "late")):
        f = folds["f00"] if w.half == "e0" else folds[w.id]
        tr = _train_rows(panel, f)
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
        Z = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1) * sign)
        c = Z.mean(1)
        q = np.quantile(c[tr], Q_EXTREME)
        m = w.mask(panel.date)
        comp[m] = c[m]
        ext = (c >= q).astype(float)
        falling = c < _shift(c, panel.lengths)
        for k in KS:
            events[k][m] = ((_recent_max(ext, panel.lengths, k) > 0) & falling)[m]
        if w.half != "e0":
            trh = tr & hourly & np.isfinite(ys[PRIMARY])
            lr = LogisticRegression(max_iter=1000).fit(Z[trh], ys[PRIMARY][trh] > 0.5)
            te = MN._rows(panel, w, "all")
            te = te[np.isfinite(ys[PRIMARY][te])]
            logit[w.id] = {"half": w.half, "logit_auc": S.auc(lr.predict_proba(Z[te])[:, 1], ys[PRIMARY][te]),
                           "composite_auc": S.auc(c[te], ys[PRIMARY][te]),
                           "coef": dict(zip(cols, np.round(lr.coef_[0], 4).tolist()))}

    per = {}
    for t, y in ys.items():
        per[t] = {}
        for pop in POPS:
            rows = {}
            for w in W.all_windows(("e0", "early", "late")):
                i = MN._rows(panel, w, pop)
                yy = y[i]
                rows[w.id] = {
                    "half": w.half, "auc": S.auc(comp[i], yy),
                    "single": {cc: S.auc(X[i, j] * sign[j], yy) for j, cc in enumerate(cols)},
                    "aro_auc": S.auc(pa[i], yy) if w.half != "e0" else None,
                    "lift": {k: MN._lift(events[k][i].astype(float), yy, pr) for k in KS},
                    "wb": MN._lift(d["cmp:width_book"][i], yy, pr),
                    "vote": MN._lift(vote[i].astype(float), yy, pr)}
            per[t][pop] = rows
    res = {"label": "SCREEN (descriptive), post-lock addendum; preregistration_composite.json",
           "members": mem, "logistic": logit, "composite": {}, "event": {}}
    kb = _choose_k(per[PRIMARY]["all"], pr)
    res["k_best"] = kb
    for t in per:
        for pop, rows in per[t].items():
            res["composite"].setdefault(t, {})[pop] = _confirm_comp(rows, cols)
            res["event"].setdefault(t, {})[pop] = _confirm_event(rows, kb)
    res["per_window"] = per
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return res


def _h(rows, half, f):
    return _med([f(v) for v in rows.values() if v["half"] == half])


def _choose_k(rows, pr) -> int | None:
    med = {}
    for k in KS:
        vals = [v["lift"][k][0] for v in rows.values() if v["half"] == "early" and np.isfinite(v["lift"][k][0])]
        med[k] = float(np.median(vals)) if len(vals) >= pr["pairs"]["min_windows"] else None
    ok = [k for k in KS if med[k] is not None]
    return max(ok, key=lambda k: med[k]) if ok else None


def _confirm_comp(rows, cols) -> dict:
    a = {h: _h(rows, h, lambda v: v["auc"]) for h in ("e0", "early", "late")}
    singles = {c: _h(rows, "late", lambda v, c=c: abs(v["single"][c] - 0.5)) for c in cols}
    best = max([s for s in singles.values() if s is not None], default=None)
    dir_e = np.sign((a["early"] or 0.5) - 0.5)
    late_ok = a["late"] is not None and np.sign(a["late"] - 0.5) == dir_e and abs(a["late"] - 0.5) > (best or 0)
    e0_ok = a["e0"] is not None and np.sign(a["e0"] - 0.5) == dir_e and abs(a["e0"] - 0.5) >= MIN_E0
    return {"auc": a, "best_single_late_strength": best,
            "aro_auc": {h: _h(rows, h, lambda v: v["aro_auc"]) for h in ("early", "late")},
            "confirmed": "yes" if late_ok and e0_ok else "no"}


def _confirm_event(rows, k) -> dict:
    if k is None:
        return {"k": None, "confirmed": "no"}
    g = lambda half, key, j=0: _h(rows, half, lambda v: (v[key][k] if key == "lift" else v[key])[j])
    late, wb_late, vote_late = g("late", "lift"), g("late", "wb"), g("late", "vote")
    e0 = next(v for v in rows.values() if v["half"] == "e0")
    e0l, e0wb = e0["lift"][k][0], e0["wb"][0]
    late_ok = late is not None and late > (wb_late or 0) and late > (vote_late or 0)
    e0_ok = np.isfinite(e0l) and e0l >= 1 and (not np.isfinite(e0wb) or e0l >= e0wb)
    return {"k": k, "lift": {"early": g("early", "lift"), "late": late, "e0": MN._f(e0l)},
            "fire": {"early": g("early", "lift", 1), "late": g("late", "lift", 1)},
            "width_book": {"early": g("early", "wb"), "late": wb_late, "e0": MN._f(e0wb)},
            "vote_4of8": {"early": g("early", "vote"), "late": vote_late, "e0": MN._f(e0["vote"][0])},
            "confirmed": "no" if not late_ok else ("yes" if e0_ok else "early+late only")}


def main() -> int:
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    r = run()
    print("members:", [(m["column"], m["dir"]) for m in r["members"]], "k_best:", r["k_best"])
    for part in ("composite", "event"):
        for t, by in r[part].items():
            for pop, c in by.items():
                print(part, t, pop, json.dumps(c, default=float))
    lg = r["logistic"]
    for h in ("early", "late"):
        print(h, "logit AUC", _med([v["logit_auc"] for v in lg.values() if v["half"] == h]),
              "composite AUC", _med([v["composite_auc"] for v in lg.values() if v["half"] == h]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
