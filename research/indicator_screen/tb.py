"""037 addendum (after the lock, user request 2026-10-17): the screen against a gbb-NEUTRAL look-ahead
label, with gbb as a comparator. Rules are the locked ones (preregistration.json); this file's own
choices are in preregistration_tb.json, committed before any score.

`--label tb` (default) or `--label tbatr` (outputs suffixed _tbatr; see LABELS):
- `label:tb_buy`: 034's triple-barrier candidate on EVERY bar (Framework/TrainingSignals
  labels_triple_barrier, horizon 48, +2% before -5%, min_gain 0.02); no gbb condition. Last 48
  bars NaN. `label:tbatr_buy` (2026-10-17, preregistration_tbatr.json): symmetric +-4 x ATR barriers,
  direction of resolution only (fixed % barriers made volatility look predictive).
- Comparator signals, scored like raw signals (lift): `base:gbb_candidate` (the gbb trigger) and
  `cmp:width_book` (gbb AND bb_width >= the fold's width-book cut, NNNC_LS_CONTROL_SZ.WIDTH_CUTS;
  E0 rows use f00's cut, which was picked on data overlapping E0 -- comparator only).
- Single screen (all halves, all bars + candidate bars) -> single_tb.json; pair screen
  (select on early + E0, confirm on late) -> pairs_label_tb_buy.json.
- Residual beyond ARO (descriptive): per test window, the column's AUC within quintiles of ARO's
  own p_buy (research/nnnc_framework/signals/ARO_fNN, test windows only; quintile edges from the
  window's rows -- a conditioning, nothing is fitted), for label:tb_buy and label:buy -> resid_aro.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tb --labels
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tb --single
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tb --pairs
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tb --resid
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tb --report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns
from research.indicator_screen.screen_single import HERE, prereg

LABELS = {
    # 034 candidates.TB: fixed +2% before -5% (timeouts = 0) -- favours volatile bars
    "tb": {"target": "label:tb_buy", "params": {"horizon": W.LABEL_HORIZON, "pt": 0.02, "sl": 0.05, "min_gain": 0.02}},
    # volatility-neutral: symmetric +-4 x ATR(14)/close (no %-floor), 1 = up first, 0 = down first,
    # NaN = timeout or same-bar tie; k = 4.0 puts the median barrier at 2% on f00's training span
    "tbatr": {"target": "label:tbatr_buy", "params": {"horizon": W.LABEL_HORIZON, "pt": 1e-9, "sl": 1e-9,
                                                      "atr_scale": 4.0}},
}
NAME = "tb"
TARGET = LABELS[NAME]["target"]
TB = LABELS[NAME]["params"]


def use(name: str) -> None:
    """Point the module at one label: its target, params and output names."""
    global NAME, TARGET, TB
    NAME, TARGET, TB = name, LABELS[name]["target"], LABELS[name]["params"]


def _out(stem: str, ext: str = "json") -> Path:
    """tb keeps its committed names (single_tb.json, resid_aro.json, report_tb.md)."""
    if stem == "resid_aro":
        return HERE / (f"resid_aro.{ext}" if NAME == "tb" else f"resid_aro_{NAME}.{ext}")
    return HERE / f"{stem}_{NAME}.{ext}"
COMPARATORS = ("base:gbb_candidate", "cmp:width_book")
PREREG_TB = HERE / "preregistration_tb.json"
ARO_SIGNALS = HERE.parent / "nnnc_framework" / "signals"
RESID_BINS = 5


# ---------------------------------------------------------------------------- labels + comparators


def width_cuts() -> dict[str, float]:
    """{window id: cut}; E0 -> f00's cut."""
    import ast
    src = (H.ROOT / "user_data" / "strategies" / "NNNC" / "NNNC_LS_CONTROL_SZ.py").read_text()
    body = src[src.index("WIDTH_CUTS = {") + len("WIDTH_CUTS = "):]
    cuts = ast.literal_eval(body[: body.index("}") + 1])
    out = {f.id: cuts[f"{f.train_start:%Y%m%d}-{f.start:%Y%m%d}"] for f in W.folds()}
    return out | {"E0": out["f00"]}


def width_book(frame: pd.DataFrame, cuts: dict[str, float]) -> np.ndarray:
    d = pd.DatetimeIndex(frame["date"])
    gm, bw = frame["base:guard_metric"].to_numpy(float), frame["base:bb_width"].to_numpy(float)
    out = np.full(len(frame), np.nan)
    for w in [W.e0(), *W.folds()]:
        m = w.mask(d)
        out[m] = ((gm[m] < 0.3) & (bw[m] > 0.035) & (bw[m] >= cuts[w.id])).astype(float)
    return out


def tb_label(ohlcv: pd.DataFrame) -> np.ndarray:
    from Framework.TrainingSignals import labels_triple_barrier, labels_triple_barrier_sell

    d = ohlcv[["date", *H.OHLCV]].reset_index(drop=True)
    if NAME == "tb":
        y = labels_triple_barrier(d, **TB).to_numpy(float).copy()
    else:
        kw = {**TB, "min_gain": None}
        up = labels_triple_barrier(d, **kw).to_numpy(float)
        dn = labels_triple_barrier_sell(d, **{**TB, "min_loss": None}).to_numpy(float)
        y = np.where(up > 0.5, 1.0, np.where(dn > 0.5, 0.0, np.nan))
    y[-W.LABEL_HORIZON:] = np.nan
    return y


def add_labels(pairs: list[str]) -> None:
    H._quiet()
    cfg, cuts = H.base_config(), width_cuts()
    for pair in pairs:
        frame = H.read_cache(pair)
        y = tb_label(H.load_ohlcv(cfg, pair))
        if len(y) != len(frame):
            raise SystemExit(f"{pair}: label rows {len(y)} != cache rows {len(frame)}")
        frame[TARGET] = y.astype(np.float32)
        frame["cmp:width_book"] = width_book(frame, cuts).astype(np.float32)
        frame.to_feather(H.cache_path(pair))
        sc = json.loads(H.sidecar_path(pair).read_text())
        sc["columns"] = [c for c in frame.columns if c not in ("date", *H.OHLCV)]
        sc[NAME] = {"params": TB, "buy_share": float(np.nanmean(y)), "nan_share": float(np.mean(np.isnan(y))),
                    "added": "2026-10-17 (post-lock)"}
        H.sidecar_path(pair).write_text(json.dumps(sc, indent=1))
        print(f"{pair}: {TARGET} {np.nanmean(y):.4f}, NaN {np.mean(np.isnan(y)):.3f}", flush=True)


class TBTargets:
    market_wide = ()

    def __init__(self, panel: Panel):
        self.p = panel
        self.kind, self.ids = NAME, (TARGET,)
        self._y = panel.columns([TARGET])[TARGET].astype(np.float64)

    def y(self, target: str, window: W.Window) -> np.ndarray:
        return self._y

    def baseline(self, target: str, window=None):
        return None

    def columns_for(self, target: str, cols: list[str]) -> list[str]:
        return [c for c in cols if not c.startswith("btc:")]


def columns() -> list[str]:
    return scored_columns(include_btc=False) + list(COMPARATORS)


# ---------------------------------------------------------------------------- screens


def single(halves=("e0", "early", "late")) -> int:
    from research.indicator_screen import screen_single as SS

    panel = Panel()
    return SS.run(NAME, list(halves), panel=panel, targets=TBTargets(panel), cols=columns(),
                  summary_path=_out("single"), red_path=HERE / "redundancy.json")


def comparator_lifts(panel: Panel, targets, pr) -> dict:
    """Per population and window: the lift of each comparator signal on label:tb_buy."""
    from research.indicator_screen.screen_pairs import populations, score

    out = {}
    for pop in populations(TARGET):
        sc = score(panel, targets, TARGET, pop, W.all_windows(("e0", "early", "late")), list(COMPARATORS), pr)
        out[pop] = {c: {wid: (None if not np.isfinite(v[0][i]) else float(v[0][i])) for wid, v in sc.items()}
                    for i, c in enumerate(COMPARATORS)}
    return out


def pairs() -> int:
    from research.indicator_screen import screen_pairs as SP

    pr = prereg()
    panel = Panel()
    t = TBTargets(panel)
    sel = SP.select(panel, t, TARGET, W.all_windows, columns(), pr)
    res = SP.confirm(panel, t, sel, W.all_windows, pr)
    res["comparator_lifts"] = comparator_lifts(panel, t, pr)
    SP.out_path(TARGET).write_text(json.dumps(res, indent=1))
    for pop, b in res["populations"].items():
        print(f"{TARGET} [{pop}]: {b['n_passing_rule']} pass the rule, {len(b['survivors'])} survivors, "
              f"confirmed {sum(s.get('confirmed') == 'yes' for s in b['survivors'])}")
    return 0


# ---------------------------------------------------------------------------- residual beyond ARO


def aro_p(panel: Panel) -> np.ndarray:
    p = np.full(len(panel), np.nan)
    key = pd.DataFrame({"date": panel.date.tz_convert("UTC"), "pair": np.asarray(panel.pairs)[panel.pair],
                        "i": np.arange(len(panel))})
    for f in W.folds():
        t = pd.read_parquet(ARO_SIGNALS / f"ARO_{f.id}.parquet", columns=["date", "pair", "p_buy"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        m = key.merge(t, on=["date", "pair"], how="inner")
        p[m["i"].to_numpy()] = m["p_buy"].to_numpy(float)
    return p


def resid(cols: list[str] | None = None) -> dict:
    pr = prereg()
    panel = Panel()
    cols = cols or scored_columns(include_btc=False)
    pa = aro_p(panel)
    ys = {TARGET: panel.columns([TARGET])[TARGET].astype(float), "label:buy": panel.core["label:buy"]}
    pops = {"all": np.ones(len(panel), bool), "candidate": panel.candidate}
    ctx = {}
    for w in W.folds():
        for pop, pm in pops.items():
            idx = np.flatnonzero(w.mask(panel.date) & panel.data_ok & pm & np.isfinite(pa))
            for t in ((TARGET,) if pop == "all" else (TARGET, "label:buy")):
                ctx[(w.id, pop, t)] = idx
    recs = []
    for k in range(0, len(cols), 64):
        data = panel.columns(cols[k: k + 64])
        for c in cols[k: k + 64]:
            x = data[c].astype(float)
            for (wid, pop, t), idx in ctx.items():
                if len(idx) < pr["min_rows"]:
                    continue
                a = pa[idx]
                e = S.quantile_edges(a, RESID_BINS)
                yy = ys[t][idx]
                recs.append({"column": c, "target": t, "population": pop, "window": wid,
                             "half": W.by_id(wid).half,
                             "auc": S.auc(x[idx], yy),
                             "resid_auc": S.stratified_auc(x[idx], yy, a, np.zeros(len(idx)), e, [])})
        print(f"  resid {min(k + 64, len(cols))}/{len(cols)}", flush=True)
    df = pd.DataFrame(recs)
    out = {"label": "037 addendum: AUC within ARO p_buy quintiles (test windows f00-f14; descriptive)",
           "aro_auc": {}, "results": []}
    for (t, pop), g in df.groupby(["target", "population"]):
        for (c,), gc in g.groupby(["column"]):
            h = {hh: gc[gc["half"] == hh] for hh in ("early", "late")}
            out["results"].append({
                "column": c, "target": t, "population": pop,
                **{f"{hh}_auc": _med(v["auc"]) for hh, v in h.items()},
                **{f"{hh}_resid_auc": _med(v["resid_auc"]) for hh, v in h.items()},
                "resid_both_halves": bool(all(_med(v["resid_auc"]) is not None
                                              and abs(_med(v["resid_auc"]) - 0.5) >= pr["residual_min_strength"]
                                              for v in h.values())
                                          and np.sign(_med(h["early"]["resid_auc"]) - 0.5)
                                          == np.sign(_med(h["late"]["resid_auc"]) - 0.5))})
    for (wid, pop, t), idx in ctx.items():
        out["aro_auc"].setdefault(f"{t} [{pop}]", {})[wid] = S.auc(pa[idx], ys[t][idx])
    _out("resid_aro").write_text(json.dumps(out, indent=1))
    return out


def _med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


# ---------------------------------------------------------------------------- report


def report() -> str:
    from research.indicator_screen import tags as TG

    tags = TG.load()
    from research.indicator_screen.screen_pairs import out_path

    st = json.loads(_out("single").read_text())["results"]
    pp = json.loads(out_path(TARGET).read_text())
    rr = json.loads(_out("resid_aro").read_text()) if _out("resid_aro").exists() else None
    f = lambda v, n=3: "–" if v is None else f"{v:.{n}f}"
    L = ["# 037 addendum: indicators vs a gbb-neutral triple-barrier label", "",
         "**SCREEN (descriptive), added after the pre-registration lock (user request, 2026-10-17); "
         "rules are the locked ones, this addendum's choices are in `preregistration_tb.json`.** "
         "Nothing here is a P&L claim.", "",
         f"Label `{TARGET}`: triple barrier on every bar, {TB}"
         + (" (symmetric ATR barriers; timeouts and ties are NaN, so the label is the direction of "
            "resolution)" if NAME != "tb" else "") + ". Comparators: the gbb trigger "
         "(`base:gbb_candidate`) and the width book's entry rule (`cmp:width_book`).", ""]
    cl = pp["comparator_lifts"]
    L += ["## Comparator lifts (median per half)", "", "| population | comparator | E0 | early | late |", "|---|---|---|---|---|"]
    for pop, d in cl.items():
        for c, per in d.items():
            m = {h: _med([v for wid, v in per.items() if W.by_id(wid).half == h and v is not None])
                 for h in ("e0", "early", "late")}
            L.append(f"| {pop} | `{c}` | {f(m['e0'])} | {f(m['early'])} | {f(m['late'])} |")
    L += ["", "## Single columns (all bars): top 20 by median strength, indicator-tagged only", "",
          "| column | median AUC | early | late | E0 | lift (signals) | E0 agrees |", "|---|---|---|---|---|---|---|"]
    rows = sorted((r for r in st if r["population"] == "all" and r["median_strength"] is not None
                   and TG.tag_of(r["column"], tags) == "indicator"), key=lambda r: -r["median_strength"])
    for r in rows[:20]:
        h = r["halves"]
        L.append(f"| `{r['column']}` | {f(r['median_auc'])} | {f(h['early']['median_auc'])} | "
                 f"{f(h['late']['median_auc'])} | {f(h['e0']['median_auc'])} | {f(r['median_lift'], 2)} | "
                 f"{r['candidate_flags'].get('e0_agrees')} |")
    sig = sorted((r for r in st if r["population"] == "all" and r["median_lift"] is not None),
                 key=lambda r: -r["median_lift"])
    L += ["", "## Signals (all bars): top 15 by median lift, beside the comparators", "",
          "| signal | median lift (15 test windows) | fire rate | E0 agrees |", "|---|---|---|---|"]
    for r in sig[:15] + [r for r in sig if r["column"] in COMPARATORS and r not in sig[:15]]:
        L.append(f"| `{r['column']}` | {f(r['median_lift'], 2)} | {f(r['median_fire_rate'], 3)} | "
                 f"{r['candidate_flags'].get('e0_agrees')} |")
    L += ["", "## Signal pairs (both must fire)", ""]
    for pop, b in pp["populations"].items():
        yes = [s for s in b["survivors"] if s.get("confirmed") == "yes"]
        L.append(f"- {pop} bars: {b['n_passing_rule']} pass the rule, {len(b['survivors'])} survivors, "
                 f"{len(yes)} confirmed (E0 agrees).")
        for s in yes[:10]:
            L.append(f"  - `{s['a']}` AND `{s['b']}`: early {f(s['early']['median_lift'], 2)}, "
                     f"late {f(s['late']['median_lift'], 2)}, E0 {f(s['e0_lift'], 2)}")
    if rr:
        L += ["", "## Beyond ARO: AUC within ARO p_buy quintiles (test windows; descriptive)", ""]
        for key, per in sorted(rr["aro_auc"].items()):
            L.append(f"- ARO p_buy AUC on {key}: early {f(_med([v for w, v in per.items() if W.by_id(w).half == 'early']))}, "
                     f"late {f(_med([v for w, v in per.items() if W.by_id(w).half == 'late']))}")
        for (t, pop) in ((TARGET, "all"), (TARGET, "candidate"), ("label:buy", "candidate")):
            g = sorted((r for r in rr["results"] if r["target"] == t and r["population"] == pop and r["resid_both_halves"]
                        and TG.tag_of(r["column"], tags) == "indicator"),
                       key=lambda r: -min(abs(r["early_resid_auc"] - 0.5), abs(r["late_resid_auc"] - 0.5)))
            L += ["", f"**{t} [{pop}]:** {len(g)} indicator columns keep strength >= "
                  f"{prereg()['residual_min_strength']} within ARO quintiles in both halves.", ""]
            if g:
                L += ["| column | resid AUC early | late | plain AUC early | late |", "|---|---|---|---|---|"]
                for r in g[:12]:
                    L.append(f"| `{r['column']}` | {f(r['early_resid_auc'])} | {f(r['late_resid_auc'])} | "
                             f"{f(r['early_auc'])} | {f(r['late_auc'])} |")
    text = "\n".join(L) + "\n"
    _out("report", "md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for k in ("labels", "single", "pairs", "resid", "report"):
        g.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--label", choices=sorted(LABELS), default="tb")
    a = ap.parse_args(argv)
    use(a.label)
    if not a.report:
        busy = H._busy()
        if busy:
            print(f"REFUSED: a training/backtest chain is running: {busy}")
            return 1
    if a.labels:
        add_labels(H.pairs())
    elif a.single:
        return single()
    elif a.pairs:
        return pairs()
    elif a.resid:
        resid()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
