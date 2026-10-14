"""Does the NNNC model rank trades WITHIN the gbb rule's candidates? (2026-10-13; diagnostic only,
no training, no backtest.)

The calibration data (data.md) shows AUC 0.93-0.98 on all four models, but ~80% of bars score ~0
(the gbb label is first a rule: guard_metric < 0.3 AND bb_width > 0.035), and forward return is flat
for p 0.1-0.75 in the late half. A high AUC can come entirely from learning the rule. The model is
only worth improving if it ranks the rule's CANDIDATES.

On rule-eligible bars (guard_metric < 0.3, bb_width > 0.035), per model and half:

  * conditional AUC of p_buy vs the model's own label;
  * Spearman with the 48-bar forward return of: p_buy, band depth (-guard_metric), band width;
  * the MATCHED-SELECTIVITY test: the bars the model selects (p >= its fold threshold) vs the same
    NUMBER of eligible bars chosen by depth (deepest first) and chosen at random (mean of 20 draws)
    -- mean forward return and +2.2 % hit rate within 48 bars.

A matched band-WIDTH selection was added AFTER the first run (depth turned out to be the weaker rule
feature); it is reported as a post-hoc baseline and does not change the declared verdict.

READING (declared before the run): the model adds within-candidate value if its selection beats
the depth selection on mean forward return in BOTH halves on at least 3 of 4 models. If depth
matches it, the model is mostly re-deriving the rule and model work should target the residual
(train on candidates); if random matches it, the threshold is doing no selection.

    PYTHONPATH=.:NNNC ../../.venv/bin/python -m research.model_calibration.within_candidates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "NNNC"))

from research.model_calibration.collect import EARLY, FOLDS, MODELS, SIG, auc, labels_for  # noqa: E402


HERE = Path(__file__).resolve().parent
GUARD, BBW = 0.3, 0.035
DRAWS = 20


def _sp(a, b) -> float | None:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 30:
        return None
    return round(float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")), 4)


def _sel(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "fwd_pct": round(float(d["fwd"].mean() * 100), 3) if len(d) else None,
            "hit48": round(float(d["hit48"].mean()), 4) if len(d) else None}


def main() -> int:
    meta = json.loads((SIG / "meta.json").read_text())
    rng = np.random.default_rng(0)
    out = {"label": "DIAGNOSTIC -- PROVISIONAL", "eligible": f"guard_metric < {GUARD} and "
           f"bb_width > {BBW}", "models": {}}
    verdict = {}
    for m, cls in MODELS.items():
        rows = {"early": [], "late": []}
        tabs = {f: pd.read_parquet(SIG / f"{m}_{f}.parquet") for f in FOLDS}
        pairs = sorted(set().union(*[set(t["pair"].unique()) for t in tabs.values()]))
        L = labels_for(cls, pairs)
        for f, t in tabs.items():
            thr = round(float(meta[f"{m}_{f}"]["threshold"]), 2)
            dt = pd.to_datetime(t["date"])
            t = t.assign(date=(dt.dt.tz_localize("UTC") if dt.dt.tz is None else dt)
                         .astype("datetime64[ns, UTC]"))
            parts = [g.merge(L[p], on="date", how="left") for p, g in t.groupby("pair")]
            d = pd.concat(parts, ignore_index=True).dropna(subset=["label", "fwd", "hit48"])
            d = d[(d["guard_metric"] < GUARD) & (d["bb_width"] > BBW)].copy()
            d["sel"] = d["p_buy"] >= thr
            rows["early" if f in EARLY else "late"].append(d)
        res = {}
        for h, parts in rows.items():
            d = pd.concat(parts, ignore_index=True)
            k = int(d["sel"].sum())
            model = d[d["sel"]]
            depth = d.nsmallest(k, "guard_metric")
            width = d.nlargest(k, "bb_width")  # POST-HOC baseline (added after the first run)
            rand = [d.sample(k, random_state=int(rng.integers(1 << 31))) for _ in range(DRAWS)]
            res[h] = {
                "eligible_bars": int(len(d)), "label_rate": round(float(d["label"].mean()), 4),
                "auc_within": auc(d["label"].to_numpy(), d["p_buy"].to_numpy()),
                "spearman_fwd": {"p_buy": _sp(d["p_buy"].to_numpy(), d["fwd"].to_numpy()),
                                 "depth": _sp(-d["guard_metric"].to_numpy(), d["fwd"].to_numpy()),
                                 "bb_width": _sp(d["bb_width"].to_numpy(), d["fwd"].to_numpy())},
                "selected": {"model": _sel(model), "depth": _sel(depth),
                             "bb_width_posthoc": _sel(width),
                             "random": {"n": k,
                                        "fwd_pct": round(float(np.mean([r["fwd"].mean() for r in rand]) * 100), 3),
                                        "hit48": round(float(np.mean([r["hit48"].mean() for r in rand])), 4)},
                             "all_eligible": _sel(d)},
                "overlap_model_depth": round(len(set(model.index) & set(depth.index)) / max(k, 1), 4),
            }
            print(m, h, json.dumps(res[h]), flush=True)
        out["models"][m] = res
        verdict[m] = all(res[h]["selected"]["model"]["fwd_pct"] > res[h]["selected"]["depth"]["fwd_pct"]
                         for h in res)
    n = sum(verdict.values())
    out["verdict"] = {"model_beats_depth_both_halves": verdict, "models": n,
                      "model_adds_within_candidate_value": n >= 3}
    out["posthoc_model_beats_bb_width_both_halves"] = {
        m: all(out["models"][m][h]["selected"]["model"]["fwd_pct"]
               > out["models"][m][h]["selected"]["bb_width_posthoc"]["fwd_pct"] for h in ("early", "late"))
        for m in out["models"]}
    print("VERDICT", json.dumps(out["verdict"]))
    (HERE / "within_candidates.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
