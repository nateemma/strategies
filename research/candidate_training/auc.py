"""032 S1: within-candidate AUC of the candidate-only model vs the current model (T019).

On IDENTICAL bars: the inner join, per fold, of the current model's and the retrained model's
test-window signal tables (`research/nnnc_framework/signals/<M>_fNN` and `CO_<M>_fNN`) on
(pair, date), restricted to bars passing the gbb rule (guard_metric < 0.3 AND bb_width > 0.035)
with a label; the label is the model's own gbb_path label (`collect.labels_for`, identical for both
since the label inputs are unchanged). S1 (pre-registered): AUC(retrained) - AUC(current) > 0 in
BOTH halves. A pair-day block bootstrap 95% CI of each difference is reported for information only.

    PYTHONPATH=.:NNNC ../../.venv/bin/python -m research.candidate_training.auc
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
BOOT, SEED = 300, 0


def _utc(t: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(t["date"])
    return t.assign(date=(d.dt.tz_localize("UTC") if d.dt.tz is None else d)
                    .astype("datetime64[ns, UTC]"))


def joined(cur: pd.DataFrame, new: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Eligible, labelled bars present in BOTH tables: columns pair, date, label, p_cur, p_new."""
    cur, new = _utc(cur), _utc(new)
    j = cur[["pair", "date", "p_buy", "guard_metric", "bb_width"]].rename(columns={"p_buy": "p_cur"})
    j = j.merge(new[["pair", "date", "p_buy"]].rename(columns={"p_buy": "p_new"}),
                on=["pair", "date"], how="inner")
    j = j[(j["guard_metric"] < GUARD) & (j["bb_width"] > BBW)]
    parts = [g.merge(labels[p][["date", "label"]], on="date", how="inner")
             for p, g in j.groupby("pair") if p in labels]
    out = pd.concat(parts, ignore_index=True) if parts else j.assign(label=[])
    return out.dropna(subset=["p_cur", "p_new", "label"])


def diff_ci(d: pd.DataFrame, rng) -> tuple[float, float]:
    blocks = d["pair"].astype(str) + "|" + d["date"].dt.floor("D").astype(str)
    codes, uniq = pd.factorize(blocks)
    idx_by = pd.Series(np.arange(len(d))).groupby(codes).apply(np.asarray).to_list()
    y, pc, pn = d["label"].to_numpy(int), d["p_cur"].to_numpy(float), d["p_new"].to_numpy(float)
    diffs = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by[k] for k in pick])
        a, b = auc(y[ix], pn[ix]), auc(y[ix], pc[ix])
        if a is not None and b is not None:
            diffs.append(a - b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return round(float(lo), 4), round(float(hi), 4)


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {"label": "032 S1 (pre-registered, research/candidate_training/preregistration.json)",
           "models": {}}
    for m in ("ARO", "GBR"):
        cur = {f: pd.read_parquet(SIG / f"{m}_{f}.parquet") for f in FOLDS}
        new = {f: pd.read_parquet(SIG / f"CO_{m}_{f}.parquet") for f in FOLDS}
        pairs = sorted(set().union(*[set(t["pair"].unique()) for t in cur.values()]))
        L = labels_for(MODELS[m], pairs)
        halves = {"early": [], "late": []}
        for f in FOLDS:
            halves["early" if f in EARLY else "late"].append(joined(cur[f], new[f], L))
        res = {}
        for h, parts in halves.items():
            d = pd.concat(parts, ignore_index=True)
            y = d["label"].to_numpy(int)
            a_new, a_cur = auc(y, d["p_new"].to_numpy(float)), auc(y, d["p_cur"].to_numpy(float))
            res[h] = {"bars": int(len(d)), "label_rate": round(float(y.mean()), 4),
                      "auc_retrained": round(a_new, 4), "auc_current": round(a_cur, 4),
                      "diff": round(a_new - a_cur, 4), "ci95_info_only": diff_ci(d, rng)}
        res["S1_pass"] = all(res[h]["diff"] > 0 for h in ("early", "late"))
        out["models"][m] = res
        print(m, json.dumps(res), flush=True)
    (HERE / "auc.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
