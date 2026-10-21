"""Reversal selector screen (preregistration.json): does the 1-day cross-sectional return rank split the ARO
width-book trades? Instrument = 034 S1's width book (research/label_screen/separation.py), both halves.

At each trade's signal bar (entry - 1): r1d = log close now - log close 24 h earlier, for every pair with both
timestamps; rank_pct 0 = the pair that fell most. agree = rank_pct <= 0.5. No backtest P&L, no fitting.

    PYTHONPATH=. ../../.venv/bin/python -m research.reversal_selector.screen
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(STRAT))

from research.xsec_screen.screen import committed  # noqa: E402

PREREG = HERE / "preregistration.json"
TWIN = STRAT / "research" / "model_calibration" / "width_twin.json"
EXPECTED_EARLY = 412  # 034 S1's width book


def rank_pct(own: float, others: list[float]) -> float:
    """Ascending rank of `own` among own + others (ties averaged), scaled to [0, 1]; 0 = lowest."""
    v = np.array([own, *others], float)
    r = pd.Series(v).rank(method="average").iloc[0]
    return float((r - 1) / (len(v) - 1))


def trade_ranks(d: pd.DataFrame, pairs: list[str], bars_for, lag: pd.Timedelta, min_pairs: int) -> np.ndarray:
    """rank_pct of each trade's pair by log return over `lag` ending at the signal bar (entry - 1); NaN if
    fewer than `min_pairs` pairs have both timestamps (the trade pair must be one of them)."""
    closes = {p: pd.Series(bars_for(p)[0].close, index=bars_for(p)[1]) for p in pairs}
    out = []
    for r in d.itertuples():
        t = bars_for(r.pair)[1][r.e - 1]
        ret = {}
        for p, c in closes.items():
            if t in c.index and (t - lag) in c.index:
                ret[p] = float(np.log(c[t]) - np.log(c[t - lag]))
        if r.pair not in ret or len(ret) < min_pairs:
            out.append(np.nan)
            continue
        own = ret.pop(r.pair)
        out.append(rank_pct(own, list(ret.values())))
    return np.array(out, float)


def perm_p(pp: np.ndarray, agree: np.ndarray, fold: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """One-sided p of the agree - disagree gap, agree labels shuffled within fold."""
    def gap(a):
        return pp[a].mean() - pp[~a].mean()
    obs, rng = gap(agree), np.random.default_rng(seed)
    idx = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    hits = 0
    for _ in range(n):
        a = agree.copy()
        for ix in idx:
            a[ix] = rng.permutation(a[ix])
        hits += gap(a) >= obs
    return float((hits + 1) / (n + 1))


def main() -> int:
    if not committed(PREREG):
        print("REFUSED: preregistration.json is not committed or has changes")
        return 2
    from scipy.stats import spearmanr

    from research.exit_coupling import feasibility as F
    from research.label_screen.separation import split_stats
    from research.model_calibration.width_sizing import _signal_rows
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    pr = json.loads(PREREG.read_text())
    sig, halves = pr["signal"], pr["instrument"]["folds"]
    base = E.Model("ARO")
    base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
    folds, cuts = sorted(base.folds), json.loads(TWIN.read_text())["models"]["ARO"]["cuts"]
    rows = []
    for i, f in enumerate(folds):
        t = base.tables[f]
        g = _gates(t, E.fold_params(base.prefix, f))
        cut = cuts[folds[i - 1]] if i else cuts[f]
        d = base.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
        if d.empty:
            continue
        cur = _signal_rows(t, d)
        d = d.assign(fold=f, agree_ctl=cur["agree"].to_numpy(), p_ctl=cur["p_buy"].to_numpy(),
                     rank1=trade_ranks(d, sig["pairs"], F.bars_for, pd.Timedelta("1D"), sig["min_pairs"]),
                     rank3=trade_ranks(d, sig["pairs"], F.bars_for, pd.Timedelta("3D"), sig["min_pairs"]))
        rows.append(d)
    a = pd.concat(rows, ignore_index=True)
    a["half"] = np.where(a["fold"].isin(halves["early"]), "early", "late")
    n_early = int((a["half"] == "early").sum())
    out = {"prereg": "728887e4", "trades": {"early": n_early, "late": int((a["half"] == "late").sum())},
           "unranked_share": round(float(a["rank1"].isna().mean()), 4), "halves": {}}
    reasons = []
    if n_early != EXPECTED_EARLY:
        reasons.append(f"early trades {n_early} != {EXPECTED_EARLY}")
    if out["unranked_share"] > 0.05:
        reasons.append(f"unranked share {out['unranked_share']}")
    r = a[a["rank1"].notna()].copy()
    r["agree"] = r["rank1"] <= 0.5
    sh = r.groupby("fold")["agree"].mean()
    deg = sorted(sh[(sh == 0) | (sh == 1)].index)
    if deg:
        reasons.append(f"degenerate folds {deg}")
    for h, x in r.groupby("half"):
        st = split_stats(x["pp"], x["agree"], -x["rank1"])
        st["agree_share"] = round(float(x["agree"].mean()), 4)
        st["within_ctl_agree"] = split_stats(x.loc[x["agree_ctl"], "pp"], x.loc[x["agree_ctl"], "agree"])
        st["within_ctl_disagree"] = split_stats(x.loc[~x["agree_ctl"], "pp"], x.loc[~x["agree_ctl"], "agree"])
        st["spearman_rank_vs_p_ctl"] = round(float(spearmanr(x["rank1"], x["p_ctl"]).statistic), 4)
        y = x[x["rank3"].notna()]
        st["r3d_variant"] = split_stats(y["pp"], y["rank3"] <= 0.5, -y["rank3"])
        out["halves"][h] = st
    p = perm_p(r["pp"].to_numpy(float), r["agree"].to_numpy(bool), r["fold"].to_numpy())
    out["pooled"] = split_stats(r["pp"], r["agree"], -r["rank1"]) | {"perm_p_one_sided": round(p, 4)}
    e, l = out["halves"].get("early", {}), out["halves"].get("late", {})
    passed = all(v is not None and v > 0 for v in (e.get("gap"), l.get("gap"))) \
        and all(v is not None and v > 0.5 for v in (e.get("auc"), l.get("auc"))) and p <= 0.05
    out["verdict"] = "INCONCLUSIVE" if reasons else ("PASS (PROVISIONAL)" if passed else "FAIL")
    out["inconclusive_reasons"] = reasons
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
