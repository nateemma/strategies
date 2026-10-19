"""038 post-hoc N-of-M votes (preregistration_vote.json): LONG when at least N of M members forecast a
log return above +0.003 (the 30 bp round trip) at that horizon. Members are read at their common
candles (the stride-16 grid; lgb_L is stride 1). Also writes the members' pairwise Spearman
correlation per horizon (late half), reported before any vote P&L.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.vote [--variant vote_B] [--lgb lgb_L_s2]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.forecast import common as C

KEY = ["pair", "date"]


def members_frame(members: list[str]) -> pd.DataFrame:
    """One row per (pair, date) present in every member, columns <member>__f_<n>."""
    out = None
    for m in members:
        d = pd.read_feather(C.PRED / f"{m}.feather", columns=KEY + [f"f_{n}" for n in C.HORIZONS])
        d = d.rename(columns={f"f_{n}": f"{m}__f_{n}" for n in C.HORIZONS})
        out = d if out is None else out.merge(d, on=KEY, how="inner")
    return out


def vote(df: pd.DataFrame, members: list[str], need: int, thr: float) -> pd.DataFrame:
    out = df[KEY].copy()
    for n in C.HORIZONS:
        f = df[[f"{m}__f_{n}" for m in members]].to_numpy(float)
        complete = np.isfinite(f).all(axis=1)
        with np.errstate(invalid="ignore"):
            out[f"f_{n}"] = np.where(complete, np.nanmean(f, axis=1), np.nan)
            out[f"sig_{n}"] = complete & ((f > thr).sum(axis=1) >= need)
            out[f"sigs_{n}"] = complete & ((f < -thr).sum(axis=1) >= need)
    return out.assign(failed=False, skipped_gap=lambda d: d["f_1"].isna())


def correlations(df: pd.DataFrame, members: list[str], late: pd.Series) -> dict:
    out = {}
    for n in C.HORIZONS:
        sub = df.loc[late, [f"{m}__f_{n}" for m in members]].dropna()
        c = sub.rank().corr().to_numpy()
        off = c[~np.eye(len(members), dtype=bool)]
        out[str(n)] = {"mean_offdiag": float(off.mean()), "min": float(off.min()), "max": float(off.max()),
                       "matrix": {m: dict(zip(members, map(float, row))) for m, row in zip(members, c)}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", choices=list(C.VOTES))
    ap.add_argument("--lgb", default="lgb_L", help="lgb_L_s2 rebuilds vote_B for condition (e)")
    a = ap.parse_args(argv)
    rc = C.refuse_unless_registered() or C.refuse_unless_registered(prereg=C.AMEND_VOTE)
    if rc:
        return rc
    pr = json.loads(C.AMEND_VOTE.read_text())
    thr = float(pr["threshold_log_return"])
    corr_p = C.HERE / "vote_correlations.json"
    corr = json.loads(corr_p.read_text()) if corr_p.exists() else {}
    for v in a.variant or list(C.VOTES):
        spec = pr["variants"][v]
        members = [a.lgb if m == "lgb_L" else m for m in spec["members"]]
        name = v + ("_s2" if a.lgb != "lgb_L" and "lgb_L" in spec["members"] else "")
        df = members_frame(members)
        folds = C.folds()
        late = pd.Series(False, index=df.index)
        for fid in C.LATE:
            s, e = folds[fid]
            late |= (df["date"] >= s) & (df["date"] < e)
        corr[name] = correlations(df, members, late)
        out = vote(df, members, int(spec["N"]), thr)
        C.write_feather(out, C.PRED / f"{name}.feather")
        sig = {n: int(out[f"sig_{n}"].sum()) for n in C.HORIZONS}
        print(f"{name}: members {members} N={spec['N']} rows {len(out)} signals {sig} "
              f"mean member corr (late, N=16) {corr[name]['16']['mean_offdiag']:.3f}")
    C.write_json(corr_p, corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
