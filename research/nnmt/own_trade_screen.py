"""040 follow-up: auxiliary-head relevance on NNMT's OWN operating-point trades (preregistration_own_trade_heads.json).

On mt_lr_ri_g24's early trades (f00-f07), each head's column at the trade's signal bar, in its ONE declared
direction: gap_bottom20 (trades above the fold's 20th score percentile minus those at or below it), its
fold-stratified permutation p, the partial Spearman given bb_width and its fold signs. No training, no backtest.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.own_trade_screen   -> screens_heads_own.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNMT")]

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"
EXPECTED_TRADES = 3731
OTHER = {"MT_LR_RI": "MT_LR_RI_AH", "MT_LR_RI_AH": "MT_LR_RI"}


def trades(pr: dict) -> tuple[pd.DataFrame, dict]:
    from evaluation.pool import pool
    from research.nnmt.build_arms import key_of

    t = pd.DataFrame(pool(json.loads((STRAT / "evaluation" / "folds" / "plan_mt_lr_ri_g24.json").read_text()))["trades"])
    t = t[t["fold_id"].isin(pr["early"])].rename(columns={"fold_id": "fold"}).reset_index(drop=True)
    t["signal_date"] = pd.to_datetime(t["open_date"], utc=True) - pd.Timedelta("15min")
    t["pp"] = 100.0 * t["profit_ratio"].astype(float)
    keys = {n: key_of(n) for n in ("NNMT_LR_RI_G24", "NNMT_LR_RI")}
    return t, {"trades": len(t), "keys_equal": len(set(keys.values())) == 1, "key": keys["NNMT_LR_RI"]}


def attach(t: pd.DataFrame, source: str, cols: list[str]) -> pd.DataFrame:
    parts = []
    for f, g in t.groupby("fold"):
        tb = pd.read_parquet(SIG / f"{source}_{f}.parquet", columns=["pair", "date", *cols])
        tb["date"] = pd.to_datetime(tb["date"], utc=True)
        m = g.merge(tb, left_on=["pair", "signal_date"], right_on=["pair", "date"], how="left")
        m.index = g.index
        parts.append(m[cols])
    return pd.concat(parts).loc[t.index]


def gap_bottom20(score: np.ndarray, pp: np.ndarray, fold: np.ndarray) -> float:
    num = 0.0
    for f in np.unique(fold):
        s, p = score[fold == f], pp[fold == f]
        lo = s <= np.quantile(s, 0.2)
        if lo.all() or not lo.any():
            continue
        num += len(s) * (p[~lo].mean() - p[lo].mean())
    return num / len(score)


def gap_share50(score, pp, fold) -> float:
    d = pd.DataFrame({"s": score, "pp": pp, "f": fold})
    hi = d.groupby("f")["s"].transform("median") < d["s"]
    return float(d.loc[hi, "pp"].mean() - d.loc[~hi, "pp"].mean())


def perm_p(score, pp, fold, obs: float, n: int, rng) -> float:
    idx = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    hits = 0
    s = score.copy()
    for _ in range(n):
        for i in idx:
            s[i] = score[rng.permutation(i)]
        hits += gap_bottom20(s, pp, fold) >= obs
    return (hits + 1) / (n + 1)


def screen_one(t: pd.DataFrame, raw: pd.Series, sign: str, rule: dict, seed: int = 0) -> dict:
    from research.new_inputs.screen import _sp, partial_spearman

    s = raw.to_numpy(float) * (-1.0 if sign == "-" else 1.0)
    ok = np.isfinite(s)
    d = t[ok].assign(score=s[ok])
    sc, pp, fd = d["score"].to_numpy(float), d["pp"].to_numpy(float), d["fold"].to_numpy()
    g20 = gap_bottom20(sc, pp, fd)
    per = {f: partial_spearman(g["score"], g["pp"], [g["bb_width"]]) for f, g in d.groupby("fold")}
    part = partial_spearman(d["score"], d["pp"], [d["bb_width"]])
    sg = np.sign(part or 0.0)
    sign_folds = int(sum(1 for v in per.values() if v is not None and sg != 0 and np.sign(v) == sg))
    p = perm_p(sc, pp, fd, g20, 10_000, np.random.default_rng(seed))
    passed = g20 > 0 and p <= 0.01 and (part or 0) > 0 and sign_folds >= 5
    return {"coverage": round(float(ok.mean()), 4), "gap_bottom20": round(g20, 4), "p_value": round(p, 5),
            "partial": part, "sign_folds": sign_folds, "per_fold": per,
            "gap_share50": round(gap_share50(sc, pp, fd), 4), "rho_p_buy": _sp(d["score"], d["p_buy"]), "pass": bool(passed)}


def main(argv=None) -> int:
    from research.nnmt import judge as J

    path, pr = J._prereg("own_trade_heads")
    bad = J._refused(path, {**pr, "arms": {}})
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    t, inst = trades(pr)
    t = t.join(attach(t, "MT_LR_RI", ["enter_long", "bb_width", "p_buy"]))
    inst["entry_share"] = round(float((t["enter_long"] == 1).mean()), 4)
    inst["pass"] = bool(inst["keys_equal"] and inst["entry_share"] >= 0.995 and inst["trades"] == EXPECTED_TRADES)
    out = {"label": pr["label"], "instrument": inst, "heads": {}, "descriptive": {}}
    if not inst["pass"]:
        out["heads"] = "INCONCLUSIVE (instrument failed; no score read)"
    else:
        rule = pr["relevance_rule"]
        for h, spec in pr["heads"].items():
            out["heads"][h] = {"column": spec["column"], "direction": spec["direction"], "source": spec["source"],
                               **screen_one(t, attach(t, spec["source"], [spec["column"]])[spec["column"]], spec["direction"], rule)}
            o = OTHER[spec["source"]]
            out["descriptive"][f"{h}@{o}"] = screen_one(t, attach(t, o, [spec["column"]])[spec["column"]], spec["direction"], rule)
        out["descriptive"]["p_buy@MT_LR_RI"] = screen_one(t, t["p_buy"], "+", rule)
    (HERE / "screens_heads_own.json").write_text(json.dumps(out, indent=1, default=str))
    short = lambda r: {k: r.get(k) for k in ("gap_bottom20", "p_value", "partial", "sign_folds", "gap_share50", "pass")}  # noqa: E731
    print(json.dumps({"instrument": inst,
                      "heads": {h: short(r) for h, r in out["heads"].items()} if isinstance(out["heads"], dict) else out["heads"],
                      "descriptive": {h: short(r) for h, r in out["descriptive"].items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
