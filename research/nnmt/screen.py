"""040 trade-relevance screen (research R9, R10 screen 2; preregistration_screens.json): does a score
rank the width book's early trades? No training, no backtest, early half only (the late half stays blind
for any arm that follows).

On the 412 early width-book trades (research.new_inputs.trades.trades(): it reproduces the trade count and
cs_aro's 0.1004 control gap -- the screen refuses otherwise), a score from an NNMT signal table at each
trade's signal bar, in ONE pre-declared direction ("+" or "-"):
  gap_share50, partial (given bb_width), sign_folds, rho_pctl (orthogonality with the control's p_buy)
-- the 039 metrics (research.nnpredict.screen), reused unchanged.

    sizer  score p_buy of the selected model <M>, direction "+"           -> screens_sizer.json
    heads  each head's declared column of <M>_ah, in its declared direction -> screens_heads.json["relevance"]

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.screen sizer|heads
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"


def attach(t: pd.DataFrame, tables: dict, col: str) -> pd.DataFrame:
    parts = []
    for f, g in t.groupby("fold"):
        tb = tables[f][["pair", "date", col]].copy()
        tb["date"] = pd.to_datetime(tb["date"], utc=True)
        g = g.assign(signal_date=pd.to_datetime(g["signal_date"], utc=True))
        m = g.merge(tb, left_on=["pair", "signal_date"], right_on=["pair", "date"], how="left")
        m.index = g.index
        parts.append(m.drop(columns="date").rename(columns={col: "score"}))
    return pd.concat(parts).loc[t.index]


def screen_one(t: pd.DataFrame, tables: dict, col: str, sign: str, pr: dict) -> dict:
    from research.nnpredict.screen import decide, score

    a = attach(t, tables, col)
    if sign == "-":
        a = a.assign(score=-a["score"])
    return decide(score(a), {"screen_rule": pr["relevance_rule"]}) | {"column": col, "direction": sign}


def _instrument_ok(t: pd.DataFrame, pr: dict) -> tuple[bool, dict]:
    chk = t.attrs.get("check") or {}
    want = pr["relevance_rule"]
    # trades() itself raises when the count or gap does not reproduce; this re-checks the recorded values
    ok = len(t) == want["trades"] and abs((chk.get("control_at_share_gap") or 0) - want["control_gap"]) < 1e-3
    return ok, {"trades": len(t), "check": chk}


def main(argv=None) -> int:
    from research.new_inputs import trades as TR
    from research.nnmt import judge as J

    stage = (argv or sys.argv[1:] or ["sizer"])[0]
    path, pr = J._prereg("screens")
    bad = J._refused(path, {**pr, "folds": [], "arms": {}})
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    sel = json.loads((HERE / "verdicts_models.json").read_text())["selected"]
    t = TR.trades()
    ok, inst = _instrument_ok(t, pr)
    if not ok:
        print("REFUSED: the 412-trade / 0.1004 control-gap instrument did not reproduce", inst)
        return 2
    folds = sorted(t["fold"].unique())
    if stage == "sizer":
        model = sel.upper()
        tables = {f: pd.read_parquet(SIG / f"{model}_{f}.parquet") for f in folds}
        r = screen_one(t, tables, "p_buy", "+", pr) | {"model": model, "instrument": inst}
        (HERE / "screens_sizer.json").write_text(json.dumps(r, indent=1, default=str))
        print(json.dumps({k: v for k, v in r.items() if k != "per_fold"}, default=str))
        return 0
    model = f"{sel}_ah".upper()
    tables = {f: pd.read_parquet(SIG / f"{model}_{f}.parquet") for f in folds}
    out = {h: screen_one(t, tables, spec["column"], spec["direction"], pr)
           for h, spec in pr["heads"].items()}
    p = HERE / "screens_heads.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev["relevance"] = out
    prev["relevance_instrument"] = inst
    p.write_text(json.dumps(prev, indent=1, default=str))
    print(json.dumps({h: {k: r.get(k) for k in ("gap_share50", "partial", "sign_folds", "pass")}
                      for h, r in out.items()}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
