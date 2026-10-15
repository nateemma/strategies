"""035 T026: S2 -- does adding a survivor input sharpen the model's ranking of the early
width-book trades? (FR-015; the control sizer's instrument, early half only.)

Per survivor: its NI_<T> signal tables on f00-f07, agree = p_buy > the fold's sizer threshold
(plan_ni_<t>_sz.json, 0.5 share on the validation tail, research.control_sizer.pick), on the same
412 trades (research.new_inputs.trades, which first reproduces the control's 0.1004 gap). PASS =
early gap > the control-at-share gap with <= 2 of 8 folds degenerate (agree share 0 or 1). The S2
winner is the passer with the largest gap; margins are recorded (S2 selects, S3 decides).

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.separation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.new_inputs import arms as A  # noqa: E402


HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"
FOLDS = STRAT / "evaluation" / "folds"
EARLY = tuple(f"f{i:02d}" for i in range(8))
MAX_DEGENERATE = 2


def gap_stats(pp, agree) -> dict:
    pp, agree = np.asarray(pp, float), np.asarray(agree, bool)
    a, d = pp[agree], pp[~agree]
    g = None if not len(a) or not len(d) else round(float(a.mean() - d.mean()), 4)
    return {"n": int(len(pp)), "agree_n": int(agree.sum()), "gap": g,
            "agree_share": round(float(agree.mean()), 4) if len(pp) else None}


def degenerate(t: pd.DataFrame, col: str) -> list[str]:
    s = t.groupby("fold")[col].mean()
    return sorted(s[(s == 0) | (s == 1)].index)


def judge(rows: dict, reference: float) -> dict:
    passed = {k: v for k, v in rows.items() if v.get("gap") is not None
              and len(v["degenerate_folds"]) <= MAX_DEGENERATE and v["gap"] > reference}
    for k, v in rows.items():
        v["S2_pass"] = k in passed
        v["margin"] = None if v.get("gap") is None else round(v["gap"] - reference, 4)
    win = max(passed, key=lambda k: passed[k]["gap"]) if passed else None
    return {"winner": win}


def main(argv=None) -> int:
    import argparse

    from research.new_inputs import trades as TR

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=A.MODELS, default="ARO")
    model = ap.parse_args(argv).model
    sfx = "" if model == "ARO" else "_gbr"
    ref_t = TR.trades(model=model)
    reference = ref_t.attrs["check"]["control_at_share_gap"]
    rows = {}
    for tag in A.survivors(model):
        thr = json.loads((FOLDS / f"plan_{A.plan_tag(tag, 'sz', model)}.json").read_text()) \
            .get("thresholds", {})
        tn = A.table_name(tag, model)
        missing = [f for f in EARLY if f not in thr or not (SIG / f"{tn}_{f}.parquet").is_file()]
        if missing:
            rows[tag] = {"status": "INCONCLUSIVE", "reason": f"tables/thresholds missing {missing}",
                         "gap": None, "degenerate_folds": []}
            continue
        t = TR.trades(lambda f, tn=tn: pd.read_parquet(SIG / f"{tn}_{f}.parquet"), model=model)
        t["agree"] = [p > float(thr[f]) for p, f in zip(t["p_model"], t["fold"], strict=True)]
        r = gap_stats(t["pp"], t["agree"]) | {"degenerate_folds": degenerate(t, "agree")}
        r["per_fold"] = {f: gap_stats(g["pp"], g["agree"]) for f, g in t.groupby("fold")}
        rows[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "per_fold"}), flush=True)
    j = judge(rows, reference)
    out = {"label": f"035 S2 {model} (pre-registered; early half only)", "reference_gap": reference,
           "inputs": rows, "s2_winner": j["winner"]}
    (HERE / f"separation{sfx}.json").write_text(json.dumps(out, indent=1, default=str))
    vp = HERE / f"verdicts{sfx}.json"
    v = json.loads(vp.read_text()) if vp.is_file() else {}
    v["s2_winner"] = {"tag": j["winner"]} if j["winner"] else None
    vp.write_text(json.dumps(v, indent=1))
    print("reference", reference, "S2 winner:", j["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
