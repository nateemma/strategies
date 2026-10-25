"""T053 follow-up: model-free width-book cuts (research/train_window/preregistration_width_free.json).

w_aro's cut is matched to the ARO model book's entry count, so the width bar moves with ARO's seed and
training window. Two rules that read no model:

  w_free   cut_i = the k-th largest bb_width (k = ceil(S * n)) over the n candidate bars of the 90 days
           before fold i's test_start, all pairs pooled; S fixed in the pre-registration
  w_fix    one constant cut (DESCRIPTIVE)

Steps (one process each):
  trail    the f00 pre-window table (20221002-20230101, f00 class)  -> signals/ARO_pre_f00.parquet
  cuts     both rules' cuts, asserting legacy == purged plans       -> width_free/cuts.json
  build    NNNC_W_FREE / NNNC_W_FIX (+ .json), plan_w_free / plan_w_fix (from plan_w_aro), model keys asserted
  check    the arm's own enter_long on the test window == the candidate count at its cut (w_aro too)
  report   pp per fold / half, realised S, comparators            -> width_free/report.json

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.train_window.width_free cuts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]
HERE = Path(__file__).resolve().parent
OUT = HERE / "width_free"
FOLDS = STRAT / "evaluation" / "folds"
ARMS = STRAT / "regime" / "arms"
PREREG = json.loads((HERE / "preregistration_width_free.json").read_text())
S = PREREG["arms"]["w_free"]["S"]
FIX = 0.10447016087339737
TRAIL_DAYS = 90
PRE_F00 = HERE / "signals" / "ARO_pre_f00.parquet"
PRE_F00_RANGE = "20221002-20230101"
COMPARATORS = ("w_aro", "cs_aro_tr", "cs_aro", "wf_aro_g24")


def _plan(tag: str) -> dict:
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def _sig(fid: str) -> pd.DataFrame:
    from research.nnnc_framework import entries as E

    return pd.read_parquet(E.SIG / f"ARO_{fid}.parquet")


def _cands(t: pd.DataFrame, fid: str) -> np.ndarray:
    """bb_width of the candidate bars (width_twin._gates at the fold's params)."""
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    return t["bb_width"].to_numpy(float)[_gates(t, E.fold_params("NNNC_ARO", fid))]


def trail() -> int:
    from research.nnnc_framework.signals import fold_table

    f = _plan("wf_aro_g24")["folds"][0]
    t = fold_table(f["strategy_class"], PRE_F00_RANGE)
    PRE_F00.parent.mkdir(exist_ok=True)
    t.to_parquet(PRE_F00)
    print(json.dumps({"rows": len(t), "from": str(t["date"].min()), "to": str(t["date"].max())}))
    return 0


def _free_cuts(plan: dict) -> dict:
    pool = pd.concat([pd.read_parquet(PRE_F00)] + [_sig(f["fold_id"]) for f in plan["folds"]], ignore_index=True)
    pool = pool.assign(date=pd.to_datetime(pool["date"], utc=True)).drop_duplicates(["date", "pair"])
    out = {}
    for f in plan["folds"]:
        ts = pd.Timestamp(f["test_start"], tz="UTC")
        w = pool[(pool["date"] >= ts - pd.Timedelta(days=TRAIL_DAYS)) & (pool["date"] < ts)]
        c = np.sort(_cands(w, f["fold_id"]))[::-1]
        k = math.ceil(S * len(c))
        out[f["fold_id"]] = {"cut": float(c[k - 1]), "n": int(len(c)), "k": k,
                             "window": [str(w["date"].min()), str(w["date"].max())]}
    return out


def cuts() -> int:
    legacy, purged = _free_cuts(_plan("wf_aro_g24")), _free_cuts(_plan("wf_aro_g24_pw"))
    if {f: v["cut"] for f, v in legacy.items()} != {f: v["cut"] for f, v in purged.items()}:
        raise SystemExit("REFUSED: w_free cuts differ between the legacy and purged plans")
    OUT.mkdir(exist_ok=True)
    (OUT / "cuts.json").write_text(json.dumps({"S": S, "trail_days": TRAIL_DAYS, "w_free": legacy, "w_fix": FIX,
                                               "legacy_equals_purged": True}, indent=1))
    print(json.dumps({f: round(v["cut"], 5) for f, v in legacy.items()}))
    return 0


TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""{doc} (research/train_window/width_free.py; GENERATED, preregistration_width_free.json): w_aro with
model-free cuts; only WIDTH_CUTS differ from NNNC_W_ARO."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class {name}(NNNC_W_ARO):
    WIDTH_CUTS = {cuts}
'''


def build() -> int:
    from research.nnmt.build_arms import key_of

    c = json.loads((OUT / "cuts.json").read_text())
    src = _plan("w_aro")
    by_fold = {"w_free": {f["fold_id"]: c["w_free"][f["fold_id"]]["cut"] for f in src["folds"]},
               "w_fix": {f["fold_id"]: c["w_fix"] for f in src["folds"]}}
    docs = {"w_free": "Width book, cut = trailing-90-day candidate quantile at the fixed share S",
            "w_fix": "Width book, one constant cut (DESCRIPTIVE)"}
    for tag, cut in by_fold.items():
        name = f"NNNC_W_{tag.split('_')[1].upper()}"
        lit = "{\n" + "".join(f"        {f['train_timerange']!r}: {cut[f['fold_id']]!r},\n" for f in src["folds"]) + "    }"
        (STRAT / "NNNC" / f"{name}.py").write_text(TMPL.format(doc=docs[tag], name=name, cuts=lit))
        bj = json.loads((STRAT / "NNNC" / "NNNC_W_ARO.json").read_text())
        bj["strategy_name"] = name
        (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")
        p = json.loads(json.dumps(src))
        p.update(run_id=tag, base_strategy=name, width_note=f"model-free width book ({tag}); preregistration_width_free.json")
        for f in p["folds"]:
            f.update(strategy_class=f"{name}_{f['fold_id']}", arm_tag=f"{tag}_{f['fold_id']}", status="planned")
        (FOLDS / f"plan_{tag}.json").write_text(json.dumps(p, indent=1) + "\n")
    sys.path.insert(0, str(STRAT / "NNNC"))
    keys = {n: key_of(n) for n in ("NNNC_W_ARO", "NNNC_W_FREE", "NNNC_W_FIX")}
    if len(set(keys.values())) != 1:
        raise SystemExit(f"REFUSED: model keys differ: {keys}")
    print(json.dumps({"built": ["NNNC_W_FREE", "NNNC_W_FIX"], "key": keys["NNNC_W_ARO"]}))
    return 0


def _rec(tag: str, i: int) -> dict:
    return json.loads((ARMS / f"{tag}_f{i:02d}_F{i:02d}.json").read_text())


def _cut_of(tag: str, f: dict) -> float:
    import importlib

    sys.path.insert(0, str(STRAT / "NNNC"))
    name = _plan(tag)["base_strategy"]
    return getattr(importlib.import_module(name), name).WIDTH_CUTS[f["train_timerange"]]


def check(tags: list[str], folds: list[str] | None) -> int:
    """the arm's own enter_long on its test window == the candidates at its cut; one (tag, fold) per call
    keeps one model load per process. Results merge into width_free/check.json."""
    from research.nnnc_framework.signals import fold_table
    import run_folds as RF

    OUT.mkdir(exist_ok=True)
    cp = OUT / "check.json"
    res = json.loads(cp.read_text()) if cp.is_file() else {}
    for tag in tags:
        plan = _plan(tag)
        for f in plan["folds"]:
            fid = f["fold_id"]
            if folds and fid not in folds:
                continue
            RF.generate_fold(f, plan["base_strategy"], plan["family"], json.loads((STRAT.parents[1] / plan["params_snapshot"]).read_text()),
                             threshold=plan["thresholds"][fid], seed=1)
            got = int((fold_table(f["strategy_class"], f["test_timerange"])["enter_long"] == 1).sum())
            want = int((_cands(_sig(fid), fid) >= _cut_of(tag, f)).sum())
            res[f"{tag}:{fid}"] = {"got": got, "want": want}
            print(json.dumps({"arm": tag, "fold": fid, "got": got, "want": want}), flush=True)
    cp.write_text(json.dumps(res, indent=1))
    return 1 if any(v["got"] != v["want"] for v in res.values()) else 0


def report() -> int:
    chk = json.loads((OUT / "check.json").read_text())
    missing = [f"{t}:f{i:02d}" for t in ("w_free", "w_fix") for i in range(15) if f"{t}:f{i:02d}" not in chk]
    bad = {k: v for k, v in chk.items() if v["got"] != v["want"]}
    if missing or bad:
        raise SystemExit(f"REFUSED: entry check incomplete {missing} or failed {bad} (width_free/check.json)")
    half = {f: h for h in ("early", "late") for f in PREREG[h]}
    pp = lambda tag: {f"f{i:02d}": float(_rec(tag, i).get("wallet_profit_pct") or 0.0) for i in range(15)}  # noqa: E731
    sums = lambda d: {"early": round(sum(v for f, v in d.items() if half[f] == "early"), 3),  # noqa: E731
                      "late": round(sum(v for f, v in d.items() if half[f] == "late"), 3),
                      "total": round(sum(d.values()), 3)}
    out = {"label": "PROVISIONAL / DESCRIPTIVE -- no pass rule (preregistration_width_free.json)", "arms": {}, "beside": {}}
    for tag in ("w_free", "w_fix"):
        d = pp(tag)
        ent = {f"f{i:02d}": _rec(tag, i)["entry_condition_counts"]["all"]["pass"] for i in range(15)}
        cand = {fid: int(len(_cands(_sig(fid), fid))) for fid in ent}
        out["arms"][tag] = {**sums(d), "per_fold": {f: round(v, 3) for f, v in d.items()},
                            "trades": sum(_rec(tag, i).get("n_trades") or 0 for i in range(15)),
                            "entries": ent, "S_realised": {f: round(ent[f] / cand[f], 4) for f in ent}}
    base = pp("w_free")
    for tag in COMPARATORS:
        d = pp(tag)
        diff = [d[f] - base[f] for f in d]
        out["beside"][tag] = {**sums(d), "vs_w_free": {"mean_fold_diff": round(float(np.mean(diff)), 3),
                                                       "folds_ahead": int(sum(x > 0 for x in diff)), "folds": len(diff)}}
    out["beside"]["w_aro_draws"] = {"legacy_s1": 51.44, "legacy_s2": 54.77, "purged_s1": 37.83, "purged_s2": 47.66}
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {t: {h: v[h] for h in ("early", "late", "total")} for t, v in out[k].items() if isinstance(v, dict) and "total" in v}
                      for k in ("arms", "beside")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("trail", "cuts", "build", "check", "report"))
    ap.add_argument("--arms", default="w_free,w_fix")
    ap.add_argument("--folds")
    a = ap.parse_args(argv)
    return {"trail": trail, "cuts": cuts, "build": build, "report": report,
            "check": lambda: check(a.arms.split(","), a.folds.split(",") if a.folds else None)}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
