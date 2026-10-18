"""ATR-barrier study, direction readout (preregistration_dir.json): S1 re-scored, picks, instrument
tables and verdicts for cs_tba_d / cs_tbb_d (the trained TBA / TBB weights, agree score
(1 + p_buy - p_sell) / 2). ONE step per process:

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.dir_arms screen
    ... dir_arms pick --arm cs_tba_d --fold f00 [--verify]
    ... dir_arms judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_dir.json"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
DATA = STRAT.parents[1] / "user_data" / "data" / "tb_dir"
ARMS = {"cs_tba_d": "CS_TBA", "cs_tbb_d": "CS_TBB"}
TARGET = "label:tbatr_buy"


def dir_score(t: pd.DataFrame) -> np.ndarray:
    return 0.5 * (1.0 + t["p_buy"].to_numpy(float) - t["p_sell"].to_numpy(float))


def _tables(panel, prefix: str, direction: bool) -> np.ndarray:
    from research.indicator_screen import windows as W

    p = np.full(len(panel), np.nan)
    key = pd.DataFrame({"date": panel.date.tz_convert("UTC"), "pair": np.asarray(panel.pairs)[panel.pair],
                        "i": np.arange(len(panel))})
    for f in W.folds():
        t = pd.read_parquet(SIG / f"{prefix}_{f.id}.parquet")
        t["date"] = pd.to_datetime(t["date"], utc=True)
        t["s"] = dir_score(t) if direction else t["p_buy"].to_numpy(float)
        m = key.merge(t[["date", "pair", "s"]], on=["date", "pair"], how="inner")
        p[m["i"].to_numpy()] = m["s"].to_numpy(float)
    return p


def screen() -> dict:
    from research.indicator_screen import scores as S
    from research.indicator_screen import windows as W
    from research.indicator_screen.panel import Panel

    panel = Panel(tradeable_only=True)
    y = panel.columns([TARGET])[TARGET].astype(float)
    sc = {"ARO": _tables(panel, "ARO", False), **{a: _tables(panel, p, True) for a, p in ARMS.items()}}
    rows = {}
    for f in W.folds():
        base = f.mask(panel.date) & panel.data_ok & np.isfinite(y) & np.all([np.isfinite(v) for v in sc.values()], axis=0)
        r = {"half": f.half}
        for pop, m in (("cand", base & panel.candidate), ("all", base)):
            i = np.flatnonzero(m)
            for k, p in sc.items():
                r[f"{k}_{pop}_auc"] = S.auc(p[i], y[i])
        rows[f.id] = r
    summary = {h: {k: float(np.nanmedian([r[k] for r in rows.values() if r["half"] == h]))
                   for k in rows["f00"] if k != "half"} for h in ("early", "late")}
    s1 = {a: all(summary[h][f"{a}_cand_auc"] > summary[h]["ARO_cand_auc"] for h in ("early", "late")) for a in ARMS}
    out = {"label": "S1 on the direction readout (preregistration_dir.json)", "summary": summary,
           "s1_pass": s1, "folds": rows}
    (HERE / "screen_dir.json").write_text(json.dumps(out, indent=1))
    for h in ("early", "late"):
        print(h, {k: round(v, 3) for k, v in summary[h].items()})
    print("S1 pass:", s1)
    return out


def _plan_path(arm: str) -> Path:
    from research.control_sizer import pick as CP
    return CP.FOLDS / f"plan_{arm}.json"


def pick(arm: str, fid: str) -> tuple[float, dict]:
    import run_folds as rf
    from research.control_sizer import pick as CP
    from research.hgb_sizer.pick import _scores
    from research.width_relative import pick as WP

    plan_path = _plan_path(arm)
    plan = json.loads(plan_path.read_text())
    rf.generate_fold(CP._fold(plan, fid), plan["base_strategy"], plan["family"], CP._snapshot(plan))
    frames, strat = WP._frames(plan, fid, 0.0)
    score, _ = _scores(frames)
    share = float(strat.AGREE_SHARE)
    t, achieved = WP.threshold_for_share(score, share)
    extra = {"target_share": share, "achieved_share": round(achieved, 4), "run_starts": len(score)}
    return CP.store(plan_path, fid, t, extra), extra


def verify(arm: str, fid: str) -> bool:
    from research.hgb_sizer.pick import verify as V
    return V(fid, _plan_path(arm))


def instrument_tables(arm: str) -> Path:
    out = DATA / arm
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted(SIG.glob(f"{ARMS[arm]}_f??.parquet")):
        t = pd.read_parquet(src)
        t["p_buy"] = dir_score(t)
        t.to_parquet(out / f"signals_{src.stem.rsplit('_', 1)[1]}.parquet", index=False)
    return out


def arm_check(arm: str, spec: dict, pr: dict) -> dict:
    from evaluation.fingerprint import load_strategy_class
    from research.control_sizer.instrument import _plan
    from research.hgb_sizer import instrument as HI
    from research.label_screen.instrument import sz_arm_summary

    HI.OUT = instrument_tables(arm)
    plan, wplan = _plan(arm), _plan(spec["width"])
    cuts = load_strategy_class(plan["base_strategy"]).WIDTH_CUTS
    thr = plan.get("thresholds", {})
    folds = {f["fold_id"]: HI.fold_checks(f, fw, thr, cuts) for f, fw in zip(plan["folds"], wplan["folds"])}
    out = {"folds": folds, "halves": {}}
    for h in ("early", "late"):
        s = sz_arm_summary({k: folds[k] for k in pr[h]})
        n = sum(folds[k]["n_trades"] for k in pr[h])
        miss = sum(folds[k]["entries_without_hgb_p"] for k in pr[h])
        s["missing_score_share"] = round(miss / n, 4) if n else None
        s["missing_score_ok"] = bool(n) and miss / n <= HI.MAX_MISSING
        out["halves"][h] = s
    out["pass"] = all(s["S0_pass"] and not s["S0_inconclusive"] and s["missing_score_ok"]
                      for s in out["halves"].values())
    return out


def judge() -> int:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        print(f"REFUSED: {rel} has uncommitted changes")
        return 2
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    passed = [a for a, ok in json.loads((HERE / "screen_dir.json").read_text())["s1_pass"].items() if ok]
    bad = [x for a in passed for x in CJ.refusals({**pr, "arm": a}, commit_ts=ts)] if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "arms": {}}
    for arm in passed:
        spec = pr["arms"][arm]
        inst = arm_check(arm, spec, pr)
        vw = vc = vs = None
        if inst["pass"]:
            vw, vc = judge_one(pr, arm, spec["width"]), judge_one(pr, arm, spec["control"])
            vs = judge_one(pr, arm, spec["secondary"])
        r = {"instrument": inst, "decision": CJ.decide(pr, {"pass": inst["pass"]}, vw, vc),
             "question_vs_cs_aro": None if vs is None else {k: vs.get(k) for k in ("verdict", "early_gain_pp", "late_gain_pp")},
             "vs_width": vw, "vs_control": vc, "vs_cs_aro": vs}
        out["arms"][arm] = r
        print(arm, "instrument", inst["pass"], json.dumps({k: r[k] for k in ("decision", "question_vs_cs_aro")}, default=str), flush=True)
    (HERE / "verdicts_dir.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "pick", "judge"))
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--fold")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "screen":
        screen()
        return 0
    if a.step == "judge":
        return judge()
    if a.verify:
        return 0 if verify(a.arm, a.fold) else 1
    t, extra = pick(a.arm, a.fold)
    print(f"{a.arm} {a.fold}: threshold {t} {json.dumps(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
