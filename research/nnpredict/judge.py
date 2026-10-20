"""039 NNPredict study: the pre-registered verdicts, per stage.

  models     instrument per arm; judge_one(arm, nnp_ridge); the R7 model selection; the baseline
             reported beside w_aro / cs_aro / st_g24_mom          -> verdicts_models.json
  framework  identity (fw0 trades_hash == the selected model's, 15/15) first; then each feature arm
             judge_one(arm, fw0) with liveness (folds_changed > 0) -> verdicts_framework.json
  threshold  the threshold amendment (preregistration_threshold.json): flip / excursion / H24 arms vs
             nnp_ridge_op, as the op stage; R7 re-applied over the op arms + these
                                                                     -> verdicts_threshold.json
  followups  [followups2/3, mix: the same with their own pre-registration] identity (when declared) and
             any instrument_files (each must have "pass": true) gate the verdicts;
             identity (revx trades_hash == st_g24_rev's, 15/15) gates revg; then each arm judge_one(arm,
             comparator), plus the also_reported_vs comparisons    -> verdicts_followups.json
  sizer      judge_one(nnp_cs, w_aro) (must PASS); judge_one(nnp_cs, cs_aro) reported
                                                                     -> verdicts_sizer.json

Each stage refuses while its pre-registration is uncommitted/dirty or any of its arm records predates
the commit (research.pair_triggers.judge.refusals).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.judge models|framework|sizer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.pair_triggers import judge as PJ  # noqa: E402

HERE = Path(__file__).resolve().parent
BASELINE = "nnp_ridge"
COMPARATORS = ("w_aro", "cs_aro", "st_g24_mom")


def _prereg(stage: str) -> tuple[Path, dict]:
    p = HERE / f"preregistration_{stage}.json"
    return p, json.loads(p.read_text())


def _refused(path: Path, pr: dict) -> list[str]:
    PJ.PREREG = path
    return PJ.refusals(pr)


def instrument(pr: dict, arm: str) -> dict:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan

    p = pool(_plan(arm))
    per = {f["fold_id"]: f.get("n_trades", 0) for f in p["per_fold"]}
    halves = {h: sum(per.get(f, 0) or 0 for f in pr[h]) for h in ("early", "late")}
    ok = (not p["folds_missing"]) and all(halves.values())
    return {"pass": bool(ok), "folds_missing": p["folds_missing"], "trades_by_half": halves,
            "trades": p["n_trades_pooled"], "pooled_pp": p.get("total_pp", p.get("pooled_pp"))}


def _no_worse_half(v: dict) -> bool:
    return all((v.get("halves", {}).get(h, {}) or {}).get("verdict") != "WORSENED" for h in ("early", "late"))


def select_model(baseline_pp: float, judged: dict) -> str:
    """R7: highest pooled total among arms with no WORSENED half vs the baseline; ties -> baseline."""
    best, best_pp = BASELINE, baseline_pp
    for arm, r in judged.items():
        v = r.get("alone")
        if not r["instrument"]["pass"] or v is None or not _no_worse_half(v):
            continue
        if v["arm_pp"] > best_pp:
            best, best_pp = arm, v["arm_pp"]
    return best


def models(stage: str = "models") -> int:
    """The model stage; `stage="op"` is the operating-point amendment (preregistration_op.json), the
    same procedure with that file's baseline and arms."""
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg(stage)
    global BASELINE
    BASELINE = pr.get("baseline", BASELINE)
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    base_inst = instrument(pr, BASELINE)
    out = {"label": pr["label"], "baseline": {"instrument": base_inst, "beside": {}}, "arms": {}}
    if not base_inst["pass"]:
        print("REFUSED: the baseline's instrument failed", base_inst)
        out["selected"] = None
        (HERE / f"verdicts_{stage}.json").write_text(json.dumps(out, indent=1, default=str))
        return 1
    for comp in COMPARATORS:
        v = judge_one(pr, BASELINE, comp)
        out["baseline"]["beside"][comp] = {k: v.get(k) for k in ("verdict", "arm_pp", "ctl_pp", "early_gain_pp",
                                                                 "late_gain_pp", "arm_trades", "ctl_trades")}
        print(f"baseline {BASELINE} beside {comp}: {v.get('arm_pp')} vs {v.get('ctl_pp')} "
              f"(early {v.get('early_gain_pp')}, late {v.get('late_gain_pp')})", flush=True)
    baseline_pp = None
    for arm, spec in pr["arms"].items():
        if arm == BASELINE:
            continue
        inst = instrument(pr, arm)
        r = {"instrument": inst}
        if inst["pass"]:
            v = judge_one(pr, arm, spec["comparator"])
            r["alone"] = v
            baseline_pp = v["ctl_pp"]
            print(f"{arm}: {v['verdict']} {v['arm_pp']} vs {v['ctl_pp']} early {v['early_gain_pp']} late "
                  f"{v['late_gain_pp']} paired {(v.get('paired') or {}).get('verdict')}", flush=True)
        else:
            r["decision"] = "INCONCLUSIVE (instrument)"
            print(f"{arm}: INCONCLUSIVE (instrument) {inst}", flush=True)
        out["arms"][arm] = r
    if baseline_pp is None:
        baseline_pp = judge_one(pr, BASELINE, "w_aro")["arm_pp"]
    out["baseline_pp"] = baseline_pp
    out["selected"] = select_model(baseline_pp, out["arms"])
    print("R7 selected model:", out["selected"], flush=True)
    if stage == "threshold":  # R7 again over the operating-point arms + these (same baseline, same rules)
        op = json.loads((HERE / "verdicts_op.json").read_text())
        out["selected_union"] = select_model(baseline_pp, {**op["arms"], **out["arms"]})
        print("R7 selected over op + threshold arms:", out["selected_union"], flush=True)
    (HERE / f"verdicts_{stage}.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def identity(pr: dict, base: str, fw0: str) -> dict:
    """fw0 (the ported framework, everything off) must reproduce the selected model's trades on every
    fold: equal trades_hash (the same record field judge_one compares)."""
    from research.nnnc_switches.judge import _recs

    ra, rb = _recs(fw0, pr["folds"]), _recs(base, pr["folds"])
    diff = [f for f in pr["folds"] if (ra.get(f) or {}).get("trades_hash") != (rb.get(f) or {}).get("trades_hash")]
    return {"pass": not diff, "folds_differing": diff, "folds": len(pr["folds"])}


def framework() -> int:
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg("framework")
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    base, fw0 = pr["selected"], pr["identity_arm"]
    out = {"label": pr["label"], "selected": base, "identity": identity(pr, base, fw0), "arms": {}}
    print("identity:", out["identity"], flush=True)
    if out["identity"]["pass"]:
        for arm in pr["feature_arms"]:
            inst = instrument(pr, arm)
            r = {"instrument": inst}
            if not inst["pass"]:
                r["decision"] = "INCONCLUSIVE (instrument)"
            else:
                v = judge_one(pr, arm, fw0)
                r["alone"] = v
                if not v.get("folds_changed"):
                    r["decision"] = "INERT (defect: no fold's trades changed)"
                else:
                    r["decision"] = ("candidate default" if v["verdict"] == "PASS" else v["verdict"])
                    whole = (v["arm_pp"] or 0) - (v["ctl_pp"] or 0)
                    if v["verdict"] == "PASS" and whole < pr["seed_rule_pp"]:
                        r["decision"] = "PASS, seed-2 rerun required (gain < 1 pp)"
                print(f"{arm}: {r['decision']} {v['arm_pp']} vs {v['ctl_pp']} early {v['early_gain_pp']} "
                      f"late {v['late_gain_pp']} changed {v.get('folds_changed')}", flush=True)
            out["arms"][arm] = r
    else:
        print("identity FAILED: feature verdicts are not read", flush=True)
    (HERE / "verdicts_framework.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def followups(stage: str = "followups") -> int:
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg(stage)
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    idn = pr.get("identity") or {}
    out = {"label": pr["label"], "arms": {},
           "identity": identity(pr, idn["reference"], idn["arm"]) if idn else {"pass": True, "note": "none declared"}}
    print("identity:", out["identity"], flush=True)
    extra = {f: json.loads((HERE / f).read_text()) if (HERE / f).exists() else {"pass": False, "missing": True}
             for f in pr.get("instrument_files", [])}
    out["instrument_files"] = {f: {k: v for k, v in x.items() if k != "per_fold"} for f, x in extra.items()}
    extra_ok = all(x.get("pass") for x in extra.values())
    for arm, spec in pr["arms"].items():
        if "comparator" not in spec:
            continue
        inst = instrument(pr, arm)
        r = {"instrument": inst}
        if not inst["pass"] or not extra_ok:
            r["decision"] = "INCONCLUSIVE (instrument)"
        elif idn and spec["comparator"] == idn["reference"] and not out["identity"]["pass"]:
            r["decision"] = "NOT READ (identity failed)"
        else:
            v = judge_one(pr, arm, spec["comparator"])
            r["alone"] = v
            whole = (v["arm_pp"] or 0) - (v["ctl_pp"] or 0)
            r["decision"] = ("INERT (defect: no fold's trades changed)" if not v.get("folds_changed")
                             else "PASS, seed-2 rerun required (gain < 1 pp)"
                             if v["verdict"] == "PASS" and whole < pr["seed_rule_pp"] else v["verdict"])
            r["paired"] = (v.get("paired") or {}).get("verdict")
            r["also"] = {c: {k: w.get(k) for k in ("verdict", "arm_pp", "ctl_pp", "early_gain_pp", "late_gain_pp")}
                         | {"paired": (w.get("paired") or {}).get("verdict")}
                         for c in spec.get("also_reported_vs", []) for w in [judge_one(pr, arm, c)]}
            print(f"{arm} vs {spec['comparator']}: {r['decision']} {v['arm_pp']} vs {v['ctl_pp']} early "
                  f"{v['early_gain_pp']} late {v['late_gain_pp']} paired {r['paired']}", flush=True)
            for c, w in r["also"].items():
                print(f"  vs {c}: {w}", flush=True)
        out["arms"][arm] = r
    (HERE / f"verdicts_{stage}.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def sizer(stage: str = "sizer") -> int:
    """`stage` names the pre-registration; one that declares "selected" sizes that model (follow-ups
    2: sizer_exc32), else the selected model; its screen file is pr["screen_file"] (default screen.json)."""
    from research.nnnc_switches.judge import judge_one

    from research.nnpredict.build_arms import selected

    path, pr = _prereg(stage)
    sel = pr.get("selected") or selected()[0]
    pr = {**pr, "sizer_arm": pr["sizer_arm"].format(selected=sel),
          "sizer_control": pr["sizer_control"].format(selected=sel)}
    pr["arms"] = {pr["sizer_arm"]: {}, pr["sizer_control"]: {}}
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    sf = HERE / pr.get("screen_file", "screen.json")
    scr = json.loads(sf.read_text()) if sf.exists() else {}
    out = {"label": pr["label"], "selected": sel, "screen": {k: v for k, v in scr.items() if k != "directions"},
           "screen_directions": {d: {k: v for k, v in x.items() if k != "per_fold"}
                                 for d, x in scr.get("directions", {}).items()}}
    if not scr.get("pass"):
        out["decision"] = "screen did not pass: no sizer backtest (a recorded result)"
        print(out["decision"])
    else:
        arm, ctl = pr["sizer_arm"], pr["sizer_control"]
        inst = instrument(pr, arm)
        out["instrument"] = inst
        if inst["pass"]:
            v = judge_one(pr, arm, ctl)
            out["primary"] = v
            out["secondary"] = {c: judge_one(pr, arm, c) for c in pr["secondary"]}
            whole = (v["arm_pp"] or 0) - (v["ctl_pp"] or 0)
            out["decision"] = ("PASS, seed-2 rerun required (gain < 1 pp)"
                               if v["verdict"] == "PASS" and whole < pr["seed_rule_pp"] else v["verdict"])
            print(f"{arm} vs {ctl}: {out['decision']} {v['arm_pp']} vs {v['ctl_pp']} early {v['early_gain_pp']} "
                  f"late {v['late_gain_pp']}", flush=True)
            for c, w in out["secondary"].items():
                print(f"  vs {c}: {w['verdict']} {w['arm_pp']} vs {w['ctl_pp']}", flush=True)
        else:
            out["decision"] = "INCONCLUSIVE (instrument)"
    (HERE / f"verdicts_{stage}.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    stage = (argv or sys.argv[1:] or ["models"])[0]
    if stage in ("models", "op", "threshold"):
        return models(stage)
    if stage == "framework":
        return framework()
    if stage.startswith("sizer"):
        return sizer(stage)
    if stage.startswith("followups") or stage == "mix":
        return followups(stage)
    raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
