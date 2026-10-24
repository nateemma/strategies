"""040 NNMT study: the pre-registered verdicts, per stage (research R3-R13; contracts/cli.md).

  identity   mt_fw0 (the framework port, everything off) vs mt (pre-port): trades_hash equal on 15/15
                                                                       -> verdicts_models.json["identity"]
  models     mt reported beside the comparators; mt_op vs mt_fw0 and each variant vs mt_op with the
             model-decided share; the R7 selection                     -> verdicts_models.json
  framework  each feature arm and the twin vs the selected model (INERT if no fold changed)
                                                                       -> verdicts_framework.json
  arms       a generic stage (screens = sizer + heads / transition): each pre-registered arm vs its comparator,
             gated by the pre-registration's instrument files          -> verdicts_<stage>.json

Each stage refuses while its pre-registration is uncommitted/dirty or any of its arm records predates the
commit (research.pair_triggers.judge.refusals).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.judge identity|models|framework|screens|transition
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
SIGNALS = STRAT / "research" / "nnnc_framework" / "signals"
COMPARATORS = ("wf_aro_g24", "cs_aro", "w_aro", "st_g24_rev")


def _prereg(stage: str) -> tuple[Path, dict]:
    p = HERE / f"preregistration_{stage}.json"
    return p, json.loads(p.read_text())


def _refused(path: Path, pr: dict) -> list[str]:
    from research.pair_triggers import judge as PJ

    PJ.PREREG = path
    return PJ.refusals(pr)


def _write(name: str, out: dict) -> None:
    (HERE / name).write_text(json.dumps(out, indent=1, default=str))


def instrument(pr: dict, arm: str) -> dict:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan

    p = pool(_plan(arm))
    per = {f["fold_id"]: f.get("n_trades", 0) for f in p["per_fold"]}
    halves = {h: sum(per.get(f, 0) or 0 for f in pr[h]) for h in ("early", "late")}
    ok = (not p["folds_missing"]) and all(halves.values())
    return {"pass": bool(ok), "folds_missing": p["folds_missing"], "trades_by_half": halves,
            "trades": p["n_trades_pooled"]}


def identity(pr: dict, ref: str, arm: str) -> dict:
    """`arm` must reproduce `ref`'s trades on every fold: equal trades_hash (regime/run_arm)."""
    from research.nnnc_switches.judge import _recs

    ra, rb = _recs(arm, pr["folds"]), _recs(ref, pr["folds"])
    missing = [f for f in pr["folds"] if f not in ra or f not in rb]
    diff = [f for f in pr["folds"] if f not in missing and ra[f].get("trades_hash") != rb[f].get("trades_hash")]
    return {"pass": not diff and not missing, "folds_differing": diff, "folds_missing": missing,
            "folds": len(pr["folds"])}


def guard_pass(df, buy: dict) -> "object":
    """Bars the entry guard chain lets through, mirroring BaseStrategy.get_entry_conditions exactly:
    volume > 0 and rvol > thr always; with guards enabled also atr_pct_roll >, guard_metric <,
    close_norm <, adx >, bb_width > their thresholds."""
    import numpy as np

    def col(c):
        return df[c].to_numpy(float)

    ok = (col("volume") > 0) & (col("rvol") > buy["entry_rvol_threshold"])
    if buy.get("entry_enable_guards", True):
        ok &= col("atr_pct_roll") > buy["entry_atr_pct"]
        ok &= col("guard_metric") < buy["entry_guard_threshold"]
        ok &= col("close_norm") < buy["entry_close_norm_threshold"]
        ok &= col("adx") > buy["entry_adx_threshold"]
        ok &= col("bb_width") > buy["entry_bb_width_threshold"]
    return np.asarray(ok)


def model_share(df, buy: dict) -> dict:
    """data-model § model-decided share: 1 - entries / guard-passing bars (higher = the model decides)."""
    g = guard_pass(df, buy)
    n_g = int(g.sum())
    n_e = int((df["enter_long"].fillna(0).to_numpy() == 1).sum())
    return {"guard_pass_bars": n_g, "entries": n_e, "model_share": (1.0 - n_e / n_g) if n_g else None}


def arm_share(tag: str, buy: dict, folds: list[str]) -> dict:
    import pandas as pd

    per = {}
    for f in folds:
        p = SIGNALS / f"{tag.upper()}_{f}.parquet"
        if p.exists():
            per[f] = model_share(pd.read_parquet(p), buy)
    g = sum(v["guard_pass_bars"] for v in per.values())
    e = sum(v["entries"] for v in per.values())
    return {"model_share": (1.0 - e / g) if g else None, "folds": len(per), "per_fold": per}


def _no_worse_half(v: dict) -> bool:
    return all((v.get("halves", {}).get(h, {}) or {}).get("verdict") != "WORSENED" for h in ("early", "late"))


def select_model(baseline: str, baseline_pp: float, judged: dict) -> str:
    """R7: highest pooled total among arms with no WORSENED half vs the baseline; ties -> baseline."""
    best, best_pp = baseline, baseline_pp
    for arm, r in judged.items():
        v = r.get("alone")
        if not r["instrument"]["pass"] or v is None or not _no_worse_half(v):
            continue
        if v["arm_pp"] > best_pp:
            best, best_pp = arm, v["arm_pp"]
    return best


def _decide(v: dict, seed_rule_pp: float) -> str:
    whole = (v["arm_pp"] or 0) - (v["ctl_pp"] or 0)
    if not v.get("folds_changed"):
        return "INERT (defect: no fold's trades changed)"
    if v["verdict"] == "PASS" and whole < seed_rule_pp:
        return "PASS, seed-2 rerun required (gain < 1 pp)"
    return v["verdict"]


def _brief(v: dict) -> dict:
    return {k: v.get(k) for k in ("verdict", "arm_pp", "ctl_pp", "early_gain_pp", "late_gain_pp",
                                  "arm_trades", "ctl_trades", "folds_changed")} | {
        "paired": (v.get("paired") or {}).get("verdict")}


def run_identity() -> int:
    path, pr = _prereg("models")
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    idn = identity(pr, pr["identity"]["reference"], pr["identity"]["arm"])
    out = json.loads((HERE / "verdicts_models.json").read_text()) if (HERE / "verdicts_models.json").exists() else {}
    out["identity"] = idn
    _write("verdicts_models.json", out)
    print("identity:", idn, flush=True)
    return 0 if idn["pass"] else 1


def models() -> int:
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg("models")
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = json.loads((HERE / "verdicts_models.json").read_text()) if (HERE / "verdicts_models.json").exists() else {}
    if not (out.get("identity") or {}).get("pass"):
        print("REFUSED: run `judge identity` first; the variants are not read until the port is proven inert")
        return 2
    out.update(label=pr["label"], baseline={"instrument": instrument(pr, "mt"), "beside": {}}, arms={})
    for comp in COMPARATORS:
        out["baseline"]["beside"][comp] = _brief(judge_one(pr, "mt", comp))
    shares = {}
    for arm, spec in pr["arms"].items():
        if spec.get("buy") is not None:
            shares[arm] = arm_share(arm, spec["buy"], pr["folds"])
    out["model_share"] = {a: {k: s[k] for k in ("model_share", "folds")} for a, s in shares.items()}
    for arm, spec in pr["arms"].items():
        if spec.get("comparator") is None or arm in ("mt", pr["identity"]["arm"]):
            continue
        inst = instrument(pr, arm)
        r = {"instrument": inst}
        share = (shares.get(arm) or {}).get("model_share")
        if not inst["pass"]:
            r["decision"] = "INCONCLUSIVE (instrument)"
        elif arm != "mt_op" and share is not None and share < pr["min_model_share"]:
            r["decision"] = f"INCONCLUSIVE (model-decided share {share:.3f} < {pr['min_model_share']})"
            r["alone"] = judge_one(pr, arm, spec["comparator"])
        else:
            v = judge_one(pr, arm, spec["comparator"])
            r["alone"] = v
            r["decision"] = _decide(v, pr["seed_rule_pp"])
        if "alone" in r:
            v = r["alone"]
            print(f"{arm} vs {spec['comparator']}: {r['decision']} {v['arm_pp']} vs {v['ctl_pp']} early "
                  f"{v['early_gain_pp']} late {v['late_gain_pp']} share {share}", flush=True)
        out["arms"][arm] = r
    variants = {a: r for a, r in out["arms"].items() if a in pr["selection"]["candidates"] and a != "mt_op"
                and not r.get("decision", "").startswith("INCONCLUSIVE")}
    base_pp = judge_one(pr, "mt_op", "mt_fw0")["arm_pp"]
    out["selected"] = select_model("mt_op", base_pp, variants)
    print("R7 selected model:", out["selected"], flush=True)
    _write("verdicts_models.json", out)
    return 0


def framework() -> int:
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg("framework")
    sel = json.loads((HERE / "verdicts_models.json").read_text())["selected"]
    pr = json.loads(json.dumps(pr).replace("{M}", sel))
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "selected": sel, "arms": {}}
    for arm, spec in pr["arms"].items():
        inst = instrument(pr, arm)
        r = {"instrument": inst}
        if not inst["pass"]:
            r["decision"] = "INCONCLUSIVE (instrument)"
        else:
            v = judge_one(pr, arm, spec["comparator"])
            r["alone"] = v
            r["decision"] = _decide(v, pr["seed_rule_pp"])
            if r["decision"] == "PASS":
                r["decision"] = "PASS: candidate default (seed 2 required)"
            r["also"] = {c: _brief(judge_one(pr, arm, c)) for c in spec.get("also_reported_vs", [])}
            print(f"{arm} vs {spec['comparator']}: {r['decision']} {v['arm_pp']} vs {v['ctl_pp']} early "
                  f"{v['early_gain_pp']} late {v['late_gain_pp']}", flush=True)
        out["arms"][arm] = r
    _write("verdicts_framework.json", out)
    return 0


def _by_plan_arm_tag(J) -> None:
    """judge_one finds records as `<tag>_<fid>_<FID>.json`; a seed-2 repeat's judge plan
    (plan_<arm>_rs1.json -> `<x>_fNN_s2` records) names them by its folds' arm_tag instead. Resolve
    through the plan when there is one: for every plan whose arm_tag is `<tag>_<fid>`, the same path."""
    orig = J._rec_path
    if getattr(orig, "_by_plan", False):
        return

    def rec_path(tag: str, fid: str) -> Path:
        p = FOLDS_DIR / f"plan_{tag}.json"
        if p.exists():
            for f in json.loads(p.read_text())["folds"]:
                if f["fold_id"] == fid:
                    return STRAT / "regime" / "arms" / f"{f['arm_tag']}_{fid.upper()}.json"
        return orig(tag, fid)

    rec_path._by_plan = True
    J._rec_path = rec_path


def arms(stage: str) -> int:
    """sizer / heads / transition: the pre-registration's arms vs their comparators, gated by its
    instrument (screen) files; an arm absent because its screen failed is recorded as such."""
    from research.nnnc_switches import judge as J
    from research.nnnc_switches.judge import judge_one

    _by_plan_arm_tag(J)
    path, pr = _prereg(stage)
    sel = json.loads((HERE / "verdicts_models.json").read_text())["selected"]
    pr = json.loads(json.dumps(pr).replace("{M}", sel))
    out = {"label": pr["label"], "selected": sel, "screens": {}, "arms": {}}
    for f in pr.get("instrument_files", []):
        p = HERE / f
        out["screens"][f] = json.loads(p.read_text()) if p.exists() else {"missing": True}
    run = {a: s for a, s in pr["arms"].items() if (FOLDS_DIR / f"plan_{a}.json").exists()}
    pr_run = {**pr, "arms": run}
    bad = _refused(path, pr_run)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    for arm, spec in pr["arms"].items():
        if arm not in run:
            out["arms"][arm] = {"decision": "not run: its screen did not pass (a recorded result)"}
            continue
        inst = instrument(pr, arm)
        r = {"instrument": inst}
        if not inst["pass"]:
            r["decision"] = "INCONCLUSIVE (instrument)"
        elif spec.get("rule") == "sign_vs_zero":
            r["alone"] = sign_vs_zero(pr, arm)
            r["decision"] = r["alone"]["verdict"]
            r["comparator_reported"] = {spec["comparator"]: sign_vs_zero(pr, spec["comparator"])}
        else:
            v = judge_one(pr, arm, spec["comparator"])
            r["alone"] = v
            r["decision"] = _decide(v, pr["seed_rule_pp"])
            r["also"] = {c: _brief(judge_one(pr, arm, c)) for c in spec.get("also_reported_vs", [])}
        if r["decision"].startswith("PASS"):
            # pr["seeds"] (rs1: reported) and, when committed, its amendment (bs1: decisive)
            decisive = None
            for kind in ("rs1", "bs1"):
                seed = f"{arm}_{kind}"
                if not (FOLDS_DIR / f"plan_{seed}.json").exists():
                    continue
                if kind == "bs1":
                    bad = _refused(*_prereg(f"{stage}_seed"))
                    if bad:
                        r["seed_bs1"] = {"decision": "REFUSED", "why": bad}
                        decisive = r["seed_bs1"]
                        continue
                si = instrument(pr, seed)
                if not si["pass"]:
                    s = {"instrument": si, "decision": "INCONCLUSIVE (instrument)"}
                elif spec.get("rule") == "sign_vs_zero":
                    s = {"instrument": si, "alone": sign_vs_zero(pr, seed)}
                    s["decision"] = s["alone"]["verdict"]
                else:
                    s = {"instrument": si, "alone": judge_one(pr, seed, spec["comparator"])}
                    s["decision"] = _decide(s["alone"], pr["seed_rule_pp"])
                r[f"seed_{kind}"] = s
                decisive = s
            if (HERE / f"preregistration_{stage}_seed.json").exists() and "seed_bs1" not in r:
                decisive = None  # the amendment names bs1 as the seed; rs1 alone does not decide
            if decisive is not None:
                r["decision"] = ("PASS (both seeds)" if decisive["decision"].startswith("PASS")
                                 else f"FAIL at seed 2 ({decisive['decision']})")
            else:
                r["decision"] += " (seed 2 pending)"
        print(f"{arm}: {r['decision']}", flush=True)
        out["arms"][arm] = r
    _write(f"verdicts_{stage}.json", out)
    return 0


FOLDS_DIR = STRAT / "evaluation" / "folds"


def sign_vs_zero(pr: dict, arm: str) -> dict:
    """research R12 tr_long: PASS if the pooled pp is > 0 in both halves AND the bootstrap 95% CI of the
    per-trade mean pp excludes 0 (seed 0)."""
    import numpy as np

    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan

    p = pool(_plan(arm))
    per = {f["fold_id"]: f["return_pct"] or 0.0 for f in p["per_fold"]}
    e, l = sum(per.get(f, 0) for f in pr["early"]), sum(per.get(f, 0) for f in pr["late"])
    pp = np.array([t["profit_ratio"] * 100 for t in p["trades"]], float)
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(pp, len(pp)).mean() for _ in range(2000)]) if len(pp) else np.array([0.0])
    lo, hi = np.percentile(means, [2.5, 97.5])
    ok = e > 0 and l > 0 and lo > 0
    return {"verdict": "PASS" if ok else "FAIL", "early_pp": e, "late_pp": l, "trades": int(len(pp)),
            "ci_mean_pp": [float(lo), float(hi)]}


def _early_view(tag: str, folds: list[str]) -> dict:
    from research.nnnc_switches.judge import _plan

    p = _plan(tag)
    return {**p, "folds": [f for f in p["folds"] if f["fold_id"] in folds]}


def arch_screen(pr: dict) -> dict:
    """preregistration_arch.json "screen": mt_ls vs mt_lr on the early folds only."""
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _recs

    arm, ctl, early, rule = pr["arm"], pr["control"], pr["early"], pr["rule"]
    ap, cp = pool(_early_view(arm, early)), pool(_early_view(ctl, early))
    if ap["folds_missing"] or cp["folds_missing"]:
        return {"verdict": "MISSING", "missing": ap["folds_missing"] + cp["folds_missing"]}
    share = arm_share(arm, pr["buy"], early)
    ra, rc = _recs(arm, early), _recs(ctl, early)
    changed = sum(ra[f]["trades_hash"] != rc[f]["trades_hash"] for f in early)
    gain = sum(ra[f]["wallet_profit_pct"] for f in early) - sum(rc[f]["wallet_profit_pct"] for f in early)
    cmp, rk = compare_data(ap, cp), risk.compare(ap["trades"], cp["trades"])
    p = rule["paired_map"][cmp["verdict"]]
    r = [rule["risk_map"][rk["verdict_sharpe"]], rule["risk_map"][rk["verdict_maxdd"]]]
    failed = []
    if not all(ra[f]["n_trades"] for f in early):
        failed.append("instrument")
    if share["folds"] < len(early) or share["model_share"] is None or share["model_share"] < pr["min_model_share"]:
        failed.append("model share")
    if cp["n_trades_pooled"] < pr["min_control_trades"] or changed == 0 \
            or p == "inconclusive" or "inconclusive" in r:
        verdict = "INCONCLUSIVE"
    else:
        if not gain > 0:
            failed.append("early gain")
        if p == "fail":
            failed.append("paired")
        if "fail" in r:
            failed.append("risk")
        verdict = "FAIL" if failed else "PASS"
    return {"arm": arm, "control": ctl, "verdict": verdict, "failed": failed, "folds": early,
            "arm_pp": round(sum(ra[f]["wallet_profit_pct"] for f in early), 3),
            "ctl_pp": round(sum(rc[f]["wallet_profit_pct"] for f in early), 3), "early_gain_pp": round(gain, 3),
            "arm_trades": ap["n_trades_pooled"], "ctl_trades": cp["n_trades_pooled"], "folds_changed": changed,
            "model_share": share["model_share"],
            "paired": {"verdict": cmp["verdict"], "delta_pp": cmp.get("delta_pp"), "ci_pp_yr": cmp.get("ci_pp_yr"),
                       "p": (cmp.get("bootstrap") or {}).get("p")},
            "risk": {"sharpe": rk["verdict_sharpe"], "maxdd": rk["verdict_maxdd"]}}


def arch() -> int:
    """preregistration_arch.json: the screen on f00-f07; the 15-fold confirmation once mt_ls has every fold."""
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg("arch")
    bad = _refused(path, {**pr, "arms": {pr["arm"]: {}}})
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "screen": arch_screen(pr)}
    if out["screen"]["verdict"] == "PASS" and not instrument(pr, pr["arm"])["folds_missing"]:
        out["confirm"] = judge_one(pr, pr["arm"], pr["control"])
    _write("verdicts_arch.json", out)
    print(json.dumps({k: v.get("verdict") for k, v in out.items() if isinstance(v, dict)}),
          json.dumps({k: out["screen"].get(k) for k in ("arm_pp", "ctl_pp", "early_gain_pp", "model_share", "failed")}))
    return 0


def main(argv=None) -> int:
    stage = (argv or sys.argv[1:] or ["models"])[0]
    if stage == "identity":
        return run_identity()
    if stage == "models":
        return models()
    if stage == "framework":
        return framework()
    if stage == "arch":
        return arch()
    if stage in ("screens", "transition"):  # screens = the sizer + head arms (preregistration_screens.json)
        return arms(stage)
    raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
