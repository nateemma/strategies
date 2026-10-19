"""039 NNPredict study: the pre-registered verdicts, per stage.

  models     instrument per arm; judge_one(arm, nnp_ridge); the R7 model selection; the baseline
             reported beside w_aro / cs_aro / st_g24_mom          -> verdicts_models.json
  framework  identity (fw0 trades_hash == the selected model's, 15/15) first; then each feature arm
             judge_one(arm, fw0) with liveness (folds_changed > 0) -> verdicts_framework.json
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


def models() -> int:
    from research.nnnc_switches.judge import judge_one

    path, pr = _prereg("models")
    bad = _refused(path, pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    base_inst = instrument(pr, BASELINE)
    out = {"label": pr["label"], "baseline": {"instrument": base_inst, "beside": {}}, "arms": {}}
    if not base_inst["pass"]:
        print("REFUSED: the baseline's instrument failed", base_inst)
        out["selected"] = None
        (HERE / "verdicts_models.json").write_text(json.dumps(out, indent=1, default=str))
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
    (HERE / "verdicts_models.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    stage = (argv or sys.argv[1:] or ["models"])[0]
    if stage == "models":
        return models()
    raise SystemExit(f"stage {stage!r} is not implemented yet (framework / sizer come with their tasks)")


if __name__ == "__main__":
    raise SystemExit(main())
