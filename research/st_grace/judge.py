"""Judge the SimpleTrigger grace arms under the committed pre-registration (preregistration.json).

Per family: A = 2023-26 PASS by `nnnc_switches.judge.judge_one`; B = 2022 gain > 0 with the paired
and risk verdicts mapping to pass; ADOPT = A and B. Refuses (exit 2) if the pre-registration is
uncommitted/dirty or an arm record predates it.

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.st_grace.judge
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.nnnc_switches.judge import _plan, _sum, judge_one  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def refusals(pr: dict) -> list[str]:
    from evaluation.fingerprint import load_strategy_class

    out = []
    rel = PREREG.relative_to(STRAT)
    if _git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = _git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        return out + [f"{rel} is not committed"]
    for fam in pr["families"].values():
        grace = load_strategy_class(fam["class"]).SELL_OVERRIDES.get("stoploss_grace_hours")
        if grace != pr["declared"]["SELL_OVERRIDES.stoploss_grace_hours"]:
            out.append(f"{fam['class']} grace {grace!r} != frozen value")
        for tag, folds in ((fam["arm"], pr["folds"]), (fam["arm_2022"], pr["folds_2022"]),
                           (fam["control_2022"], pr["folds_2022"])):
            if tag == "st_e22_prom_c2":  # 028's committed 2022 control, reused as is
                continue
            for f in folds:
                p = STRAT / "regime" / "arms" / f"{tag}_{f}_{f.upper()}.json"
                if p.is_file() and p.stat().st_mtime <= float(ts):
                    out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def judge_2022(pr: dict, arm_tag: str, ctl_tag: str) -> dict:
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from evaluation.pool import pool

    rule = pr["rule"]
    ap, cp = pool(_plan(arm_tag)), pool(_plan(ctl_tag))
    miss = ap["folds_missing"] + cp["folds_missing"]
    if miss:
        return {"verdict": "MISSING", "missing": miss}
    ids = set(pr["folds_2022"])
    gain = _sum(ap, ids) - _sum(cp, ids)
    cmp = compare_data(ap, cp)
    rk = risk.compare(ap["trades"], cp["trades"])
    mapped = [rule["paired_map"][cmp["verdict"]], rule["risk_map"][rk["verdict_sharpe"]],
              rule["risk_map"][rk["verdict_maxdd"]]]
    failed = [] if gain > 0 else ["2022 gain"]
    failed += [n for n, m in zip(("paired", "risk sharpe", "risk maxdd"), mapped) if m == "fail"]
    verdict = "INCONCLUSIVE" if "inconclusive" in mapped else ("FAIL" if failed else "PASS")
    return {"arm": arm_tag, "control": ctl_tag, "verdict": verdict, "failed": failed,
            "gain_pp": gain, "arm_pp": _sum(ap, ids), "ctl_pp": _sum(cp, ids),
            "arm_trades": ap["n_trades_pooled"], "ctl_trades": cp["n_trades_pooled"],
            "per_fold": {f["fold_id"]: f["return_pct"] for f in ap["per_fold"]},
            "per_fold_ctl": {f["fold_id"]: f["return_pct"] for f in cp["per_fold"]},
            "paired": {"verdict": cmp["verdict"], "delta_pp": cmp.get("delta_pp"),
                       "p": (cmp.get("bootstrap") or {}).get("p")},
            "risk": rk}


def main() -> int:
    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    res = {"label": pr["label"], "families": {}}
    for key, fam in pr["families"].items():
        a = judge_one(pr, fam["arm"], fam["control"])
        b = judge_2022(pr, fam["arm_2022"], fam["control_2022"])
        decision = "ADOPTED" if a["verdict"] == "PASS" and b["verdict"] == "PASS" else \
            ("INCONCLUSIVE" if "INCONCLUSIVE" in (a["verdict"], b["verdict"]) else "NOT ADOPTED")
        res["families"][key] = {"decision": decision, "A_2023_26": a, "B_2022": b}
        print(f"{key}: {decision} -- 2023-26 {a['verdict']} ({a.get('early_gain_pp', 0):+.2f}/"
              f"{a.get('late_gain_pp', 0):+.2f}), 2022 {b['verdict']} ({b.get('gain_pp', 0):+.2f})")
    (HERE / "verdicts.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
