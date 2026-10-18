"""Seed-2 repeat (preregistration_seed2.json): copy the seed-2 picks into the six run plans.

Reads plan_cs_aro_s2train.json's thresholds (research.control_sizer.pick --seed 2) and the per-fold
--verify exit codes passed as f00=0,f01=0,..., writes research/tb_model/seed2_picks.json, and fills
`thresholds` in plan_<book>_r2.json unchanged. Refuses if any fold lacks a pick.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.seed2_fill f00=0,f01=0,...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
FOLDS = STRAT / "evaluation" / "folds"
BOOKS = ("cs_aro", "cs_aro_all2k", "cs_aro_all3k", "cs_aro_hpany_lev2", "cs_aro_hpany", "cs_aro_lev2")


def main(argv=None) -> int:
    codes = dict(x.split("=") for x in (argv or sys.argv[1:])[0].split(","))
    train = json.loads((FOLDS / "plan_cs_aro_s2train.json").read_text())
    th = train.get("thresholds", {})
    ids = [f["fold_id"] for f in train["folds"]]
    if set(th) != set(ids):
        print(f"REFUSED: picks missing for {sorted(set(ids) - set(th))}")
        return 2
    picks = {"folds": {f: {"threshold": th[f], "pick": train["threshold_picks"][f], "verify": codes.get(f) == "0"}
                       for f in ids}}
    (Path(__file__).resolve().parent / "seed2_picks.json").write_text(json.dumps(picks, indent=1))
    for b in BOOKS:
        p = FOLDS / f"plan_{b}_r2.json"
        plan = json.loads(p.read_text())
        plan["thresholds"] = {f: th[f] for f in ids}
        p.write_text(json.dumps(plan, indent=2))
    print("filled", {f: th[f] for f in ids}, "verify", {f: v["verify"] for f, v in picks["folds"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
