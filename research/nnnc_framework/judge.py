"""Judge the NNNC framework study's stage-2 arms under its committed pre-registration.

Each candidate (chosen by `select.py` from early-half screens) is declared on the four NNNC models
and run walk-forward f00-f14 on their own per-fold weights (model key, no training). Per model:
PASS / FAIL / INCONCLUSIVE by `nnnc_switches.judge.judge_one` with this study's rule; a candidate
is ADOPTED for NNNC only if it passes on ARO and on at least `replications_needed` of the others.
Refuses (exit 2) if the pre-registration is uncommitted/dirty or an arm record predates it.

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.nnnc_framework.judge
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.nnnc_switches.judge import judge_one  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def refusals(pr: dict) -> list[str]:
    out = []
    rel = PREREG.relative_to(STRAT)
    if _git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = _git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        return out + [f"{rel} is not committed"]
    for c in pr["candidates"]:
        for m, arm in c["arms"].items():
            for f in pr["folds"]:
                p = STRAT / "regime" / "arms" / f"{arm}_{f}_{f.upper()}.json"
                if p.is_file() and p.stat().st_mtime <= float(ts):
                    out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def main() -> int:
    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    res = {"label": pr["label"], "candidates": {}}
    for c in pr["candidates"]:
        per = {m: judge_one(pr, arm, c["controls"][m]) for m, arm in c["arms"].items()}
        reps = [m for m in per if m != pr["primary_model"]]
        n_pass = sum(per[m]["verdict"] == "PASS" for m in reps)
        prim = per[pr["primary_model"]]["verdict"]
        if prim == "PASS" and n_pass >= pr["replications_needed"]:
            decision = "ADOPTED"
        elif prim == "INCONCLUSIVE" or any(per[m]["verdict"] in ("INCONCLUSIVE", "MISSING")
                                           for m in per):
            decision = "NOT ADOPTED (inconclusive cells present)" if prim == "FAIL" \
                else "INCONCLUSIVE"
        else:
            decision = "NOT ADOPTED"
        res["candidates"][c["name"]] = {"lever": c["lever"], "value": c["value"],
                                        "decision": decision, "per_model": per}
    (HERE / "verdicts.json").write_text(json.dumps(res, indent=1, default=str))
    for n, r in res["candidates"].items():
        cells = ", ".join(f"{m} {v['verdict']} ({v['early_gain_pp']:+.2f}/{v['late_gain_pp']:+.2f})"
                          if "early_gain_pp" in v else f"{m} {v['verdict']}"
                          for m, v in r["per_model"].items())
        print(f"{n}: {r['decision']} -- {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
