"""Control-sizer study: the pre-registered verdict for `cs_aro` (preregistration.json).

  1. Refuses while the pre-registration is uncommitted or dirty, or if any `cs_aro` record is
     older than its commit.
  2. Instrument (instrument.json) must pass in both halves; otherwise INCONCLUSIVE and no P&L is
     read.
  3. judge_one (research/nnnc_switches/judge.py: early gain > 0, late gain > 0, paired and risk
     not WORSENED) against the width book `w_aro` AND against the control `wf_aro_g24`; PASS
     (PROVISIONAL) needs both.
  4. Small margin: if any of the four half gains is under small_margin_pp, a PASS becomes
     INCONCLUSIVE until the seed-2 repeat (every margin > 0 on both seeds).

    PYTHONPATH=. ../../.venv/bin/python -m research.control_sizer.judge
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
ARMS_DIR = STRAT / "regime" / "arms"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def refusals(pr: dict, commit_ts: float | None = None, arms_dir: Path = ARMS_DIR) -> list[str]:
    out = []
    if commit_ts is None:
        rel = PREREG.relative_to(STRAT)
        if _git("status", "--porcelain", "--", str(rel)):
            out.append(f"{rel} has uncommitted changes")
        ts = _git("log", "-1", "--format=%ct", "--", str(rel))
        if not ts:
            return out + [f"{rel} is not committed"]
        commit_ts = float(ts)
    for f in pr["folds"]:
        p = arms_dir / f"{pr['arm']}_{f}_{f.upper()}.json"
        if p.is_file() and p.stat().st_mtime <= commit_ts:
            out.append(f"record older than the pre-registration commit: {p.name}")
    return out


def decide(pr: dict, inst: dict, vs_width: dict | None, vs_control: dict | None) -> dict:
    if not inst.get("pass"):
        return {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
    gains = {f"{c}_{h}": r[f"{h}_gain_pp"] for c, r in (("width", vs_width), ("control", vs_control))
             for h in ("early", "late")}
    verdicts = {"width": vs_width["verdict"], "control": vs_control["verdict"]}
    if "INCONCLUSIVE" in verdicts.values() or "MISSING" in verdicts.values():
        return {"decision": "INCONCLUSIVE", "reason": f"judge_one {verdicts}", "gains": gains}
    if "FAIL" in verdicts.values():
        return {"decision": "FAIL", "reason": f"judge_one {verdicts}", "gains": gains,
                "failed": {"width": vs_width.get("failed"), "control": vs_control.get("failed")}}
    small = min(gains.values()) < pr["small_margin_pp"]
    if small:
        return {"decision": "INCONCLUSIVE", "reason": "margin under small_margin_pp: seed-2 repeat",
                "gains": gains}
    return {"decision": "PASS (PROVISIONAL)", "reason": "both judge_one PASS", "gains": gains}


def main() -> int:
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = json.loads((HERE / "instrument.json").read_text())
    vw = vc = None
    if inst.get("pass"):
        vw = judge_one(pr, pr["arm"], pr["comparators"]["width"])
        vc = judge_one(pr, pr["arm"], pr["comparators"]["control"])
    out = {"label": pr["label"], "decision": decide(pr, inst, vw, vc),
           "vs_width": vw, "vs_control": vc}
    print(json.dumps(out["decision"], indent=1))
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
