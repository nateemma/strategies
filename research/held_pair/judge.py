"""031 T027: judge the held-pair add (DCA) arms under the committed pre-registration.

NNNC books: `research.nnnc_switches.judge.judge_one` over f00-f14 (late half confirmatory).
SimpleTrigger books: `research.st_grace.judge.judge_2022` on e00-e02 (confirmatory), with the
2023-26 `judge_one` result reported as exploratory. On both: zero adds -> INCONCLUSIVE.
Each book also gets exposure, the crowd-out split and the leverage control
(`research.held_pair.exposure`).

Refuses (exit 2) if the pre-registration is uncommitted or dirty, or if any DCA arm record not
listed in its `exempt_records` predates its commit.

    PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.judge
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
ARMS = STRAT / "regime" / "arms"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def _rec_path(tag: str, fid: str) -> Path:
    return ARMS / f"{tag}_{fid}_{fid.upper()}.json"


def _arm_folds(pr: dict) -> list[tuple[str, str]]:
    out = []
    for b in pr["books"].values():
        out += [(b["arm"], f) for f in b["folds"]]
        if b.get("arm_2022"):
            out += [(b["arm_2022"], f) for f in pr["folds_2022"]]
    return out


def refusals(pr: dict, prereg: Path = PREREG, commit_ts: float | None = None) -> list[str]:
    out = []
    if commit_ts is None:
        rel = prereg.relative_to(STRAT)
        if _git("status", "--porcelain", "--", str(rel)):
            out.append(f"{rel} has uncommitted changes")
        ts = _git("log", "-1", "--format=%ct", "--", str(rel))
        if not ts:
            return out + [f"{rel} is not committed"]
        commit_ts = float(ts)
    exempt = {tuple(x.split(":")) for x in pr.get("exempt_records", [])}
    for tag, f in _arm_folds(pr):
        p = _rec_path(tag, f)
        if (tag, f) in exempt or not p.is_file():
            continue
        if p.stat().st_mtime <= commit_ts:
            out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def _n_adds(tag: str, folds: list[str]) -> dict:
    per = {}
    for f in folds:
        p = _rec_path(tag, f)
        per[f] = json.loads(p.read_text()).get("n_adds", 0) if p.is_file() else None
    return per


def _zero_adds(verdict: dict, adds: dict) -> dict:
    if verdict.get("verdict") in ("PASS", "FAIL") and not sum(v or 0 for v in adds.values()):
        verdict = {**verdict, "verdict": "INCONCLUSIVE", "inconclusive_reason": "zero adds"}
    return verdict


def judge_book(pr: dict, key: str, b: dict) -> dict:
    from research.held_pair import exposure as X
    from research.nnnc_switches.judge import judge_one
    from research.st_grace.judge import judge_2022

    adds = _n_adds(b["arm"], b["folds"])
    a = _zero_adds(judge_one(pr, b["arm"], b["control"]), adds)
    ex = X.report(b["arm"], b["control"])
    res = {"book": key, "kind": b["kind"], "A_2023_26": a, "n_adds": adds, "exposure": ex,
           "beats_leverage": ex["beats_leverage"]}
    if b["kind"] == "NNNC":
        res["evidence"] = "late half CONFIRMATORY (PROVISIONAL: one era); early half EXPLORATORY"
        decision = {"PASS": "ADOPTED", "FAIL": "NOT ADOPTED"}.get(a["verdict"], "INCONCLUSIVE")
    else:
        adds22 = _n_adds(b["arm_2022"], pr["folds_2022"])
        b22 = _zero_adds(judge_2022(pr, b["arm_2022"], b["control_2022"]), adds22)
        ex22 = X.report(b["arm_2022"], b["control_2022"])
        res.update({"B_2022": b22, "n_adds_2022": adds22, "exposure_2022": ex22,
                    "beats_leverage_2022": ex22["beats_leverage"],
                    "evidence": "2022 CONFIRMATORY; 2023-26 EXPLORATORY"})
        decision = {"PASS": "ADOPTED", "FAIL": "NOT ADOPTED"}.get(b22["verdict"], "INCONCLUSIVE")
    res["decision"] = decision
    return res


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prereg", default=str(PREREG))
    a = ap.parse_args(argv)
    prereg = Path(a.prereg).resolve()
    pr = json.loads(prereg.read_text())
    bad = refusals(pr, prereg)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "books": {}}
    for key, b in pr["books"].items():
        r = judge_book(pr, key, b)
        out["books"][key] = r
        a = r["A_2023_26"]
        line = (f"{key}: {r['decision']} -- 2023-26 {a['verdict']} "
                f"({a.get('early_gain_pp', 0):+.2f}/{a.get('late_gain_pp', 0):+.2f})")
        if "B_2022" in r:
            line += f", 2022 {r['B_2022']['verdict']} ({r['B_2022'].get('gain_pp', 0):+.2f})"
        line += (f", adds {sum(v or 0 for v in r['n_adds'].values())}, leverage "
                 f"{r['exposure']['leverage']['leverage_pp']:+.2f}")
        print(line)
    name = "verdicts.json" if prereg == PREREG else f"verdicts_{prereg.stem.split('_', 1)[-1]}.json"
    (HERE / name).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
