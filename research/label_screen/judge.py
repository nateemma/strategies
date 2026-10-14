"""034 T013: staged verdicts under the committed pre-registration (FR-012..FR-014, FR-018).

Per candidate: S0a (audit.json) -> S0 (instrument.json) -> S1 (separation.json) -> survivor
(survivors.json). Per survivor arm (role sz | es; BOG sz only), ARO, early folds f00-f07:
  S2  early pp > width book (`w_aro`) AND > control (`wf_aro_g24`); a minimum margin under
      `small_margin_pp` is resolved by the paired seed-2 repeat (every margin > 0 on BOTH seeds),
      and fails otherwise; the arm's S2 instrument (instrument.json "arms") must pass first.
The S2 winner = the passing arm with the largest minimum margin. Only the winner reaches
  S3  judge_one(winner, control) PASS AND late pp > width book; small-margin rule as S2;
and on an S3 PASS the GBR repeat (`ls_gbr_<tag>_<role>` vs `fs_gbr_gr` / `w_gbr`, all 15 folds,
S2 + S3 rules). A failed stage stops that path: later stages are `not_reached` and their records
are not read. Outcomes: PASS (PROVISIONAL) | FAIL | INCONCLUSIVE; never adopted.

Refuses (exit 2) if the pre-registration is uncommitted or dirty, or any `ls_*` arm record predates
its latest commit (names in `exempt_records` excepted).

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.judge [--seed-repeat 2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.label_screen import candidates as C  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
ARMS_DIR = STRAT / "regime" / "arms"
NR = "not_reached"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def refusals(pr: dict, commit_ts: float | None = None, arms_dir: Path | None = None) -> list[str]:
    out = []
    if commit_ts is None:
        rel = PREREG.relative_to(STRAT)
        if _git("status", "--porcelain", "--", str(rel)):
            out.append(f"{rel} has uncommitted changes")
        ts = _git("log", "-1", "--format=%ct", "--", str(rel))
        if not ts:
            return out + [f"{rel} is not committed"]
        commit_ts = float(ts)
    exempt = set(pr.get("exempt_records", []))
    for p in sorted((arms_dir or ARMS_DIR).glob("ls_*.json")):
        if p.name in exempt or p.name.endswith(".smoke.json"):
            continue
        if p.stat().st_mtime <= commit_ts:
            out.append(f"record older than the pre-registration commit: {p.name}")
    return out


def _sum(pooled: dict, ids) -> float:
    return float(sum(f["return_pct"] or 0.0 for f in pooled["per_fold"] if f["fold_id"] in ids))


def judge_candidate(tag: str, audit: dict, inst: dict, sep: dict, surv: dict) -> dict:
    res = {"tag": tag, "S0a": NR, "S0": NR, "S1": NR, "survivor": NR}

    def done(decision, stage, why):
        res.update(decision=decision, stopped_at=stage, reason=why)
        return res

    a = audit["candidates"][tag]
    res["S0a"] = {"status": a["status"], "reason": a.get("reason")}
    if a["status"] not in ("PASS", "REFERENCE"):
        return done("INCONCLUSIVE", "S0a", a.get("reason") or "label audit failed")
    s0 = inst["candidates"][tag]
    res["S0"] = {"pass": s0["S0_pass"]}
    if not s0["S0_pass"]:
        return done("INCONCLUSIVE", "S0", "weights, keys or thresholds")
    s1 = sep["candidates"][tag]
    res["S1"] = s1
    if s1.get("status") == "INCONCLUSIVE":
        return done("INCONCLUSIVE", "S1", s1.get("reason"))
    if not s1["S1_pass"]:
        return done("FAIL", "S1", f"early gap {s1['gap']} not above both references "
                                  f"{sep['references']}")
    sv = surv["candidates"][tag]
    res["survivor"] = sv
    if not sv["survives"]:
        return done("FAIL", "survivor", sv["reason"])
    return done("SURVIVOR", "survivor", "S1 pass, within the top 3: backtested at S2")


def judge_s2(pr: dict, arm: str, comps: dict, inst_arm: dict | None, pool_fn,
             seed2_fn=None) -> dict:
    """comps: {"width": tag, "control": tag}. Early half only."""
    res = {"arm": arm, "S2": NR}
    if inst_arm is None:
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S2", "reason": "no S2 instrument"}
    if inst_arm.get("S0_inconclusive"):
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S2",
                      "reason": f"zero-trade folds {inst_arm['zero_trade_folds']}"}
    if not inst_arm["S0_pass"]:
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S2", "reason": "arm instrument failed"}
    early = set(pr["early"])
    ap = pool_fn(arm)
    cp = {c: pool_fn(t) for c, t in comps.items()}
    miss = {n: sorted(set(p["folds_missing"]) & early) for n, p in (("arm", ap), *cp.items())}
    if any(miss.values()):
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S2", "reason": f"folds missing {miss}"}
    e = {"arm": _sum(ap, early)} | {c: _sum(p, early) for c, p in cp.items()}
    m = {f"vs_{c}": e["arm"] - e[c] for c in comps}
    res["S2"] = {"early_pp": e, "margins": m, "pass": all(v > 0 for v in m.values())}
    if not res["S2"]["pass"]:
        return res | {"decision": "FAIL", "stopped_at": "S2", "reason": "early half did not beat both"}
    if min(m.values()) < float(pr["small_margin_pp"]):
        res["S2"]["small_margin"] = True
        if seed2_fn is None:
            return res | {"decision": "INCONCLUSIVE", "stopped_at": "S2",
                          "reason": f"margin under {pr['small_margin_pp']} pp: seed-2 repeat"}
        s2 = seed2_fn(arm)
        res["S2"]["seed2"] = s2
        if not all(v > 0 for v in s2["margins"].values()):
            return res | {"decision": "FAIL", "stopped_at": "S2", "reason": "seed-2 margin not > 0"}
    return res | {"decision": "S2_PASS", "stopped_at": "S2", "min_margin": min(m.values())}


def pick_winner(s2: dict) -> str | None:
    ok = [(r["min_margin"], a) for a, r in s2.items() if r.get("decision") == "S2_PASS"]
    return max(ok)[1] if ok else None


def judge_s3(pr: dict, arm: str, comps: dict, pool_fn, judge_one_fn) -> dict:
    late = set(pr["late"])
    jp = judge_one_fn(pr, arm, comps["control"])
    wp, ap = pool_fn(comps["width"]), pool_fn(arm)
    m = {"vs_width": _sum(ap, late) - _sum(wp, late), "vs_control": jp.get("late_gain_pp")}
    res = {"S3": {"judge_vs_control": jp, "late_margins": m}}
    if jp["verdict"] in ("INCONCLUSIVE", "MISSING"):
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S3", "reason": f"judge_one {jp['verdict']}"}
    if not (jp["verdict"] == "PASS" and m["vs_width"] > 0):
        return res | {"decision": "FAIL", "stopped_at": "S3",
                      "reason": "judge_one vs control not PASS or late pp <= width"}
    if min(v for v in m.values() if v is not None) < float(pr["small_margin_pp"]):
        return res | {"decision": "INCONCLUSIVE", "stopped_at": "S3",
                      "reason": f"margin under {pr['small_margin_pp']} pp: seed-2 repeat"}
    return res | {"decision": "PASS (PROVISIONAL)", "stopped_at": "S3",
                  "confirmation_route": pr["confirmation_route"]}


def judge_all(pr, audit, inst, sep, surv, pool_fn, judge_one_fn, seed2_fn=None) -> dict:
    out = {"label": pr["label"], "candidates": {}, "arms": {}, "s2_winner": None}
    comps = pr["comparators"]["ARO"]
    for c in C.CANDIDATES:
        if c.tag == "CONTROL":
            continue
        out["candidates"][c.tag] = judge_candidate(c.tag, audit, inst, sep, surv)
    s2 = {}
    for tag in surv["survivors"]:
        for role in C.BY_TAG[tag].roles:
            arm = C.plan_tag(C.BY_TAG[tag], role)
            s2[arm] = judge_s2(pr, arm, comps, inst.get("arms", {}).get(arm), pool_fn, seed2_fn)
            s2[arm] |= {"tag": tag, "role": role}
    out["arms"] = s2
    win = pick_winner(s2)
    if win is None:
        return out
    out["s2_winner"] = {"arm": win, "tag": s2[win]["tag"], "role": s2[win]["role"]}
    s3 = judge_s3(pr, win, comps, pool_fn, judge_one_fn)
    out["arms"][win] |= s3
    if s3["decision"] == "PASS (PROVISIONAL)":
        g = C.plan_tag(C.BY_TAG[s2[win]["tag"]], s2[win]["role"], "GBR")
        gc = pr["comparators"]["GBR"]
        r = judge_s2(pr, g, gc, inst.get("arms", {}).get(g), pool_fn)
        if r["decision"] == "S2_PASS":
            r |= judge_s3(pr, g, gc, pool_fn, judge_one_fn)
        out["gbr_repeat"] = r
    return out


def main(argv=None) -> int:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan, judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-repeat", type=int, default=None)
    a = ap.parse_args(argv)
    seed2_fn = None
    if a.seed_repeat:
        def seed2_fn(arm):
            from research.width_relative.judge import seed_plan

            early = set(pr["early"])
            comps = pr["comparators"]["ARO"]
            ap_ = pool(seed_plan(arm, a.seed_repeat))
            e = {"arm": _sum(ap_, early),
                 "width": _sum(pool(_plan(comps["width"])), early),
                 "control": _sum(pool(seed_plan(comps["control"], a.seed_repeat)), early)}
            return {"seed": a.seed_repeat, "early_pp": e,
                    "margins": {f"vs_{c}": e["arm"] - e[c] for c in ("width", "control")}}
    out = judge_all(pr, *(json.loads((HERE / n).read_text()) for n in
                          ("audit.json", "instrument.json", "separation.json", "survivors.json")),
                    lambda t: pool(_plan(t)), judge_one, seed2_fn)
    for t, r in out["candidates"].items():
        print(f"{t}: {r['decision']} ({r['stopped_at']}: {r['reason']})")
    for arm, r in out["arms"].items():
        print(f"{arm}: {r['decision']} ({r['stopped_at']}: {r.get('reason', '')})")
    print("S2 winner:", out["s2_winner"])
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
