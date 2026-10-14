"""033 T024: judge the width-relative sizer arms under the committed pre-registration, stage by
stage, per model and per arm (preregistration.json).

  exit_off (`wr_<m>_sz`):  S0 (instrument.json) -> S1 (separation.json, early only) ->
                           S2 early pp > width book AND > control ->
                           S3 judge_one(arm, control) PASS AND late pp > width book.
  exit_on  (`wr_<m>_szx`): S0 -> S2 early pp > exit_off arm AND > width book ->
                           S3 judge_one(exit_on, exit_off) PASS AND late pp > width book.

A failed stage stops THAT arm: its later stages are `not_reached` and are not judged. The exit-off
arm's records stay readable as the exit-on arm's comparator (pre-registration, analyze I4). A stage
passing by under `small_margin_pp` is INCONCLUSIVE until a paired seed-2 repeat (T025a). Thirds
are context, computed only when S3 is reached. Outcomes: PASS (PROVISIONAL) | FAIL | INCONCLUSIVE --
never a finding (constitution II).

Refuses (exit 2) if the pre-registration is uncommitted or dirty, or any `wr_*` arm record predates
its latest commit.

    PYTHONPATH=. ../../.venv/bin/python -m research.width_relative.judge
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
ARMS_DIR = STRAT / "regime" / "arms"
NR = "not_reached"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def _rec_path(tag: str, fid: str) -> Path:
    return ARMS_DIR / f"{tag}_{fid}_{fid.upper()}.json"


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
    for b in pr["books"].values():
        for tag in (b["exit_off"], b["exit_on"]):
            for f in pr["folds"]:
                p = _rec_path(tag, f)
                if (tag, f) in exempt or not p.is_file():
                    continue
                if p.stat().st_mtime <= commit_ts:
                    out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def _sum(pooled: dict, ids) -> float:
    return float(sum(f["return_pct"] or 0.0 for f in pooled["per_fold"] if f["fold_id"] in ids))


def judge_arm(pr: dict, m: str, arm: str, inst: dict, sep: dict, pool_fn, judge_one_fn) -> dict:
    """One model-arm's staged verdict. `pool_fn(tag)` and `judge_one_fn(pr, a, c)` are called
    only for stages that are reached."""
    b = pr["books"][m]
    small = float(pr["small_margin_pp"])
    tag = b[arm]
    comps = ("width", "control") if arm == "exit_off" else ("exit_off", "width")
    comp_tag = {"width": b["width"], "control": b["control"], "exit_off": b["exit_off"]}
    res = {"arm": tag, "S0": NR, "S1": NR if arm == "exit_off" else "n/a", "S2": NR, "S3": NR}

    def done(decision: str, stage: str, why: str) -> dict:
        res.update(decision=decision, stopped_at=stage, reason=why)
        if decision == "PASS (PROVISIONAL)":
            res["confirmation_route"] = pr["confirmation_route"]
        return res

    s0 = inst["models"][m]["arms"][arm]
    res["S0"] = s0 | {"label_pass": inst["models"][m]["label_pass"]}
    if s0.get("S0_inconclusive"):
        return done("INCONCLUSIVE", "S0", f"zero-trade folds {s0['zero_trade_folds']}")
    if not s0["S0_pass"]:
        return done("INCONCLUSIVE", "S0", "instrument or label gate failed")

    if arm == "exit_off":
        s1 = sep["models"][m]
        res["S1"] = {"early": s1.get("early"), "pass": bool(s1["S1_pass"]),
                     "current_gap_early_preregistered": s1["current_gap_early_preregistered"]}
        if not s1["S1_pass"]:
            return done("FAIL", "S1", "width-relative separation inside the width book is not "
                                      "above the current model's (early half)")

    ap = pool_fn(tag)
    cp = {c: pool_fn(comp_tag[c]) for c in comps}
    for name, p in (("arm", ap), *cp.items()):
        if p["folds_missing"]:
            return done("INCONCLUSIVE", "S2", f"{name} folds missing: {p['folds_missing']}")
    early, late = set(pr["early"]), set(pr["late"])
    e = {"arm": _sum(ap, early)} | {c: _sum(p, early) for c, p in cp.items()}
    m2 = {f"vs_{c}": e["arm"] - e[c] for c in comps}
    res["S2"] = {"early_pp": e, "margins": m2, "pass": all(v > 0 for v in m2.values())}
    if not res["S2"]["pass"]:
        return done("FAIL", "S2", f"early half did not beat both {comps}")
    if min(m2.values()) < small:
        res["S2"]["small_margin"] = True
        return done("INCONCLUSIVE", "S2", f"margin under {small} pp: paired seed-2 repeat (T025a)")

    primary = "control" if arm == "exit_off" else "exit_off"
    jp = judge_one_fn(pr, tag, comp_tag[primary])
    jw = judge_one_fn(pr, tag, b["width"])
    lt = {"arm": _sum(ap, late)} | {c: _sum(p, late) for c, p in cp.items()}
    m3 = {"vs_width": lt["arm"] - lt["width"], f"vs_{primary}": jp.get("late_gain_pp")}
    res["S3"] = {"late_pp": lt, "margins": m3, f"judge_vs_{primary}": jp,
                 "judge_vs_width_reported": jw}
    res["thirds"] = {t: {"arm": _sum(ap, set(ids))} | {c: _sum(p, set(ids)) for c, p in cp.items()}
                     for t, ids in pr["thirds"].items()}
    if jp["verdict"] in ("INCONCLUSIVE", "MISSING"):
        res["S3"]["pass"] = None
        return done("INCONCLUSIVE", "S3", f"judge_one vs {primary} {jp['verdict']}")
    res["S3"]["pass"] = jp["verdict"] == "PASS" and m3["vs_width"] > 0
    if not res["S3"]["pass"]:
        return done("FAIL", "S3", f"late half: judge_one vs {primary} not PASS or late pp <= width")
    if min(m3["vs_width"], m3[f"vs_{primary}"]) < small:
        res["S3"]["small_margin"] = True
        return done("INCONCLUSIVE", "S3", f"margin under {small} pp: paired seed-2 repeat (T025a)")
    return done("PASS (PROVISIONAL)", "S3", "all stages passed; provisional, not a finding")


def main(argv=None) -> int:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan, judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = json.loads((HERE / "instrument.json").read_text())
    sep = json.loads((HERE / "separation.json").read_text())
    out = {"label": pr["label"], "models": {}}
    for m in pr["books"]:
        out["models"][m] = {}
        for arm in ("exit_off", "exit_on"):
            r = judge_arm(pr, m, arm, inst, sep, lambda t: pool(_plan(t)), judge_one)
            out["models"][m][arm] = r
            print(f"{m} {arm}: {r['decision']} (stopped at {r['stopped_at']}: {r['reason']})",
                  flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
