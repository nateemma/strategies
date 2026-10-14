"""032 T020: judge the candidate-only arms under the committed pre-registration, stage by stage.

Per model, in order (preregistration.json): instrument gates (simulator fidelity >= 0.95, zero
entry violations, trades in every fold) -> S1 within-candidate AUC (auc.json) -> S2 early half
(arm vs width book AND control) -> S3 late half (judge_one vs control PASS, AND late pp vs width
book; judge_one vs width reported). A stage that fails stops the model: later stages are
`not_reached` and their records are NOT read. A stage passing by under `small_margin_pp` is
INCONCLUSIVE until the paired seed-2 repeat (T021a). Outcomes: PASS (PROVISIONAL) | FAIL |
INCONCLUSIVE -- a pass is a provisional result, never a finding (constitution II).

Refuses (exit 2) if the pre-registration is uncommitted or dirty, or if any arm record predates its
commit.

    PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.judge
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
        for f in pr["folds"]:
            p = _rec_path(b["arm"], f)
            if (b["arm"], f) in exempt or not p.is_file():
                continue
            if p.stat().st_mtime <= commit_ts:
                out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def _sum(pooled: dict, ids) -> float:
    return float(sum(f["return_pct"] or 0.0 for f in pooled["per_fold"] if f["fold_id"] in ids))


def judge_model(pr: dict, m: str, auc: dict, inst: dict, pool_fn, judge_one_fn) -> dict:
    """One model's staged verdict. `pool_fn(tag)` and `judge_one_fn(pr, arm, ctl)` are only
    called for stages that are reached."""
    b = pr["books"][m]
    small = float(pr["small_margin_pp"])
    res = {"book": m, "S1": NR, "S2": NR, "S3": NR}

    def done(decision: str, stage: str, why: str) -> dict:
        res.update(decision=decision, stopped_at=stage, reason=why)
        if decision == "PASS (PROVISIONAL)":
            res["confirmation_route"] = pr["confirmation_route"]
        return res

    ins = inst["models"][f"CO_{m}"]
    res["instrument"] = {k: ins[k] for k in ("fidelity", "fidelity_pass", "entry_violations",
                                             "sell_rows_kept", "trades_per_fold")}
    if not ins["fidelity_pass"] or ins["entry_violations"] or ins["sell_rows_kept"]:
        return done("INCONCLUSIVE", "instrument", "instrument gate failed")
    if any(v == 0 for v in ins["trades_per_fold"].values()):
        return done("INCONCLUSIVE", "instrument", "a fold has zero trades")

    a = auc["models"][m]
    res["S1"] = {h: a[h] for h in ("early", "late")} | {"pass": bool(a["S1_pass"])}
    if not a["S1_pass"]:
        return done("FAIL", "S1", "within-candidate AUC did not improve in both halves")

    ap, cp, wp = pool_fn(b["arm"]), pool_fn(b["control"]), pool_fn(b["width"])
    for name, p in (("arm", ap), ("control", cp), ("width", wp)):
        if p["folds_missing"]:
            return done("INCONCLUSIVE", "S2", f"{name} folds missing: {p['folds_missing']}")
    early, late = set(pr["early"]), set(pr["late"])
    e = {k: _sum(p, early) for k, p in (("arm", ap), ("control", cp), ("width", wp))}
    m2 = {"vs_width": e["arm"] - e["width"], "vs_control": e["arm"] - e["control"]}
    res["S2"] = {"early_pp": e, "margins": m2, "pass": all(v > 0 for v in m2.values())}
    res["thirds"] = {t: {k: _sum(p, set(ids)) for k, p in (("arm", ap), ("control", cp),
                                                          ("width", wp))}
                     for t, ids in pr["thirds"].items()}
    if not res["S2"]["pass"]:
        return done("FAIL", "S2", "early half did not beat both the width book and the control")
    if min(m2.values()) < small:
        res["S2"]["small_margin"] = True
        return done("INCONCLUSIVE", "S2", f"margin under {small} pp: paired seed-2 repeat (T021a)")

    jc = judge_one_fn(pr, b["arm"], b["control"])
    jw = judge_one_fn(pr, b["arm"], b["width"])
    lt = {k: _sum(p, late) for k, p in (("arm", ap), ("control", cp), ("width", wp))}
    m3 = {"vs_width": lt["arm"] - lt["width"], "vs_control": jc.get("late_gain_pp")}
    res["S3"] = {"late_pp": lt, "margins": m3, "judge_vs_control": jc,
                 "judge_vs_width_reported": jw}
    if jc["verdict"] in ("INCONCLUSIVE", "MISSING"):
        res["S3"]["pass"] = None
        return done("INCONCLUSIVE", "S3", f"judge_one vs control {jc['verdict']}")
    res["S3"]["pass"] = jc["verdict"] == "PASS" and m3["vs_width"] > 0
    if not res["S3"]["pass"]:
        return done("FAIL", "S3", "late half: judge_one vs control not PASS or late pp <= width")
    if min(m3["vs_width"], m3["vs_control"]) < small:
        res["S3"]["small_margin"] = True
        return done("INCONCLUSIVE", "S3", f"margin under {small} pp: paired seed-2 repeat (T021a)")
    return done("PASS (PROVISIONAL)", "S3", "all stages passed; provisional, not a finding")


def main(argv=None) -> int:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan, judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    auc = json.loads((HERE / "auc.json").read_text())
    inst = json.loads((HERE / "instrument.json").read_text())
    out = {"label": pr["label"], "models": {}}
    for m in pr["books"]:
        r = judge_model(pr, m, auc, inst, lambda t: pool(_plan(t)), judge_one)
        out["models"][m] = r
        print(f"{m}: {r['decision']} (stopped at {r['stopped_at']}: {r['reason']})", flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
