"""Judge the NNNC framework-switch experiment under its committed pre-registration.

Each switch (028's C2 ATR take-profit; 026's reversion regime gate) is declared on four NNNC
models running on their own borrowed per-fold weights. Per model the switch PASSes, FAILs or is
INCONCLUSIVE under `preregistration.json`'s rule; the switch is ADOPTed for NNNC only if it
passes on arm D (ARO) and on at least 2 of the 3 replications. Refuses (exit 2) if the
pre-registration has uncommitted changes, a class's declared value differs from the frozen one,
or an arm record predates the pre-registration commit.

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.nnnc_switches.judge
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def _plan(tag: str) -> dict:
    return json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())


def _rec_path(tag: str, fid: str) -> Path:
    return STRAT / "regime" / "arms" / f"{tag}_{fid}_{fid.upper()}.json"


def _recs(tag: str, folds: list[str]) -> dict:
    return {f: json.loads(_rec_path(tag, f).read_text()) for f in folds
            if _rec_path(tag, f).is_file()}


def _fill(s: str, m: str) -> str:
    return s.replace("{m}", m.lower()).replace("{M}", m)


def _declared(cls, key: str):
    head, _, tail = key.partition(".")
    v = getattr(cls, head, None)
    return v.get(tail) if tail and isinstance(v, dict) else v


def refusals(pr: dict) -> list[str]:
    from evaluation.fingerprint import load_strategy_class

    out = []
    rel = PREREG.relative_to(STRAT)
    if _git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = _git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        return out + [f"{rel} is not committed"]
    for m in pr["models"]:
        for sw in pr["switches"].values():
            cls = load_strategy_class(_fill(sw["class"], m))
            for k, v in sw["values"].items():
                if _declared(cls, k) != v:
                    out.append(f"{cls.__name__}.{k} = {_declared(cls, k)!r} != frozen {v!r}")
            tags = [_fill(sw["arm"], m)]
            if sw["control"] != "model":
                tags.append(_fill(sw["control"], m))
            elif m != pr["primary_model"]:
                tags.append(pr["models"][m]["control"])
            for t in tags:
                for f in pr["folds"]:
                    p = _rec_path(t, f)
                    if p.is_file() and p.stat().st_mtime <= float(ts):
                        out.append(f"record older than the pre-registration commit: {p.name}")
    return sorted(set(out))


def _sum(pooled: dict, ids) -> float:
    return float(sum(f["return_pct"] or 0.0 for f in pooled["per_fold"] if f["fold_id"] in ids))


def judge_one(pr: dict, arm_tag: str, ctl_tag: str) -> dict:
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from evaluation.halves import split
    from evaluation.pool import pool

    rule = pr["rule"]
    ap, cp = pool(_plan(arm_tag)), pool(_plan(ctl_tag))
    miss = ap["folds_missing"] + cp["folds_missing"]
    if miss:
        return {"verdict": "MISSING", "missing": miss}
    early, late = set(pr["early"]), set(pr["late"])
    ge = _sum(ap, early) - _sum(cp, early)
    gl = _sum(ap, late) - _sum(cp, late)
    cmp = compare_data(ap, cp)
    rk = risk.compare(ap["trades"], cp["trades"])
    ra, rc = _recs(arm_tag, pr["folds"]), _recs(ctl_tag, pr["folds"])
    changed = sum(ra[f]["trades_hash"] != rc[f]["trades_hash"] for f in pr["folds"])
    p = rule["paired_map"][cmp["verdict"]]
    r = [rule["risk_map"][rk["verdict_sharpe"]], rule["risk_map"][rk["verdict_maxdd"]]]
    failed = []
    if cp["n_trades_pooled"] < pr["min_control_trades"] or changed == 0 \
            or p == "inconclusive" or "inconclusive" in r:
        verdict = "INCONCLUSIVE"
    else:
        if not ge > 0:
            failed.append("early gain")
        if not gl > 0:
            failed.append("late gain")
        if p == "fail":
            failed.append("paired")
        if "fail" in r:
            failed.append("risk")
        verdict = "FAIL" if failed else "PASS"
    halves = split(ap, cp)
    ex_a, ex_c = Counter(), Counter()
    for f in pr["folds"]:
        ex_a.update(ra[f].get("exit_reason_counts") or {})
        ex_c.update(rc[f].get("exit_reason_counts") or {})
    return {
        "arm": arm_tag, "control": ctl_tag, "verdict": verdict, "failed": failed,
        "arm_trades": ap["n_trades_pooled"], "ctl_trades": cp["n_trades_pooled"],
        "arm_pp": _sum(ap, early | late), "ctl_pp": _sum(cp, early | late),
        "early_gain_pp": ge, "late_gain_pp": gl, "folds_changed": changed,
        "paired": {"verdict": cmp["verdict"], "delta_pp": cmp.get("delta_pp"),
                   "ci_pp_yr": cmp.get("ci_pp_yr"), "p": (cmp.get("bootstrap") or {}).get("p")},
        "risk": rk,
        "halves": {h: {"verdict": halves[h]["verdict"], "p": halves[h]["p"],
                       "risk": [halves[h]["risk"]["verdict_sharpe"],
                                halves[h]["risk"]["verdict_maxdd"]]} for h in ("early", "late")},
        "exits_arm": dict(ex_a), "exits_ctl": dict(ex_c),
    }


def main() -> int:
    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    res = {"label": pr["label"], "switches": {}, "startup_effect": {}, "control_checks": {}}
    for name, sw in pr["switches"].items():
        rows = {}
        for m, md in pr["models"].items():
            ctl = md["control"] if sw["control"] == "model" else _fill(sw["control"], m)
            rows[m] = judge_one(pr, _fill(sw["arm"], m), ctl)
        prim = rows[pr["primary_model"]]["verdict"] == "PASS"
        reps = sum(rows[m]["verdict"] == "PASS" for m in rows if m != pr["primary_model"])
        res["switches"][name] = {"models": rows, "replications_passed": reps,
                                 "decision": "ADOPT" if prim and reps >= pr["replications_needed"]
                                 else "NOT ADOPTED"}
    for m, md in pr["models"].items():
        res["startup_effect"][m] = judge_one(pr, f"sw_{m.lower()}_gateoff", md["control"])
        if md.get("record"):
            a, b = _recs(md["control"], pr["folds"]), _recs(md["record"], pr["folds"])
            res["control_checks"][m] = {
                "record": md["record"],
                "hash_equal": sum(a[f]["trades_hash"] == b[f]["trades_hash"]
                                  for f in pr["folds"] if f in a and f in b),
                "trades_now": sum(r["n_trades"] for r in a.values()),
                "trades_record": sum(r["n_trades"] for r in b.values())}
    (HERE / "verdicts.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (HERE / "verdicts.md").write_text(render(res))
    print(render(res))
    return 0


def _f(x):
    return "—" if x is None else f"{x:+.2f}"


def render(res: dict) -> str:
    L = ["# NNNC switch experiment — verdicts", "", f"*{res['label']}*", ""]
    for name, s in res["switches"].items():
        L += [f"## {name}: **{s['decision']}** (arm D + {s['replications_passed']}/3 replications)",
              "", "| model | trades arm / ctl | Σ pp arm / ctl | early gain | late gain | paired (p) "
              "| risk Sharpe / maxDD | folds changed | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for m, r in s["models"].items():
            if r["verdict"] == "MISSING":
                L.append(f"| {m} | missing {r['missing']} | | | | | | | MISSING |")
                continue
            p = r["paired"]
            pv = "—" if p["p"] is None else f"{p['p']:.3f}"
            L.append(f"| {m} | {r['arm_trades']} / {r['ctl_trades']} | {r['arm_pp']:+.2f} / "
                     f"{r['ctl_pp']:+.2f} | {_f(r['early_gain_pp'])} | {_f(r['late_gain_pp'])} | "
                     f"{p['verdict']} ({pv}) | "
                     f"{r['risk']['verdict_sharpe']} / {r['risk']['verdict_maxdd']} | "
                     f"{r['folds_changed']}/15 | **{r['verdict']}**"
                     f"{' (' + ', '.join(r['failed']) + ')' if r['failed'] else ''} |")
        L += ["", "Per half (reported only):", ""]
        for m, r in s["models"].items():
            if r["verdict"] != "MISSING":
                h = r["halves"]
                L.append(f"- {m}: early {h['early']['verdict']}, risk {'/'.join(h['early']['risk'])}; "
                         f"late {h['late']['verdict']}, risk {'/'.join(h['late']['risk'])}; "
                         f"exits {r['exits_arm']} vs {r['exits_ctl']}")
        L.append("")
    L += ["## Reported only: the 4096-bar startup alone (GateOff vs model control)", "",
          "| model | Σ pp gateoff / ctl | early / late gain | folds changed | paired |", "|---|---|---|---|---|"]
    for m, r in res["startup_effect"].items():
        if r["verdict"] != "MISSING":
            L.append(f"| {m} | {r['arm_pp']:+.2f} / {r['ctl_pp']:+.2f} | {_f(r['early_gain_pp'])} / "
                     f"{_f(r['late_gain_pp'])} | {r['folds_changed']}/15 | {r['paired']['verdict']} |")
    L += ["", "## Reported only: same-day native controls vs their earlier records", "",
          "| model | record | trades_hash equal | trades now / record |", "|---|---|---|---|"]
    for m, c in res["control_checks"].items():
        L.append(f"| {m} | `{c['record']}` | {c['hash_equal']}/15 | {c['trades_now']} / {c['trades_record']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
