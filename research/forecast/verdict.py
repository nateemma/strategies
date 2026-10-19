"""038 verdict per (method, N) under the committed pre-registration (research R9; FR-009, FR-010).

Refuses unless research/forecast/preregistration.json is committed, clean and older than every result
it reads, and unless the instrument checks pass. Seed-2 reruns (`<method>_s2`) inform condition (e)
and get no verdict of their own.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.verdict
"""
from __future__ import annotations

import json

from research.forecast import common as C

PASS = "PASS (PROVISIONAL)"


def _meets(h: dict, pr: dict) -> bool:
    """Conditions (a)-(d) on one method's {half: cell}."""
    key = f"mean_net_{float(pr['costs']['verdict_bps'])}"
    e, l = h["early"], h["late"]
    if min(e["n_trades"], l["n_trades"]) < pr["min_trades_per_half"]:
        return False
    if not all((c[key] or 0) > 0 and (c["edge_vs_rand"] or 0) > 0 for c in (e, l)):
        return False
    return (l["net_lb95"] is not None) and l["net_lb95"] > 0


def decide(h: dict, pr: dict, learned: bool, seed2: dict | None = None) -> tuple[str, list[str]]:
    key = f"mean_net_{float(pr['costs']['verdict_bps'])}"
    fail = max(h["early"].get("fail_share", 0.0), h["late"].get("fail_share", 0.0))
    if fail > pr["max_fail_share"]:
        return "INCONCLUSIVE", [f"fail_share {fail:.3f} > {pr['max_fail_share']}"]
    reasons = []
    for half in ("early", "late"):
        c = h[half]
        if c["n_trades"] > 0 and c[key] is not None and c[key] <= 0:
            reasons.append(f"{half} mean_net {c[key]:+.5f} <= 0")
        if c["n_trades"] > 0 and c["edge_vs_rand"] is not None and c["edge_vs_rand"] <= 0:
            reasons.append(f"{half} edge_vs_rand {c['edge_vs_rand']:+.5f} <= 0")
    if reasons:
        return "FAIL", reasons
    if not _meets(h, pr):
        n = (h["early"]["n_trades"], h["late"]["n_trades"])
        why = (f"trades {n} < {pr['min_trades_per_half']}" if min(n) < pr["min_trades_per_half"]
               else f"late net_lb95 {h['late']['net_lb95']} not > 0")
        return "INCONCLUSIVE", [why]
    if learned:
        if seed2 is None:
            return "INCONCLUSIVE", ["seed-2 rerun missing (condition e)"]
        if not _meets(seed2, pr):
            return "INCONCLUSIVE", ["seed-2 rerun does not meet (a)-(d) (condition e)"]
    return PASS, ["(a)-(d) met" + (" and seed 2 (e)" if learned else "")]


def _learned(method: str, pr: dict) -> bool:
    learned = method.startswith(("lgb_", "ridge_"))
    return pr["methods"].get(method, {}).get("tier", 2 if learned else 0) >= 2


def _deterministic(method: str) -> bool:
    """Ridge has no seed: a seed-2 rerun is identical, so the seed-1 run meets condition (e)
    (preregistration_ridge.json)."""
    return method.startswith("ridge_")


def judge(scores: dict, pr: dict) -> dict:
    cells = {}
    for m, s in scores.items():
        if m.endswith("_s2"):
            continue
        s2 = scores.get(f"{m}_s2") or (s if _deterministic(m) else None)
        cells[m] = {}
        for n, h in s["n"].items():
            # vote_B has a seeded member (lgb_L): its PASS needs the vote rebuilt on lgb_L_s2 (vote amendment)
            v, why = decide(h, pr, _learned(m, pr) or m == "vote_B", s2["n"].get(n) if s2 else None)
            cells[m][n] = {"verdict": v, "reasons": why, "numbers": {k: _brief(h[k], pr) for k in ("early", "late")}}
    nulls = [m for m in cells if m.startswith("lgb_null")]  # lgb_null (1-96) and lgb_null_long (2-4 days)
    null_passes = [n for m in nulls for n, c in cells[m].items() if c["verdict"] == PASS]
    if null_passes:
        for m, cs in cells.items():
            if _learned(m, pr) and m not in nulls:
                for c in cs.values():
                    if c["verdict"] == PASS:
                        c["verdict"], c["reasons"] = "INCONCLUSIVE", c["reasons"] + [f"null passes at N={null_passes}"]
    key = f"mean_net_{float(pr['costs']['verdict_bps'])}"
    winners = [(scores[m]["n"][n]["late"].get("net_per_pair_year") or float("-inf"), int(n), m)
               for m, cs in cells.items() if not m.startswith("lgb_null") for n, c in cs.items() if c["verdict"] == PASS]
    best = max(winners) if winners else None
    be = sorted(int(n) for m, s in scores.items() if not m.startswith("lgb_null") and not m.endswith("_s2")
                for n, h in s["n"].items()
                if all((h[x][key] or 0) > 0 and h[x]["n_trades"] > 0 for x in ("early", "late")))
    return {"cells": cells, "null_passes": null_passes,
            "most_viable_n": best[1] if best else None, "most_viable_method": best[2] if best else None,
            "break_even_n": be[0] if be else None}


def _brief(c: dict, pr: dict) -> dict:
    key = f"mean_net_{float(pr['costs']['verdict_bps'])}"
    return {k: c.get(k) for k in ("n_trades", key, "edge_vs_rand", "net_lb95", "net_per_pair_year", "fail_share")}


def main(argv=None) -> int:
    scores_p, inst_p = C.HERE / "scores.json", C.HERE / "instrument.json"
    rc = C.refuse_unless_registered([scores_p, inst_p, *C.PRED.glob("*.feather")])
    for m, amend in C.POST_HOC.items():
        f = C.PRED / f"{m}.feather"
        if not rc and f.exists():
            rc = C.refuse_unless_registered([f], prereg=amend)
    if rc:
        return rc
    inst = json.loads(inst_p.read_text()) if inst_p.exists() else {}
    bad = [k for k, v in inst.items() if isinstance(v, dict) and v.get("pass") is False]
    if not inst or bad:
        print(f"REFUSED: instrument checks {'missing' if not inst else 'failed: ' + ', '.join(bad)}")
        return 2
    pr = C.prereg()
    out = judge(json.loads(scores_p.read_text()), pr)
    out["post_hoc_methods"] = [m for m in out["cells"] if m in C.POST_HOC]
    C.write_json(C.HERE / "verdicts.json", out)
    for m, cs in out["cells"].items():
        print(f"{m:12s} " + " ".join(f"{n}:{c['verdict'][:4]}" for n, c in cs.items()))
    print({k: out[k] for k in ("null_passes", "most_viable_n", "most_viable_method", "break_even_n")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
