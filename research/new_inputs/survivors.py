"""035 T021: the S1 survivor rule (FR-010), applied mechanically to screen.json.

An input survives only if: coverage >= MIN_COVERAGE (below it: INCONCLUSIVE -- not tested);
|rho with bb_width| <= MAX_RHO_WIDTH (orthogonality); |pooled partial| >= MIN_PARTIAL; the same
sign in >= MIN_SIGN_FOLDS of 8 early folds. Survivors: the top MAX_SURVIVORS by |partial|.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.survivors
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def select(screen: dict, rule: dict) -> dict:
    out, ranked = {}, []
    for tag, r in screen["inputs"].items():
        if r.get("status") == "INCONCLUSIVE":
            out[tag] = {"status": "INCONCLUSIVE", "reason": r.get("reason", "")}
            continue
        if r.get("coverage", 0.0) < rule["min_coverage"] or r.get("partial") is None:
            out[tag] = {"status": "INCONCLUSIVE",
                        "reason": f"coverage {r.get('coverage')} < {rule['min_coverage']}"}
            continue
        why = []
        if abs(r["rho_width"]) > rule["max_rho_width"]:
            why.append(f"|rho_width| {abs(r['rho_width']):.3f} > {rule['max_rho_width']}")
        if abs(r["partial"]) < rule["min_partial"]:
            why.append(f"|partial| {abs(r['partial']):.3f} < {rule['min_partial']}")
        if r["sign_folds"] < rule["min_sign_folds"]:
            why.append(f"sign in {r['sign_folds']} folds < {rule['min_sign_folds']}")
        if why:
            out[tag] = {"status": "FAIL", "reason": "; ".join(why)}
        else:
            ranked.append((abs(r["partial"]), tag))
    ranked.sort(reverse=True)
    surv = [t for _, t in ranked[:rule["max_survivors"]]]
    for a, t in ranked:
        out[t] = {"status": "PASS" if t in surv else "FAIL",
                  "reason": f"|partial| {a:.3f}" + ("" if t in surv else
                                                     f"; outside the top {rule['max_survivors']}")}
    return {"label": "035 survivors (FR-010)", "rule": rule, "survivors": surv, "inputs": out}


def main() -> int:
    rule = json.loads((HERE / "preregistration.json").read_text())["stages"]["S1"]["rule"]
    res = select(json.loads((HERE / "screen.json").read_text()), rule)
    (HERE / "survivors.json").write_text(json.dumps(res, indent=1))
    print("survivors:", res["survivors"])
    for t, r in res["inputs"].items():
        print(f"  {t}: {r['status']} ({r['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
