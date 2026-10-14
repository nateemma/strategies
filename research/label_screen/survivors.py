"""034 T011: apply the pre-registered survivor rule (FR-011) mechanically.

A candidate survives S1 if its S0a status is PASS (or it is the reused GFM), its S0 passes, and its
S1 passes (early gap above BOTH references with at most 2 degenerate folds). Survivors are the top
MAX_SURVIVORS by gap, ties broken by AUC. Every candidate's reason is written to survivors.json,
which is committed before any survivor backtest.

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.survivors
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAX_SURVIVORS = 3


def select(audit: dict, instrument: dict, separation: dict) -> dict:
    out, eligible = {}, []
    for tag, s1 in separation["candidates"].items():
        a = audit["candidates"].get(tag, {}).get("status")
        s0 = instrument.get("candidates", {}).get(tag, {}).get("S0_pass")
        if a not in ("PASS", "REFERENCE"):
            out[tag] = {"survives": False, "reason": f"S0a {a}"}
        elif not s0:
            out[tag] = {"survives": False, "reason": "S0 not passed"}
        elif s1.get("status") == "INCONCLUSIVE":
            out[tag] = {"survives": False, "reason": f"S1 INCONCLUSIVE: {s1.get('reason')}"}
        elif not s1.get("S1_pass"):
            out[tag] = {"survives": False,
                        "reason": f"S1 gap {s1.get('gap')} not above both references "
                                  f"{separation['references']} (degenerate {s1.get('degenerate_folds')})"}
        else:
            eligible.append((s1["gap"], s1.get("auc") or 0.0, tag))
    ranked = sorted(eligible, reverse=True)
    for i, (gap, a, tag) in enumerate(ranked):
        ok = i < MAX_SURVIVORS
        out[tag] = {"survives": ok, "rank": i + 1, "gap": gap, "auc": a,
                    "reason": "S1 pass" + ("" if ok else f"; outside the top {MAX_SURVIVORS}")}
    return {"label": "034 survivors (FR-011)", "references": separation["references"],
            "survivors": [t for _, _, t in ranked[:MAX_SURVIVORS]], "candidates": out}


def main() -> int:
    r = select(*(json.loads((HERE / n).read_text())
                 for n in ("audit.json", "instrument.json", "separation.json")))
    (HERE / "survivors.json").write_text(json.dumps(r, indent=1))
    print("survivors:", r["survivors"])
    for t, v in r["candidates"].items():
        print(f"  {t}: {v['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
