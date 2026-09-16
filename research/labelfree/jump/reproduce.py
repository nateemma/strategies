"""T050 / SC-010 -- re-derive the verdict from saved artefacts in a fresh process.

Reads only `out/activity.json` (the measurements) and re-applies the gate rules from
`research/labelfree/jump/__init__.py`. It does NOT read `verdict.json`: reproducing a
conclusion by reading the conclusion proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.jump import (MIN_ACTIVE_FOLDS, NOVEL_BLOCK_FLOOR,   # noqa: E402
                                     OUT, SEED_AGREEMENT_FLOOR)


def derive(activity: dict) -> dict:
    ok = [f for f in activity["folds"] if f["status"] == "ok"]
    br = np.array([f["activity"]["block_rate"] for f in ok])
    nv = np.array([f["novel_block_rate"] for f in ok])
    ag = np.array([f["seed_agreement"] for f in ok])
    sp = np.array([f["across_seed_spread"] for f in ok])
    n_active = sum(1 for f in ok if f["activity"]["verdict"] == "active")

    gates = {
        "SC-020_active_folds": n_active >= MIN_ACTIVE_FOLDS,
        "SC-019_novel_block_rate": bool((nv >= NOVEL_BLOCK_FLOOR).sum() > len(nv) / 2),
        "FR-023_seed_agreement": bool(ag.mean() >= SEED_AGREEMENT_FLOOR),
        "SC-017_seed_spread_below_effect": bool(sp.mean() < br.mean() * 0.5),
    }
    return {"verdict": "KILL" if not all(gates.values()) else "PROCEED",
            "gates": gates,
            "n_active": n_active, "n_folds": len(ok),
            "novel_mean": float(nv.mean()), "seed_agreement_mean": float(ag.mean()),
            "seed_spread_mean": float(sp.mean()), "block_rate_mean": float(br.mean())}


def main() -> int:
    d = derive(json.loads((OUT / "activity.json").read_text()))
    print(json.dumps(d, indent=2))
    recorded = json.loads((OUT / "verdict.json").read_text())["verdict"]
    match = d["verdict"] == recorded
    print(f"\nrecorded {recorded} | re-derived {d['verdict']} -> "
          f"{'REPRODUCED' if match else 'MISMATCH'}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
