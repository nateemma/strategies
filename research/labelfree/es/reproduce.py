"""SC-005 and SC-006: the fee stress test, and reproduction from the saved configuration.

SC-006 is a SEPARATE PROCESS reproducing the accepted result from its saved configuration
alone. That means this module must read a config file and nothing else — no state carried
over from the run that produced it. If it needs anything the config does not hold, the result
is not reproducible and the gate has failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import evaluate as EV         # noqa: E402
from research.labelfree.es import panel as PA            # noqa: E402
from research.labelfree.es import space as SP            # noqa: E402

FEE_MULTIPLIER = 1.5          # SC-005


def at_fee_multiple(fold_id: str, params: dict, mult: float, train: bool = False) -> dict:
    """Re-score a candidate with every per-bar cost scaled by `mult`.

    The panel is copied, not mutated: the cached panel is shared with the caller, and scaling
    it in place would silently contaminate every later evaluation in the process.
    """
    panel = PA.load(fold_id, train=train)
    scaled = {**panel, "pairs": {p: {**d, "cost": d["cost"] * mult}
                                 for p, d in panel["pairs"].items()}}
    tpy, ex = EV.incumbent_reference(scaled)
    s = EV.score(scaled, params, tpy, ex)
    return {"fold_id": fold_id, "fee_multiplier": mult,
            "annualised_pp": round(s.unregularised, 4), "n_trades": s.n_trades,
            "trades_per_year": round(s.trades_per_year, 2),
            "mean_hold_h": round(s.mean_hold_h, 2),
            "mean_exposure": round(s.mean_exposure, 5),
            "admissible": s.admissible}


def fee_stress(fold_ids: list[str], params: dict) -> dict:
    """SC-005: the return delta must remain positive at 1.5x fees."""
    base = [at_fee_multiple(f, params, 1.0) for f in fold_ids]
    hi = [at_fee_multiple(f, params, FEE_MULTIPLIER) for f in fold_ids]
    mb = sum(r["annualised_pp"] for r in base) / max(1, len(base))
    mh = sum(r["annualised_pp"] for r in hi) / max(1, len(hi))
    inc_base = [at_fee_multiple(f, dict(SP.INCUMBENT), 1.0) for f in fold_ids]
    inc_hi = [at_fee_multiple(f, dict(SP.INCUMBENT), FEE_MULTIPLIER) for f in fold_ids]
    db = mb - sum(r["annualised_pp"] for r in inc_base) / max(1, len(inc_base))
    dh = mh - sum(r["annualised_pp"] for r in inc_hi) / max(1, len(inc_hi))
    return {"gate": "SC-005", "at_1x_pp": round(mb, 4), "at_1_5x_pp": round(mh, 4),
            "delta_at_1x_pp": round(db, 4), "delta_at_1_5x_pp": round(dh, 4),
            "passed": dh > 0, "per_fold": {"base": base, "stressed": hi}}


def reproduce(config_path: Path) -> dict:
    """SC-006. Everything needed must come from the file."""
    cfg = json.loads(Path(config_path).read_text())
    missing = [k for k in ("params", "folds", "annualised_pp") if k not in cfg]
    if missing:
        return {"gate": "SC-006", "passed": False,
                "reason": f"configuration lacks {missing}; it cannot be reproduced from disk"}
    got = [at_fee_multiple(f, cfg["params"], 1.0) for f in cfg["folds"]]
    mean = sum(r["annualised_pp"] for r in got) / max(1, len(got))
    ok = abs(mean - cfg["annualised_pp"]) < 1e-6
    return {"gate": "SC-006", "passed": ok, "recorded_pp": cfg["annualised_pp"],
            "reproduced_pp": round(mean, 6),
            "reason": None if ok else "reproduced value differs from the recorded one",
            "per_fold": got}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="saved configuration to reproduce (SC-006)")
    ap.add_argument("--fee-stress", action="store_true", help="SC-005")
    ap.add_argument("--folds", default="f05..f14")
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    if a.config:
        print(json.dumps(reproduce(Path(a.config)), indent=2))
    if a.fee_stress:
        params = (json.loads(Path(a.config).read_text())["params"] if a.config
                  else dict(SP.INCUMBENT))
        r = fee_stress(parse_span(a.folds), params)
        print(json.dumps({k: v for k, v in r.items() if k != "per_fold"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
