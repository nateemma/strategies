"""Stress a policy across a path bank (T029-T034).

A DISTRIBUTION, NEVER A POINT ESTIMATE. The whole reason this harness exists is that a single
backtest number cannot say how much of a record is orderings-of-luck. So the output is
quantiles, a drawdown distribution and the fraction of paths that lose money.

A VETO, NEVER A GATE (SC-009, FR-014). The verdict field is deliberately NOT called `passed`:
"NO VETO" means the harness found no reason to object, not that the policy is good. No policy
may be promoted on stress results alone.

INADMISSIBLE IS NOT FLAT (FR-013). A path on which the policy never trades is an unusable path,
not a zero return, and averaging zeros into the distribution would understate both the spread
and the loss fraction. The check routes through the study's SHARED admissibility gate with the
incumbent-relative exposure floor feature 013 added -- the absolute MIN_EXPOSURE marks
production itself degenerate on a 90-day window (measured 0.0196 on f00). That module exists
because this check kept being rewritten per site, each rewrite guarding only one side.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree import admissibility as AD               # noqa: E402
from research.labelfree.es import evaluate as EV                 # noqa: E402
from research.labelfree.es import space as SP                    # noqa: E402
from research.labelfree.stress import bank as BK                 # noqa: E402
from research.labelfree.stress import bootstrap as BS            # noqa: E402
from research.labelfree.stress import calibrate as CAL           # noqa: E402

PNL_Q = (5, 25, 50, 75, 95)
DD_Q = (50, 90, 95, 99)
EXPOSURE_FRACTION = 0.5          # of the incumbent's, on the same panel — feature 013's floor


def _policy_params(name: str) -> dict:
    if name == "incumbent":
        return dict(SP.INCUMBENT)
    cfg = json.loads(Path(name).read_text())
    return cfg["params"] if "params" in cfg else cfg


def stress(bank_path: Path, params: dict, n_paths: int | None = None) -> dict:
    prov, idx, panels = BK.load(Path(bank_path))
    ref = {f: EV.incumbent_reference(panels[f]) for f in prov["fold_ids"]}

    pnl, dd, inadmissible, reasons = [], [], 0, {}
    limit = n_paths or prov["n_paths"]
    t0 = time.time()
    for f in prov["fold_ids"]:
        tpy_ref, exp_ref = ref[f]
        for row in idx[f][:limit]:
            path = BS.resample_panel(panels[f], row)
            ts = EV.run(path, params)
            arm = AD.Arm("path", ts.annualised_pp, ts.mean_exposure,
                         len(ts) / path["span_years"] if path["span_years"] else 0.0,
                         exposure_floor=EXPOSURE_FRACTION * exp_ref)
            if arm.degenerate:
                inadmissible += 1
                r = (arm.reason or "degenerate").split(" below")[0]
                reasons[r] = reasons.get(r, 0) + 1
                continue
            pnl.append(ts.annualised_pp)
            dd.append(CAL.max_drawdown(np.asarray(ts.pnl, dtype=float)))

    n_used = len(pnl)
    total = limit * len(prov["fold_ids"])
    if n_used == 0:
        return {"verdict": "INADMISSIBLE BANK", "n_paths": total,
                "n_inadmissible": inadmissible, "inadmissible_reasons": reasons,
                "note": ("the policy never traded on any path; this is an unusable bank, not a "
                         "flat result (FR-013)")}

    a = np.asarray(pnl)
    loss_fraction = float((a < 0).mean())
    return {
        "bank": Path(bank_path).name,
        "block_length": prov["block_length"],
        "n_paths": total,
        "n_scored": n_used,
        "n_inadmissible": inadmissible,
        "inadmissible_reasons": reasons,
        "pnl_quantiles_pp_yr": {f"p{q}": round(float(np.percentile(a, q)), 4) for q in PNL_Q},
        "pnl_mean_pp_yr": round(float(a.mean()), 4),
        "drawdown_quantiles": {f"p{q}": round(float(np.percentile(dd, q)), 6) for q in DD_Q},
        "loss_path_fraction": round(loss_fraction, 4),
        "verdict": "VETO" if loss_fraction > 0.5 else "NO VETO",
        "verdict_meaning": (
            "A VETO, NEVER A GATE. 'NO VETO' means the harness found no reason to object — not "
            "that the policy is good. No policy may be promoted on stress results alone "
            "(SC-009, FR-014)."),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--policy", default="incumbent",
                    help="'incumbent' or a path to a saved config")
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "stress_report.json"))
    a = ap.parse_args()

    CAL.require_calibration()                    # BLOCKED without a passing calibration

    rep = {"policy": a.policy, **stress(Path(a.bank), _policy_params(a.policy), a.paths)}
    if a.policy != "incumbent":                  # FR-012: the incumbent beside any candidate
        rep["incumbent"] = stress(Path(a.bank), dict(SP.INCUMBENT), a.paths)
    Path(a.out).write_text(json.dumps(rep, indent=2))

    print(f"{rep['verdict']}   policy={a.policy}  L={rep.get('block_length')}  "
          f"({rep.get('seconds')}s)")
    if "pnl_quantiles_pp_yr" in rep:
        q = rep["pnl_quantiles_pp_yr"]
        print(f"  P&L pp/yr   p5 {q['p5']:+.2f}  p25 {q['p25']:+.2f}  p50 {q['p50']:+.2f}  "
              f"p75 {q['p75']:+.2f}  p95 {q['p95']:+.2f}")
        print(f"  loss-path fraction   {rep['loss_path_fraction']:.3f}")
        print(f"  paths {rep['n_scored']} scored / {rep['n_inadmissible']} inadmissible "
              f"of {rep['n_paths']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
