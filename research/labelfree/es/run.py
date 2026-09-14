"""Per-fold driver: search on TRAIN, then evaluate the test range exactly ONCE (T035-T042).

THE PROPERTY THIS FILE EXISTS TO PROTECT. With thousands of evaluations per fold, a single
leak of the test range into the search invalidates everything. So:

  * every search evaluation is written with range="train" (FR-002)
  * the best candidate is chosen from TRAINING scores alone, and frozen (FR-012)
  * exactly one range="test" evaluation follows, after the search has concluded (FR-010)

and none of that is taken on trust -- history.py --audit re-derives all of it from the log
alone, without trusting this code (FR-011).

WRITE BEFORE SCORE. Each candidate is appended to the sidecar with score: null and only then
evaluated. A crash leaves evidence of an ATTEMPT, which is what the deflation must count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import cmaes as CM            # noqa: E402
from research.labelfree.es import evaluate as EV         # noqa: E402
from research.labelfree.es import history as H           # noqa: E402
from research.labelfree.es import panel as PA            # noqa: E402
from research.labelfree.es import space as SP            # noqa: E402

CALIBRATION = Path(__file__).resolve().parent / "calibration.json"


def require_calibration(allow_missing: bool = False) -> dict:
    """THE BLOCKING GATE (T026, SC-017).

    Until the surrogate's agreement with real backtests is measured and reported, no search
    result is admissible: an uncalibrated surrogate means the entire budget optimises a
    fiction. Every downstream entry point calls this.
    """
    if not CALIBRATION.exists():
        if allow_missing:
            return {"status": "NOT RUN", "bypassed": True}
        raise SystemExit(
            "BLOCKED: no calibration.json. The surrogate's rank agreement with real "
            "freqtrade backtests must be measured before any search result is admissible "
            "(SC-017).\n  python -m research.labelfree.es.calibrate --n 40 --folds f05..f14")
    c = json.loads(CALIBRATION.read_text())
    if not c.get("passed"):
        raise SystemExit(
            f"BLOCKED: calibration FAILED — {c.get('verdict', 'see calibration.json')}. "
            f"The search would optimise a fiction.")
    return c


def search_fold(fold_id: str, method: str, budget: int, seed: int,
                path: Path = H.SIDECAR) -> dict:
    """Search one fold's TRAINING range. Returns the frozen best candidate."""
    panel = PA.load(fold_id, train=True)
    inc_tpy, inc_exp = EV.incumbent_reference(panel)
    t0 = time.time()

    def evaluate(params: dict, generation: int) -> float:
        cid = SP.candidate_id(params)
        # Written BEFORE the score exists (FR-013). Numerically identical candidates share a
        # candidate_id but get their own eval_id: one evaluation for scoring, two for the
        # trial count, because the deflation counts attempts.
        eid = H.write(cid, params, fold_id, "train", method, generation, path=path)
        s = EV.score(panel, params, inc_tpy, inc_exp)
        H.complete(eid, {"score": None if s.value == float("-inf") else s.value,
                         **{k: v for k, v in s.as_row().items() if k != "value"}}, path=path)
        return s.value

    hist = CM.METHODS[method](evaluate, budget=budget, seed=seed)
    finite = [h for h in hist if h["value"] > float("-inf")]
    if not finite:
        return {"fold_id": fold_id, "method": method, "evaluations": len(hist),
                "best": None, "note": "no admissible candidate in the budget"}
    best = max(finite, key=lambda h: h["value"])
    return {"fold_id": fold_id, "method": method, "evaluations": len(hist),
            "admissible": len(finite), "best": best,
            "incumbent_trades_per_year": inc_tpy,
            "seconds": round(time.time() - t0, 1)}


def evaluate_test_once(fold_id: str, params: dict, method: str,
                       path: Path = H.SIDECAR) -> dict:
    """The single test-range evaluation (FR-010). Called after the search has concluded."""
    panel = PA.load(fold_id, train=False)
    inc_tpy, inc_exp = EV.incumbent_reference(panel)
    cid = SP.candidate_id(params)
    eid = H.write(cid, params, fold_id, "test", method, 0, path=path)
    s = EV.score(panel, params, inc_tpy, inc_exp)
    H.complete(eid, {"score": None if s.value == float("-inf") else s.value,
                     **{k: v for k, v in s.as_row().items() if k != "value"}}, path=path)
    return {"eval_id": eid, "candidate_id": cid, **s.as_row()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--method", required=True, choices=sorted(CM.METHODS))
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-test", action="store_true",
                    help="search only; leave the single test evaluation to the driver")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="DEVELOPMENT ONLY — results produced this way are inadmissible")
    a = ap.parse_args()

    cal = require_calibration(allow_missing=a.allow_uncalibrated)
    if cal.get("bypassed"):
        print("WARNING: running UNCALIBRATED. Results are inadmissible (SC-017).", flush=True)

    res = search_fold(a.fold, a.method, a.budget, a.seed)
    print(json.dumps({k: v for k, v in res.items() if k != "best"}, indent=2), flush=True)
    if res["best"]:
        print("best (train):", json.dumps(
            {"value": round(res["best"]["value"], 4),
             "params": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in res["best"]["params"].items()}}), flush=True)
        if not a.skip_test:
            t = evaluate_test_once(a.fold, res["best"]["params"], a.method)
            print("TEST (once):", json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in t.items() if k != "params"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
