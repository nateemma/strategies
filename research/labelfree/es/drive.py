"""Sequential driver for the full 15-fold search and control (T039).

SEQUENTIAL BY CONSTRUCTION. Folds run one at a time and the two methods run one after the
other, because a concurrent freqtrade backtest would race on `.last_result.json` and because
the study's own convention is that everything runs sequentially.

The two methods are given the SAME budget and the SAME seed, and the counts are asserted equal
from the sidecar rather than from a counter this file keeps.
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

from research.labelfree.es import history as H           # noqa: E402
from research.labelfree.es import report as RP           # noqa: E402
from research.labelfree.es import run as RUN             # noqa: E402
from research.labelfree.es import space as SP            # noqa: E402
from research.labelfree.es import evaluate as EV         # noqa: E402
from research.labelfree.es import panel as PA            # noqa: E402
from research.labelfree.es import verdict as VD          # noqa: E402

STATE = Path(__file__).resolve().parent / "state.json"


def drive(fold_ids: list[str], budget: int, seed: int, allow_uncal: bool = False) -> dict:
    RUN.require_calibration(allow_missing=allow_uncal)
    t0 = time.time()
    per_fold, overfit_rows, counts = [], [], {}

    for fid in fold_ids:
        best_by_method = {}
        for method in ("cmaes", "random"):
            r = RUN.search_fold(fid, method, budget, seed)
            counts.setdefault(fid, {})[method] = r["evaluations"]
            best_by_method[method] = r
            print(f"  {fid}/{method}: {r['evaluations']} evals, "
                  f"best train {r['best']['value']:.3f} "
                  f"({r.get('seconds')}s)" if r["best"] else f"  {fid}/{method}: none",
                  flush=True)

        # FR-007: equality asserted, not assumed.
        a, b = counts[fid]["cmaes"], counts[fid]["random"]
        if a != b:
            raise SystemExit(f"{fid}: budgets differ — cmaes {a} vs random {b}")

        # FR-012: the winner is chosen from TRAINING scores alone, then frozen.
        winner = max((r for r in best_by_method.values() if r["best"]),
                     key=lambda r: r["best"]["value"], default=None)
        if winner is None:
            per_fold.append({"fold_id": fid, "annualised_pp": 0.0, "test_days": 90,
                             "note": "no admissible candidate"})
            continue

        t = RUN.evaluate_test_once(fid, winner["best"]["params"], winner["method"])
        per_fold.append({"fold_id": fid, "annualised_pp": t["unregularised"],
                         "test_days": 90, "method": winner["method"],
                         "params": winner["best"]["params"],
                         "n_trades": t["n_trades"], "trades_per_year": t["trades_per_year"],
                         "mean_hold_h": t["mean_hold_h"], "mean_exposure": t["mean_exposure"]})
        overfit_rows.append({"fold_id": fid, "candidate_id": t["candidate_id"],
                             "train_pp": winner["best"]["value"],
                             "test_pp": t["unregularised"]})
        print(f"  {fid} TEST(once): {t['unregularised']:+.3f} pp/yr "
              f"({t['n_trades']} trades)", flush=True)

    # The incumbent, evaluated the same way, is the comparator.
    inc = []
    for fid in fold_ids:
        p = PA.load(fid, train=False)
        s = EV.score(p, dict(SP.INCUMBENT), EV.incumbent_trades_per_year(p))
        inc.append(s.unregularised)
    incumbent_pp = sum(inc) / max(1, len(inc))
    best_pp = sum(f["annualised_pp"] for f in per_fold) / max(1, len(per_fold))

    # Did CMA-ES beat the control out-of-sample? (FR-008 / SC-009)
    by_method = {}
    for f in per_fold:
        if "method" in f:
            by_method.setdefault(f["method"], []).append(f["annualised_pp"])
    beats_random = None
    if "cmaes" in by_method and "random" in by_method:
        beats_random = (sum(by_method["cmaes"]) / len(by_method["cmaes"])
                        > sum(by_method["random"]) / len(by_method["random"]))

    gates = {"SC-001": VD.pooled(per_fold, incumbent_pp),
             "SC-002a": VD.per_fold_positivity(per_fold),
             "SC-008": VD.overfit_gap(overfit_rows)}
    best = max(per_fold, key=lambda f: f["annualised_pp"], default=None)
    if best and "params" in best:
        gates["SC-003"] = VD.baselines(best_pp, best.get("mean_exposure", 0.0),
                                       best.get("trades_per_year", 0.0))

    state = {"folds": fold_ids, "budget": budget, "seed": seed,
             "per_fold": per_fold, "counts": counts, "gates": gates,
             "incumbent_pp": round(incumbent_pp, 4), "best_pp": round(best_pp, 4),
             "beats_random": beats_random,
             **RP.incumbent_delta(best.get("params") if best else None, best_pp, incumbent_pp),
             "seconds": round(time.time() - t0, 1)}
    STATE.write_text(json.dumps(state, indent=2))
    RP.write_shared_summary(f"es-{seed}-{budget}",
                            {"folds": len(fold_ids), "budget": budget})
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="f00..f14")
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--allow-uncalibrated", action="store_true")
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    st = drive(parse_span(a.folds), a.budget, a.seed, a.allow_uncalibrated)
    print(json.dumps({k: v for k, v in st.items()
                      if k not in ("per_fold", "counts", "gates")}, indent=2))
    ok, rep = H.audit()
    print(f"\naudit: {'PASS' if ok else 'FAIL'}  {rep['problems'] or ''}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
