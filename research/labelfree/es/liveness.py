"""Principle IV, applied to the SEARCH SPACE itself (T028, T029).

Sweep each dimension alone across its bounds with the other nine at the incumbent, and report
the span of the objective. A dimension that cannot move the objective is reported INERT rather
than searched.

WHY THIS RUNS BEFORE THE SEARCH AND NOT AFTER. Four parameters in this study were configured,
logged, believed active and reaching nothing -- `lam_turnover` passed to a builder with no
loss-side branch, purge_bars set with the function never called, trailing_stop_positive
unreachable behind an ROI that fired first, and a hardcoded rvol literal shadowing its own
hyperparameter. Every one of them passed a wiring check. Only a measured SPAN catches this.

THRESHOLDS ARE PRE-REGISTERED (data-model.md §6). They are constants here, not arguments,
because a cut chosen after seeing the spans is unfalsifiable in a feature whose headline
number is the probability of overfitting.

    inert   span <  0.10 pp/yr     numerically indistinguishable from no effect; a truly
                                   unreached parameter gives exactly 0.00
    weak    span <  3.35 pp/yr     live, but cannot on its own clear the acceptance floor
    live    span >= 3.35 pp/yr     can move the objective past the floor
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

from research.labelfree.es import evaluate as EV         # noqa: E402
from research.labelfree.es import panel as PA            # noqa: E402
from research.labelfree.es import space as SP            # noqa: E402

# PRE-REGISTERED — data-model.md §6. Do not tune.
INERT_PP = 0.10
WEAK_PP = 3.35


def verdict(span_pp: float) -> str:
    if span_pp < INERT_PP:
        return "inert"
    return "weak" if span_pp < WEAK_PP else "live"


def sweep(fold_ids: list[str], steps: int = 25) -> dict:
    """One dimension at a time, the other nine pinned at the incumbent."""
    panels = {f: PA.load(f, train=True) for f in fold_ids}
    inc_tpy = {f: EV.incumbent_trades_per_year(p) for f, p in panels.items()}

    def objective(params: dict) -> float:
        """Mean UNREGULARISED return across folds.

        Unregularised deliberately: the distance penalty is a function of the parameters
        themselves, so including it would manufacture a span for every dimension and make
        even a genuinely inert one look live.
        """
        vals = []
        for f, p in panels.items():
            s = EV.score(p, params, inc_tpy[f])
            vals.append(s.unregularised if s.admissible else np.nan)
        return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")

    base = objective(dict(SP.INCUMBENT))
    out = []
    for d in SP.DIMS:
        grid = (np.unique(np.round(np.linspace(d.lo, d.hi, steps)).astype(int))
                if d.integer else np.linspace(d.lo, d.hi, steps))
        curve = []
        for v in grid:
            p = dict(SP.INCUMBENT)
            p[d.name] = int(v) if d.integer else float(v)
            try:
                SP.validate(p)
            except SP.Inadmissible:
                continue
            curve.append((float(v), objective(p)))
        vals = np.array([c[1] for c in curve], dtype=float)
        finite = vals[np.isfinite(vals)]
        span = float(finite.max() - finite.min()) if finite.size else 0.0
        out.append({"parameter": d.name, "span_pp": round(span, 4),
                    "verdict": verdict(span), "points": len(curve),
                    "best_value": (curve[int(np.nanargmax(vals))][0]
                                   if finite.size else None),
                    "at_incumbent_pp": round(base, 4),
                    "one_sided": d.one_sided})
    return {"folds": fold_ids, "incumbent_pp": round(base, 4),
            "thresholds": {"inert_pp": INERT_PP, "weak_pp": WEAK_PP,
                           "pre_registered": "data-model.md §6"},
            "dimensions": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="f05..f14")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "liveness.json"))
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    t0 = time.time()
    rep = sweep(parse_span(a.folds), a.steps)
    Path(a.out).write_text(json.dumps(rep, indent=2))

    print(f"incumbent {rep['incumbent_pp']:+.2f} pp/yr over {len(rep['folds'])} folds"
          f"   (inert < {INERT_PP}, weak < {WEAK_PP} pp/yr; pre-registered)")
    print(f"{'parameter':<26}{'span pp':>10}  {'verdict':<7} best")
    for d in sorted(rep["dimensions"], key=lambda x: -x["span_pp"]):
        b = "—" if d["best_value"] is None else f"{d['best_value']:.4g}"
        print(f"{d['parameter']:<26}{d['span_pp']:>10.3f}  {d['verdict']:<7} {b}")
    inert = [d["parameter"] for d in rep["dimensions"] if d["verdict"] == "inert"]
    if inert:
        print(f"\nINERT (reported, not searched): {', '.join(inert)}")
    print(f"\n{time.time() - t0:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
