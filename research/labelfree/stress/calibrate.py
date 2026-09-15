"""THE BLOCKING GATE: does resampling change what the policy experiences? (T022-T028)

THE TAIL-GAN CRITERION. Judge a generator by whether it reproduces the P&L distribution of the
strategies you care about. Until it does, nothing it says about a NEW policy means anything.

THE EVALUATOR IS HELD CONSTANT (T024a, required by the pre-implementation analyze pass).
Synthetic P&L is compared against THE SAME POLICY SCORED BY THE SAME EVALUATOR ON THE REAL
PANELS -- not against the 378 archived freqtrade trades. Comparing surrogate-on-synthetic with
freqtrade-on-real would conflate two unrelated differences:

    1. synthetic paths vs real paths                 <- the thing being measured
    2. the surrogate vs freqtrade                    <- a KNOWN, MEASURED gap:
                                                        level offset -5.30 +- 23.46 pp over 21
                                                        candidates, with residual exit-mix
                                                        divergence (roi 22 vs a real 13 on f07)

A failed test could not be attributed to either. SC-001 is this feature's ONLY gate and a
failure is a KILL (clarification Q3 left no fallback generator), so the feature could have died
of a confound. The archive is still reported, as a SECONDARY SANITY REFERENCE that never gates.

WHAT THIS GATE CAN AND CANNOT CATCH -- stated because a near-guaranteed pass must not be
reported as validation (T026). The bootstrap resamples REAL blocks, so the body of the
distribution should match by construction. What it can genuinely catch:

  * spurious gaps at block joins. A join splices two unrelated moments together, creating a
    discontinuity the real series never had. Those look like gaps, and gaps fire stops.
  * a wrong drawdown distribution. Maximum drawdown is a function of CONSECUTIVE bad blocks,
    and resampling destroys the long-range dependence that produces sustained drawdowns.

So this is a plumbing-and-tails check. The feature's informative output is the STRESS
distribution, not this.
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

from research.labelfree.es import evaluate as EV                # noqa: E402
from research.labelfree.es import space as SP                   # noqa: E402
from research.labelfree.stress import bank as BK                # noqa: E402

# --- PRE-REGISTERED, research.md R3, fixed before the first calibration run. Do not tune. ---
#
# KS_MAX is set from what the comparison can RESOLVE, not from what would impress: at n = 378
# the 95% critical value against an identical distribution is 1.36/sqrt(378) = 0.0700, so 0.10
# admits sampling noise without admitting a visibly different shape.
#
# The tail bands are ASYMMETRIC on purpose. A synthetic tail FATTER than real makes the harness
# conservative and is tolerable; a tail THINNER than real is the dangerous direction, because it
# would let a fragile policy through. DD_RATIO's lower bound was tightened 0.6 -> 0.7 by the
# analyze pass to match ES, both being tail statistics guarding the same direction.
KS_MAX = 0.10
ES_RATIO = (0.7, 1.4)
DD_RATIO = (0.7, 1.6)
ES_Q = 0.05                       # expected shortfall quantile
CALIBRATION = Path(__file__).resolve().parent / "calibration.json"


def expected_shortfall(x: np.ndarray, q: float = ES_Q) -> float:
    """Mean of the worst q-quantile. NaN-safe and empty-safe."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(np.floor(q * x.size)))
    return float(np.sort(x)[:k].mean())


def max_drawdown(pnl: np.ndarray) -> float:
    """Max peak-to-trough drop of the cumulative per-trade equity path, as a positive number.

    NOT reused from regime/phase_b_risk_normalize.py: that one consumes a per-bar LOG-RETURN
    series, while this needs drawdown over a per-TRADE additive equity path under the fixed-stake
    convention. Checked before reusing (T006); different quantity, same name.
    """
    a = np.asarray(pnl, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    eq = np.cumsum(a)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def _trades(panel: dict, params: dict) -> np.ndarray:
    return np.asarray(EV.run(panel, params).pnl, dtype=float)


def run(bank_path: Path, n_paths: int | None = None) -> dict:
    prov, idx, panels = BK.load(Path(bank_path))
    params = dict(SP.INCUMBENT)

    # REAL side — the same policy, the same evaluator, the real panels.
    real = np.concatenate([_trades(panels[f], params) for f in prov["fold_ids"]])

    # SYNTHETIC side — the same policy, the same evaluator, resampled panels.
    from research.labelfree.stress import bootstrap as BS
    syn_all, per_path_dd = [], []
    limit = n_paths or prov["n_paths"]
    t0 = time.time()
    for f in prov["fold_ids"]:
        for row in idx[f][:limit]:
            p = _trades(BS.resample_panel(panels[f], row), params)
            if p.size:
                syn_all.append(p)
                per_path_dd.append(max_drawdown(p))
    syn = np.concatenate(syn_all) if syn_all else np.array([])

    from scipy.stats import ks_2samp
    ks = float(ks_2samp(syn, real).statistic) if syn.size and real.size else float("nan")
    es_r = expected_shortfall(real)
    es_s = expected_shortfall(syn)
    es_ratio = float(es_s / es_r) if es_r not in (0.0,) and np.isfinite(es_r) else float("nan")

    # Real drawdown is measured PER FOLD, so it is comparable with a per-path figure.
    real_dd = float(np.median([max_drawdown(_trades(panels[f], params))
                               for f in prov["fold_ids"]]))
    syn_dd = float(np.median(per_path_dd)) if per_path_dd else float("nan")
    dd_ratio = float(syn_dd / real_dd) if real_dd else float("nan")

    passed = bool(ks <= KS_MAX
                  and ES_RATIO[0] <= es_ratio <= ES_RATIO[1]
                  and DD_RATIO[0] <= dd_ratio <= DD_RATIO[1])

    return {
        "bank": Path(bank_path).name,
        "block_length": prov["block_length"],
        "paths_used": limit,
        "n_real_trades": int(real.size),
        "n_synthetic_trades": int(syn.size),
        "ks_distance": round(ks, 4),
        "es_real": round(es_r, 6), "es_synthetic": round(es_s, 6),
        "es_ratio": round(es_ratio, 4),
        "dd_real": round(real_dd, 6), "dd_synthetic": round(syn_dd, 6),
        "dd_ratio": round(dd_ratio, 4),
        "tolerances": {"ks_max": KS_MAX, "es_ratio": list(ES_RATIO),
                       "dd_ratio": list(DD_RATIO),
                       "pre_registered": "research.md R3, fixed before the first run"},
        "passed": passed,
        "comparison": ("surrogate-on-synthetic vs surrogate-on-real-panels; the evaluator is "
                       "held constant so only the paths vary"),
        "caveat": ("the bootstrap resamples REAL blocks, so the body should match by "
                   "construction. This is a plumbing-and-tails check, not validation of the "
                   "harness's usefulness."),
        "seconds": round(time.time() - t0, 1),
    }


def require_calibration(path: Path = CALIBRATION) -> dict:
    """Every downstream entry point calls this (T025)."""
    if not path.exists():
        raise SystemExit(
            "BLOCKED: no calibration.json. The generator must reproduce the incumbent's P&L "
            "distribution before any stress result is admissible (SC-001).\n"
            "  python -m research.labelfree.stress.calibrate --bank <bank>")
    c = json.loads(path.read_text())
    if not c.get("passed"):
        raise SystemExit(
            f"BLOCKED: calibration FAILED on {c.get('bank')} "
            f"(ks {c.get('ks_distance')}, es {c.get('es_ratio')}, dd {c.get('dd_ratio')}). "
            f"Clarification Q3 left no fallback generator, so this is a KILL (SC-007).")
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--paths", type=int, default=None, help="cap paths per fold (dev only)")
    ap.add_argument("--out", default=str(CALIBRATION))
    a = ap.parse_args()

    rep = run(Path(a.bank), a.paths)
    Path(a.out).write_text(json.dumps(rep, indent=2))

    ok = lambda b: "PASS" if b else "FAIL"                       # noqa: E731
    print(f"{'PASSED' if rep['passed'] else 'FAILED'}  L={rep['block_length']}  "
          f"({rep['seconds']}s)")
    print(f"  ks_distance  {rep['ks_distance']:>8.4f}   <= {KS_MAX}        "
          f"{ok(rep['ks_distance'] <= KS_MAX)}")
    print(f"  es_ratio     {rep['es_ratio']:>8.4f}   {ES_RATIO}   "
          f"{ok(ES_RATIO[0] <= rep['es_ratio'] <= ES_RATIO[1])}")
    print(f"  dd_ratio     {rep['dd_ratio']:>8.4f}   {DD_RATIO}   "
          f"{ok(DD_RATIO[0] <= rep['dd_ratio'] <= DD_RATIO[1])}")
    print(f"  real {rep['n_real_trades']} trades | synthetic {rep['n_synthetic_trades']}")
    print(f"-> {a.out}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
