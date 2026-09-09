"""The feature's headline output: the ACHIEVED noise floor (FR-007..FR-011, FR-021).

`power.py` reports the floor of ONE pooled set. `regime/tail/compare.py` reports the DELTA between
two. They deliberately do not bootstrap the same quantity (research R8).

Reading the result: a floor that misses the 4.0pp target, or that falls more slowly than 1/sqrt(n)
predicts, is a RESULT to report -- the most valuable one available, because it would mean the power
ceiling is lower than the data volume implies. It is never grounds to retune the fold geometry
until the number improves (research R9).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(ROOT / "user_data/strategies"))
from regime.wf import (BASELINE_HALF_WIDTH_PP, BASELINE_N_TRADES, BASELINE_PAIRED_PP,
                       BOOTSTRAP_SEED, N_RESAMPLES, TARGET_HALF_WIDTH_PP, TARGET_POOLED_TRADES)


def bootstrap_floor(profits: np.ndarray, scale: float) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(profits)
    boot = profits[rng.integers(0, n, (N_RESAMPLES, n))].sum(axis=1) * scale
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi),
            "half_width_pp": float((hi - lo) / 2), "sd_pp": float(boot.std())}


def report(pooled: dict, subset: str | None = None) -> str:
    trades = pooled["trades"]
    per_fold = pooled["per_fold"]
    if subset == "recent-third":
        ids = [p["fold_id"] for p in per_fold]
        keep = set(ids[-max(1, len(ids) // 3):])
        trades = [t for t in trades if t["fold_id"] in keep]
        per_fold = [p for p in per_fold if p["fold_id"] in keep]

    out = []
    partial = pooled["status"] != "complete"
    if partial:
        out.append("*** PARTIAL — folds missing: "
                   f"{', '.join(pooled['folds_missing'])}. This floor is NOT the full plan's "
                   "result and must not be reported as one (FR-020).\n")
    if not trades:
        out.append("no pooled trades — nothing to measure")
        return "\n".join(out)

    p = np.array([t["profit_abs"] for t in trades], dtype=float)
    scale = trades[0]["scale_pp_per_abs"]
    bs = bootstrap_floor(p, scale)
    n = len(p)

    # ANNUALISE BEFORE COMPARING. The bootstrap runs on the SUM over the pooled span, which is
    # ~3.7 years; the baseline half-width is on a ONE-YEAR window. Comparing them directly makes a
    # working harness look like a 0.62x regression -- it did, on the first run of this report.
    years = sum(f["test_days"] for f in per_fold) / 365.0
    hw_annual = bs["half_width_pp"] / years

    # sqrt(n) prediction from the measured single-window baseline (research R9: a HYPOTHESIS).
    predicted = BASELINE_HALF_WIDTH_PP * (BASELINE_N_TRADES / n) ** 0.5
    shortfall = hw_annual - predicted

    # Inter-fold correlation: the explanation when pooling underdelivers.
    fr = [f["return_pct"] for f in per_fold if f["return_pct"] is not None]
    if len(fr) > 2:
        a = np.array(fr[:-1]); b = np.array(fr[1:])
        lag1 = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
        disp = float(np.std(fr))
    else:
        lag1, disp = float("nan"), float("nan")

    subset_note = f"  [subset: {subset}]" if subset else ""
    out += [
        f"POOLED WALK-FORWARD POWER REPORT{subset_note}",
        f"  run {pooled['run_id']}  family {pooled['family']}  params_sha {pooled.get('params_sha')}",
        f"  folds {len(per_fold)}/{pooled['folds_planned']}   status {pooled['status'].upper()}",
        "",
        f"  pooled trades                  {n:>10}      (baseline {BASELINE_N_TRADES}, "
        f"target >= {TARGET_POOLED_TRADES})",
        f"  pooled span                    {years:>10.2f} yr-equivalent",
        f"  bootstrap 95% CI (whole span)  [{bs['ci_low']:+.3f}%, {bs['ci_high']:+.3f}%]",
        f"  half-width on the whole span   {bs['half_width_pp']:>10.3f}pp",
        "",
        f"  ACHIEVED FLOOR (annualised)    {hw_annual:>10.3f}pp/yr   <- comparable to baseline",
        f"  baseline floor (1 window)      {BASELINE_HALF_WIDTH_PP:>10.3f}pp/yr"
        f"  (paired {BASELINE_PAIRED_PP}pp)",
        f"  improvement ratio              {BASELINE_HALF_WIDTH_PP/hw_annual:>10.2f}x",
        "",
        f"  MINIMUM RESOLVABLE EFFECT      {hw_annual:>10.3f}pp/yr",
        "    Anything smaller than this cannot be distinguished from noise on this instrument,",
        "    however many times it is repeated. Parameter-scale effects (1-2pp) remain",
        "    UNRESOLVABLE — that is a data limit, not a defect of the harness.",
        "",
        f"  1/sqrt(n) prediction           {predicted:>10.3f}pp",
        f"  shortfall (achieved - pred)    {shortfall:>+10.3f}pp",
        f"  fold-return lag-1 correlation  {lag1:>10.3f}   (dispersion {disp:.3f}pp)",
    ]
    out.append(f"  per-trade dispersion           "
               f"{bs['half_width_pp']/n**0.5:>10.3f}   "
               f"(baseline {BASELINE_HALF_WIDTH_PP/BASELINE_N_TRADES**0.5:.3f})")
    if shortfall > 0.25:
        out.append("    ^ POOLING UNDERDELIVERED. Fold returns are not independent; the power")
        out.append("      ceiling is lower than data volume implies. This is a FINDING (R9).")
    elif shortfall < -0.1:
        out.append("    ^ pooling OVER-delivered: per-trade dispersion across folds is lower than")
        out.append("      in the single baseline window, so the floor fell faster than 1/sqrt(n).")
    out += ["", f"  {'fold':<7}{'trades':>8}{'return%':>10}{'days':>7}  status"]
    for f in per_fold:
        r = "     n/a" if f["return_pct"] is None else f"{f['return_pct']:>10.3f}"
        out.append(f"  {f['fold_id']:<7}{f['n_trades']:>8}{r}{f['test_days']:>7}  {f['status']}")

    tgt = ("MET" if hw_annual <= TARGET_HALF_WIDTH_PP else "MISSED")
    cnt = ("MET" if n >= TARGET_POOLED_TRADES else "MISSED")
    out += ["",
            f"  SC-001  (mandatory) floor measured and reported vs baseline : SATISFIED",
            f"  SC-001a (target)    floor <= {TARGET_HALF_WIDTH_PP}pp                          : {tgt}",
            f"  SC-002              pooled trades >= {TARGET_POOLED_TRADES}                     : {cnt}"]
    if tgt == "MISSED":
        out.append("    A missed TARGET is a reportable RESULT, not a feature failure, and is")
        out.append("    never grounds to retune the fold geometry (SC-001a, research R9).")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--folds-subset", choices=["recent-third"], default=None)
    a = ap.parse_args()
    print(report(json.loads(Path(a.pooled).read_text()), a.folds_subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
