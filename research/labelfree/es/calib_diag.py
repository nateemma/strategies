"""Where does the surrogate disagree with reality? (diagnostic for the calibration gate)

A global rank correlation can hide the failure that matters most here. If the disagreements
CONCENTRATE on one parameter -- say the ATR multiplier -- then the surrogate is unreliable
exactly where the search will push hardest, and a headline rho of 0.8 would be reassuring
about the wrong thing.

So this reads calibration.json and asks, per dimension: does the surrogate's rank error grow
with that parameter's distance from the incumbent?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import space as SP            # noqa: E402


def analyse(path: Path) -> dict:
    rep = json.loads(Path(path).read_text())
    ok = [r for r in rep["rows"]
          if r.get("surrogate") is not None and r.get("backtest") is not None]
    if len(ok) < 5:
        return {"error": f"only {len(ok)} comparable candidates"}

    from scipy.stats import spearmanr
    a = np.array([r["surrogate"] for r in ok])
    b = np.array([r["backtest"] for r in ok])
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    err = np.abs(ra - rb)                     # per-candidate rank error

    per_dim = []
    for d in SP.DIMS:
        x = np.array([float(r["params"][d.name]) for r in ok])
        dist = np.abs(x - d.incumbent) / d.width
        rho = float(spearmanr(dist, err).statistic) if np.std(dist) > 0 else float("nan")
        per_dim.append({"parameter": d.name, "rank_error_vs_distance_rho": round(rho, 4)})

    per_dim.sort(key=lambda r: -(r["rank_error_vs_distance_rho"]
                                 if np.isfinite(r["rank_error_vs_distance_rho"]) else -9))
    return {"n_compared": len(ok),
            "mean_rank_error": round(float(err.mean()), 3),
            "max_rank_error": int(err.max()),
            "level_offset_mean_pp": round(float((b - a).mean()), 3),
            "level_offset_std_pp": round(float((b - a).std(ddof=1)), 3),
            "per_dimension": per_dim,
            "note": ("A positive rho means the surrogate's ranking degrades as that parameter "
                     "moves away from the incumbent — the search would be least reliable "
                     "exactly where it explores.")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration",
                    default=str(Path(__file__).resolve().parent / "calibration.json"))
    a = ap.parse_args()
    rep = analyse(Path(a.calibration))
    print(json.dumps(rep, indent=2))
    if "per_dimension" in rep:
        print(f"\nlevel offset (backtest - surrogate): "
              f"{rep['level_offset_mean_pp']:+.2f} +- {rep['level_offset_std_pp']:.2f} pp")
        print("A roughly CONSTANT offset is harmless — the search only uses ranks.")
        worst = rep["per_dimension"][0]
        print(f"\nworst dimension: {worst['parameter']} "
              f"(rank-error vs distance rho {worst['rank_error_vs_distance_rho']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
