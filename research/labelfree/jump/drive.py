"""T031 / T031a -- gate activity and rvol redundancy across all 15 folds, over the seed ensemble.

THIS IS THE FEATURE'S MVP AND ITS FIRST KILL GATE, and it needs no backtest at all. Three ways
to fail, all pre-registered:

  inert                 the gate blocked nothing                       (SC-012)
  redundant_with_rvol   novel_block_rate < 0.20 -- rvol got there first (FR-024, SC-019)
  seed artifact         block-rate spread across seeds exceeds the effect (SC-017)

Any of them is a DEFECT report under Principle IV -- "the gate never fired" -- and explicitly
not "regime gating does not help".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree import folds as FO                      # noqa: E402
from research.labelfree.jump import (K_STATES, MIN_ACTIVE_FOLDS,   # noqa: E402
                                     OUT, SEEDS, SEED_AGREEMENT_FLOOR)
from research.labelfree.jump import features as FE               # noqa: E402
from research.labelfree.jump import gate as G                    # noqa: E402
from research.labelfree.jump import model as M                   # noqa: E402
from research.labelfree.jump import redundancy as RD             # noqa: E402
from research.labelfree.jump import refit as R                   # noqa: E402


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def calibrate(fold, pairs, grid=None) -> dict:
    """lam is calibrated ONCE, on fold 0's training window, then frozen (research R5)."""
    lo = pd.Timestamp(fold.train_start, tz="UTC")
    hi = pd.Timestamp(fold.train_end, tz="UTC")
    xs = []
    for p in pairs[:4]:
        f = FE.build(p, lo, hi)
        if len(f) > 2000:
            xs.append(f.to_numpy(float)[:, :3])       # own-vol block only: shared across pairs
    x = np.vstack(xs)
    grid = np.geomspace(1e1, 1e5, 12) if grid is None else grid
    return M.calibrate_lam(x, K_STATES, FE.TARGET_SWITCH_RATE, grid=grid)


def run_fold(fold, lam: float, seeds=SEEDS) -> dict:
    lo = pd.Timestamp(fold.train_start, tz="UTC")
    hi = pd.Timestamp(fold.test_end, tz="UTC")
    first = pd.Timestamp(fold.train_end, tz="UTC")

    per_seed, red, agrees = [], [], []
    for pair in fold.pair_universe:
        try:
            f = FE.build(pair, lo, hi)
        except Exception:
            continue
        if len(f) < 5000:
            continue
        rv = RD.rvol_series(pair, lo, hi)

        # Seed agreement is taken from the DEPLOYED state series, not from a separate global
        # fit. A global fit would both double the cost and measure the wrong object: what
        # matters is whether the sequence the gate actually consumes depends on the seed.
        seqs = []
        for si, seed in enumerate(seeds):
            st = R.run(f, first, lam, k=K_STATES, seed=seed, mode="filtered")
            if st.empty:
                continue
            hv = K_STATES - 1                       # states arrive volatility-ranked
            dec = G.decide(st, hv)
            per_seed.append({"pair": pair, "seed": seed,
                             "n": len(dec), "n_blocked": int(dec.blocked.sum())})
            seqs.append(st.set_index("timestamp")["state"])
            if si == 0:
                red.append(RD.overlap(st, rv, hv))
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                a, b = seqs[i].align(seqs[j], join="inner")
                if len(a):
                    agrees.append(float((a.to_numpy() == b.to_numpy()).mean()))

    if not per_seed:
        return {"fold_id": fold.fold_id, "status": "no_data"}

    df = pd.DataFrame(per_seed)
    by_seed = df.groupby("seed").apply(
        lambda g: g.n_blocked.sum() / max(g.n.sum(), 1), include_groups=False)
    rates = by_seed.to_numpy(float)

    tot = df[df.seed == df.seed.min()]
    act = G.activity(pd.DataFrame({"blocked": np.r_[
        np.ones(int(tot.n_blocked.sum()), bool), np.zeros(int(tot.n.sum() - tot.n_blocked.sum()), bool)]}),
        fold.fold_id)

    nb = sum(r["n_blocked"] for r in red)
    nn = sum(r["n_novel"] for r in red)
    novel = (nn / nb) if nb else 0.0
    return {
        "fold_id": fold.fold_id, "status": "ok", "n_pairs": int(df.pair.nunique()),
        "activity": act,
        "block_rate_by_seed": [float(v) for v in rates],
        "block_rate_iqm": float(np.mean(np.sort(rates)[len(rates)//4: len(rates)-len(rates)//4]))
                          if len(rates) >= 4 else float(np.mean(rates)),
        "across_seed_spread": float(rates.max() - rates.min()),
        "seed_agreement": float(np.mean(agrees)) if agrees else float("nan"),
        "novel_block_rate": float(novel),
        "redundancy_verdict": "redundant_with_rvol" if novel < 0.20 else "distinct",
    }


def main() -> int:
    fl = FO.folds()
    _log(f"{len(fl)} folds")
    cal = calibrate(fl[0], list(fl[0].pair_universe))
    _log(f"lam = {cal['lam']:.2f}  achieved {cal['achieved_switch_rate']*FE.BARS_PER_YEAR:.1f} sw/yr "
         f"(target {FE.TARGET_SWITCHES_PER_YEAR}) guard_bound={cal['guard_bound']}")

    out = []
    for fold in fl:
        t0 = time.time()
        r = run_fold(fold, cal["lam"])
        out.append(r)
        if r["status"] == "ok":
            a = r["activity"]
            _log(f"{fold.fold_id}: {a['verdict']:6s} block={a['block_rate']:.3f} "
                 f"seed_spread={r['across_seed_spread']:.3f} agree={r['seed_agreement']:.3f} "
                 f"novel={r['novel_block_rate']:.3f} [{r['redundancy_verdict']}] ({time.time()-t0:.0f}s)")
        else:
            _log(f"{fold.fold_id}: {r['status']}")

    ok = [r for r in out if r["status"] == "ok"]
    n_active = sum(1 for r in ok if r["activity"]["verdict"] == "active")
    n_redundant = sum(1 for r in ok if r["redundancy_verdict"] == "redundant_with_rvol")
    spread = np.mean([r["across_seed_spread"] for r in ok]) if ok else float("nan")
    agree = np.mean([r["seed_agreement"] for r in ok]) if ok else float("nan")

    summary = {
        "lam": cal["lam"], "calibration": {k: v for k, v in cal.items() if k != "grid"},
        "n_folds": len(ok), "n_active": n_active,
        "meets_min_active": n_active >= MIN_ACTIVE_FOLDS,
        "n_redundant_with_rvol": n_redundant,
        "mean_across_seed_spread": float(spread),
        "mean_seed_agreement": float(agree),
        "seed_agreement_floor": SEED_AGREEMENT_FLOOR,
        "folds": out,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "activity.json").write_text(json.dumps(summary, indent=2))
    _log(f"WROTE {OUT/'activity.json'}")
    _log(f"active {n_active}/{len(ok)} (need {MIN_ACTIVE_FOLDS}) | "
         f"redundant {n_redundant}/{len(ok)} | mean seed spread {spread:.3f} | "
         f"mean seed agreement {agree:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
