"""Variant-grid driver with honest trial accounting (T030, T031a, T032, T032a, T033).

FR-019 / SC-015: every configuration is appended to the trial log BEFORE it runs, with
`result: null`. A log written afterwards silently omits abandoned runs, which is the exact
mechanism DSR and PBO exist to correct for.

The grid is 2 objectives x 2 turnover controls x 3 cost scales = 12 configurations, each at
>=5 seeds, screened on ONE fold before the best two proceed to the pooled run.

Principle VIII note: this factorial is a joint search and carries a documented exception in
plan.md — the objective x turnover-control interaction is the object of study, not a
nuisance to hold fixed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from regime import regimedata as rd                                     # noqa: E402
from research.labelfree import features as FT                           # noqa: E402
from research.labelfree import folds as F                               # noqa: E402
from research.labelfree import stats as S                               # noqa: E402
from research.labelfree import train as T                               # noqa: E402
from research.labelfree import trials                                   # noqa: E402
from research.labelfree.costs import CostModel                          # noqa: E402

OBJECTIVES = ("utility", "robust_sharpe")
CONTROLS = ("band", "penalty")
GAMMAS = (0.5, 1.0, 1.5)
SEEDS = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# T031a — CARRY-FORWARD RULE, pre-registered before any result is seen.
# ---------------------------------------------------------------------------
CARRY_FORWARD_RULE = {
    "registered_at": "2026-09-10",
    "n_carried": 2,
    "primary": "Rank configurations by the interquartile mean of net log return across "
               "seeds on the screening fold's TEST range; carry the top 2.",
    "resolvability_test": "Bootstrap the IQM difference between rank-1 and rank-3. The "
                          "screen RESOLVES only if that 95% CI excludes zero.",
    "fallback_when_unresolved": [
        "Carry one configuration per OBJECTIVE (the best 'utility' and the best "
        "'robust_sharpe'), regardless of their relative rank.",
        "Rationale: when the screen cannot separate configurations, ranking is noise, so "
        "carry forward along the axis the brief exists to compare rather than along a "
        "noise ordering. Ties within an objective break by lower turnover, then lower "
        "gamma, then lower seed - all pre-registered, none data-dependent.",
    ],
    "note": "Recording this after seeing results voids it (plan.md Complexity Tracking).",
}


def grid() -> list[dict]:
    out = []
    for obj, ctl, g in itertools.product(OBJECTIVES, CONTROLS, GAMMAS):
        out.append({"objective": obj, "turnover_control": ctl, "gamma": g})
    return out


def load_fold_data(fold, pairs: list[str], train: bool):
    """Pooled panel aligned ON DATE, with a validity mask.

    Pairs must be aligned by timestamp, never by position. Measured 2026-09-10: over f08's
    training range, position 51,546 is 2022-06-22 for SOL/LTC/BCH but 2024-12-20 for XRP,
    so stacking by position would present cross-sections spanning two and a half years as
    if they were simultaneous, and any cross-pair structure learned would be an artifact.

    Pairs with no data at a timestamp are masked rather than dropped: the mask forces
    exposure to zero there, so a gap costs nothing and earns nothing instead of truncating
    every other pair to the shortest history (XRP's 911-day delisting gap would otherwise
    cut the panel from 139k bars to 51k).

    Returns (X, P, C, V, M) each shaped (n_pairs, T, ...), plus the shared index.
    """
    start, end = (fold.train_start, fold.train_end) if train else (fold.test_start, fold.test_end)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    index = pd.date_range(lo, hi, freq="15min", tz="UTC")

    X, P, C, V, M = [], [], [], [], []
    for p in pairs:
        x, px, vq, cols = FT.build_features(p, start, end)
        df = rd.load_ohlcv(p.split("/")[0])
        df = df[(df.index >= lo) & (df.index <= hi)]
        n = min(len(df), len(px), len(x))
        own = df.index[:n]
        c = CostModel().per_unit(df, evaluation=True).to_numpy()[:n]

        fx = pd.DataFrame(x[:n], index=own).reindex(index)
        valid = fx.notna().all(axis=1).to_numpy()
        fx = fx.ffill().fillna(0.0)
        fp = pd.Series(px[:n], index=own).reindex(index).ffill().bfill()
        fc = pd.Series(c, index=own).reindex(index).ffill().bfill()
        fv = pd.Series(vq[:n], index=own).reindex(index).fillna(0.0)

        X.append(fx.to_numpy(dtype=np.float32)); P.append(fp.to_numpy(dtype=np.float64))
        C.append(fc.to_numpy(dtype=np.float64)); V.append(fv.to_numpy(dtype=np.float64))
        M.append(valid.astype(np.float32))
    return (np.stack(X), np.stack(P), np.stack(C), np.stack(V), np.stack(M), index)


def run(fold_id: str, pairs: list[str] | None, epochs: int, seeds=SEEDS,
        log: Path = trials.LOG) -> dict:
    fold = {f.fold_id: f for f in F.folds()}[fold_id]
    pairs = pairs or list(fold.pair_universe)
    run_id = trials.new_run_id()
    print(f"screening fold {fold_id} [run {run_id}]: train {fold.train_start}.."
          f"{fold.train_end}, test {fold.test_start}..{fold.test_end}, "
          f"{len(pairs)} pairs", flush=True)

    # Pre-register the carry-forward rule before any run (T031a).
    trials.pre_register({"trial_id": f"RULE-carry-forward-{fold_id}@{run_id}",
                         "kind": "carry_forward_rule", "fold_id": fold_id,
                         "run_id": run_id, **CARRY_FORWARD_RULE}, log)

    Xtr, Ptr, Ctr, Vtr, Mtr, itr = load_fold_data(fold, pairs, train=True)
    Xte, Pte, Cte, Vte, Mte, ite = load_fold_data(fold, pairs, train=False)
    print(f"  train panel {Xtr.shape} ({100*Mtr.mean():.0f}% valid)  "
          f"test panel {Xte.shape} ({100*Mte.mean():.0f}% valid)", flush=True)

    results: dict[str, list[float]] = {}
    for cfg_d in grid():
        key = f"{cfg_d['objective']}-{cfg_d['turnover_control']}-g{cfg_d['gamma']}"
        results[key] = []
        for seed in seeds:
            tid = trials.trial_id(cfg_d["objective"], cfg_d["turnover_control"],
                                  cfg_d["gamma"], seed, fold_id, run=run_id)
            # ---- BEFORE the run (FR-019) ----
            trials.pre_register({"trial_id": tid, "fold_id": fold_id, "seed": seed,
                                 "run_id": run_id,
                                 "n_pairs": len(pairs), "epochs": epochs,
                                 "screening_range": [fold.test_start, fold.test_end],
                                 **cfg_d}, log)
            t0 = time.time()
            try:
                cfg = T.TrainConfig(objective=cfg_d["objective"],
                                    turnover_control=cfg_d["turnover_control"],
                                    gamma=cfg_d["gamma"], seed=seed, epochs=epochs)
                pol, h = T.train_fold(Xtr, Ptr, Ctr, Vtr, cfg, mask=Mtr)
                # Rank on the HELD-OUT range, per the pre-registered rule. Ranking on the
                # training range selects for overfitting and contradicts T031a.
                ev = T.evaluate(pol, Xte, Pte, Cte, Vte, mask=Mte)
                res = {"status": "ok",
                       "test_log_return": ev["log_return"],
                       "test_return_pct": ev["return_pct"],
                       "test_mean_exposure": ev["mean_exposure"],
                       "test_turnover": ev["turnover"],
                       "test_flat_share": ev["flat_share"],
                       "train_log_return": h["total_log_return"],
                       "train_turnover": h["turnover"],
                       "final_loss": h["final_loss"],
                       "seconds": round(time.time() - t0, 1)}
                results[key].append(ev["log_return"])
            except Exception as e:                        # abandoned runs stay in the log
                trials.abandon(tid, f"{type(e).__name__}: {str(e)[:200]}", log)
                print(f"    {tid}: ABANDONED {type(e).__name__}", flush=True)
                continue
            trials.complete(tid, res, log)
            print(f"    {tid}: TEST log_ret={res['test_log_return']:+.4f} "
                  f"(train {res['train_log_return']:+.4f}) "
                  f"mean_w={res['test_mean_exposure']:.3f} ({res['seconds']}s)", flush=True)
    return {"fold": fold_id, "pairs": pairs, "results": results}


def select(results: dict[str, list[float]]) -> dict:
    """T032a + T033: resolvability first, then the pre-registered carry-forward."""
    iqms = {k: S.iqm(v) for k, v in results.items() if v}
    ranked = sorted(iqms.items(), key=lambda kv: -kv[1])
    out: dict = {"iqm_by_config": iqms, "ranked": ranked}
    if len(ranked) < 3:
        out["resolved"] = False
        out["reason"] = "fewer than 3 configurations completed"
    else:
        top, third = ranked[0][0], ranked[2][0]
        diff = np.array(results[top]) - np.array(results[third][:len(results[top])])
        lo, hi = S.bootstrap_ci(diff)
        out["resolvability"] = {"rank1": top, "rank3": third,
                                "iqm_diff": float(iqms[top] - iqms[third]),
                                "ci95": [lo, hi]}
        out["resolved"] = bool(lo > 0 or hi < 0)
    if out["resolved"]:
        out["carried"] = [ranked[0][0], ranked[1][0]]
        out["path"] = "rank (screen resolved)"
    else:
        best = {}
        for k, v in iqms.items():
            obj = k.split("-")[0]
            if obj not in best or v > iqms[best[obj]]:
                best[obj] = k
        out["carried"] = sorted(best.values())
        out["path"] = "pre-registered fallback (screen unresolved): best per objective"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="f08")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--pairs", default=None, help="comma-separated; default = fold universe")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    pairs = args.pairs.split(",") if args.pairs else None
    seeds = tuple(int(s) for s in args.seeds.split(","))
    r = run(args.fold, pairs, args.epochs, seeds)
    sel = select(r["results"])
    print("\n" + json.dumps(sel, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
