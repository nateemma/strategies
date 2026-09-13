"""The nested walk-forward veto model (T024-T031).

CAPACITY IS PINNED FOR THE FIRST FOLD, NOT THE LAST. f05 trains on 91 rows; f14 on 375.
Sizing the model for 375 would overfit 91 four times over before the design has warmed up,
and the early folds are where a nested result is decided. Every value below is fixed before
any run and recorded in the trial log.

ACCURACY IS NEVER REPORTED. The evaluation span's base win rate is 73.2%, so a model that
accepts every trade scores 0.732 and looks competent. AUC is threshold-free and
base-rate-insensitive; precision at the operating point says whether the VETOED trades were
actually losers, which is the quantity P&L follows from.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import metalabel as M                 # noqa: E402
from research.labelfree import stats as ST                    # noqa: E402
from research.labelfree import trials                         # noqa: E402

# --- PINNED. Sized for 91 rows. Changing any of these makes it a new configuration. ----
PARAMS = {
    "n_estimators": 60,
    "max_depth": 3,
    "num_leaves": 4,
    "min_child_samples": 10,      # ~11% of f05's training set
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}
FEATURE_ORDER = ("%predict_prob_buy", "%predict_prob_hold", "%predict_prob_sell",
                 "guard_metric", "rvol", "atr_pct", "bb_width", "adx", "volume", "close")


def matrix(rows: list[dict], idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[rows[i]["features"].get(c, np.nan) for c in FEATURE_ORDER] for i in idx],
                 dtype=float)
    y = np.array([rows[i]["target_win"] for i in idx], dtype=bool)
    return X, y


def fit_predict(Xtr, ytr, Xte, seed: int) -> np.ndarray:
    import lightgbm as lgb
    if len(np.unique(ytr)) < 2:
        # Degenerate training fold: every trade won (or lost). No model is learnable;
        # returning the base rate is honest, and the caller reports n_train alongside.
        return np.full(len(Xte), float(ytr.mean()))
    clf = lgb.LGBMClassifier(random_state=seed, **PARAMS)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    p, n = label.sum(), (~label).sum()
    if p == 0 or n == 0:
        return float("nan")
    return float((r[label].sum() - p * (p + 1) / 2) / (p * n))


def run_nested(rows: list[dict], assignment: list[dict], seed: int) -> dict:
    """One seed, all evaluation folds. Returns per-fold and pooled predictions."""
    per_fold, pooled_p, pooled_y, pooled_i = [], [], [], []
    for a in assignment:
        Xtr, ytr = matrix(rows, a["train_idx"])
        Xte, yte = matrix(rows, a["test_idx"])
        p = fit_predict(Xtr, ytr, Xte, seed)
        per_fold.append({"fold": a["eval_fold"], "n_train": a["n_train"],
                         "n_test": a["n_test"], "auc": auc(p, yte),
                         "mean_p": float(p.mean())})
        pooled_p += list(p); pooled_y += list(yte); pooled_i += list(a["test_idx"])
    P, Y = np.array(pooled_p), np.array(pooled_y, dtype=bool)
    return {"seed": seed, "per_fold": per_fold, "pooled_auc": auc(P, Y),
            "p": P.tolist(), "y": Y.tolist(), "idx": pooled_i}


def apply_threshold(P: np.ndarray, Y: np.ndarray, thr: float) -> dict:
    keep = P >= thr
    vetoed = ~keep
    return {"threshold": thr,
            "kept": int(keep.sum()), "vetoed": int(vetoed.sum()),
            "veto_rate": float(vetoed.mean()),
            # Precision at the operating point: of the trades we KEPT, how many won.
            "precision_kept": float(Y[keep].mean()) if keep.any() else float("nan"),
            # And of the ones we vetoed, how many would have lost (the veto's own hit rate).
            "veto_precision": float((~Y[vetoed]).mean()) if vetoed.any() else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="research/labelfree/metalabel/rows.jsonl")
    ap.add_argument("--assignment", default="research/labelfree/metalabel/assignment.json")
    ap.add_argument("--thresholds", default="0.4,0.5,0.6")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--out", default="research/labelfree/metalabel/veto.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-liveness", action="store_true",
                    help="NOT for normal use; only to bootstrap the probes themselves")
    a = ap.parse_args()

    # FR-012 / SC-014: no compute is spent on a sweep until every swept parameter is proven
    # to change what it claims to control. Four parameters in this study were configured,
    # logged, believed active and reaching nothing; 75+ runs were voided between them.
    if not a.skip_liveness:
        from research.labelfree.metalabel import liveness as LV
        LV.check_all()
        print()

    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    asg = json.loads(Path(a.assignment).read_text())
    seeds = [int(s) for s in a.seeds.split(",")]
    thrs = [float(t) for t in a.thresholds.split(",")]
    run_id = trials.new_run_id()

    print(f"NESTED VETO — run {run_id}, {len(seeds)} seeds x {len(thrs)} thresholds")
    print(f"  params: {PARAMS}\n")

    runs = []
    for seed in seeds:
        tid = f"metalabel_veto-s{seed}@{run_id}"
        trials.pre_register({"trial_id": tid, "run_id": run_id, "feature": "011",
                             "arm": "metalabel_veto", "seed": seed,
                             "n_features": len(FEATURE_ORDER), **PARAMS})
        if a.dry_run:
            continue
        t0 = time.time()
        r = run_nested(rows, asg, seed)
        r["seconds"] = round(time.time() - t0, 1)
        r["status"] = "ok"
        trials.complete(tid, {k: v for k, v in r.items() if k not in ("p", "y", "idx")})
        runs.append(r)
        print(f"  seed {seed}: pooled nested AUC={r['pooled_auc']:.4f} ({r['seconds']}s)")
    if a.dry_run:
        print("  dry run: configurations pre-registered, nothing trained")
        return 0

    aucs = [r["pooled_auc"] for r in runs]
    print(f"\n  POOLED NESTED AUC  IQM {ST.iqm(aucs):.4f}  "
          f"CI {tuple(round(x, 4) for x in ST.bootstrap_ci(aucs))}  "
          f"(in-sample reference 0.6344, SC-009 gate {M.AUC_KILL})")

    print(f"\n  per fold (IQM across seeds):")
    print(f"  {'fold':<7}{'n_train':>9}{'n_test':>8}{'AUC':>9}")
    for i, a_ in enumerate(asg):
        fa = [r["per_fold"][i]["auc"] for r in runs
              if not np.isnan(r["per_fold"][i]["auc"])]
        s = f"{ST.iqm(fa):.4f}" if fa else "n/a"
        print(f"  {a_['eval_fold']:<7}{a_['n_train']:>9}{a_['n_test']:>8}{s:>9}")

    print(f"\n  veto at each threshold (pooled, IQM across seeds):")
    print(f"  {'thr':>6}{'kept':>7}{'vetoed':>8}{'veto rate':>11}"
          f"{'prec kept':>11}{'veto prec':>11}")
    thr_rows = []
    for t in thrs:
        per = [apply_threshold(np.array(r["p"]), np.array(r["y"], dtype=bool), t)
               for r in runs]
        row = {"threshold": t,
               "kept": ST.iqm([x["kept"] for x in per]),
               "vetoed": ST.iqm([x["vetoed"] for x in per]),
               "veto_rate": ST.iqm([x["veto_rate"] for x in per]),
               "precision_kept": ST.iqm([x["precision_kept"] for x in per]),
               "veto_precision": ST.iqm([x["veto_precision"] for x in per
                                         if not np.isnan(x["veto_precision"])] or [np.nan])}
        thr_rows.append(row)
        print(f"  {t:>6.2f}{row['kept']:>7.1f}{row['vetoed']:>8.1f}"
              f"{row['veto_rate']:>11.3f}{row['precision_kept']:>11.4f}"
              f"{row['veto_precision']:>11.4f}")

    inert = [r["threshold"] for r in thr_rows if r["vetoed"] < 1.0]
    if inert:
        print(f"\n  NOTE: thresholds {inert} veto fewer than one trade. A threshold that "
              f"changes nothing\n  is INERT, and its result says nothing about the "
              f"hypothesis (Principle IV).")

    verdict = ST.iqm(aucs) >= M.AUC_KILL
    print(f"\n  SC-009: nested AUC {ST.iqm(aucs):.4f} vs gate {M.AUC_KILL} -> "
          f"{'PASS — continue to the P&L consequence' if verdict else 'KILL'}")
    Path(a.out).write_text(json.dumps(
        {"run_id": run_id, "params": PARAMS, "features": list(FEATURE_ORDER),
         "seeds": seeds, "pooled_auc_iqm": ST.iqm(aucs),
         "pooled_auc_ci": list(ST.bootstrap_ci(aucs)),
         "thresholds": thr_rows, "runs": runs}, indent=2))
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
