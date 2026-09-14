"""Build and VALIDATE a walk-forward fold plan (FR-001, FR-002).

Geometry is DECIDED in research R3 and is not re-opened here: ANCHORED (expanding) training, a
730-day minimum before the first test fold, and 90-day test folds. Production trains on all data
up to a cutoff, so an anchored window is the faithful analogue; a short fixed rolling window would
change the training regime and confound the instrument with the thing it measures.

Validation is BLOCKING. A plan that does not validate must never reach execution: a fold whose test
range touches its own training range is a lookahead bug that would silently invalidate every
downstream number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

STRAT_DIR = Path("user_data/strategies")
FOLDS_DIR = STRAT_DIR / "evaluation/folds"


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _ft(d: date) -> str:
    """freqtrade timerange format."""
    return d.strftime("%Y%m%d")


def build(family: str, data_start: str, data_end: str,
          min_train_days: int = 730, test_days: int = 90,
          min_final_days: int = 45) -> list[dict]:
    """Anchored expanding training; consecutive non-overlapping test folds."""
    ds, de = _d(data_start), _d(data_end)
    folds = []
    test_start = ds + timedelta(days=min_train_days)
    i = 0
    # A FINAL SHORTER FOLD IS KEPT (>= min_final_days). Dropping it to preserve tidy equal folds
    # discarded 85 days of genuine out-of-sample data and cost ~23 pooled trades — enough to miss
    # SC-002 (>=350) and push the estimated floor from 3.98pp to 4.11pp, past SC-001a. Folds are
    # pooled by TRADE, not averaged by fold, so unequal fold lengths are fine for the pooled
    # statistic; power.py reports each fold's length so the inter-fold correlation can account
    # for it.
    while test_start + timedelta(days=min_final_days) <= de:
        test_end = min(test_start + timedelta(days=test_days), de)
        cls = f"{family}_WF_f{i:02d}"
        folds.append({
            "fold_id": f"f{i:02d}",
            "train_start": ds.isoformat(),
            "train_end": (test_start - timedelta(days=1)).isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_timerange": f"{_ft(ds)}-{_ft(test_start)}",
            "test_timerange": f"{_ft(test_start)}-{_ft(test_end)}",
            "test_days": (test_end - test_start).days,
            "strategy_class": cls,
            "model_dir": f"user_data/strategies/saved_data/{cls}",
            "arm_tag": f"wf_{family.lower()}_f{i:02d}",
            "status": "planned",
        })
        test_start = test_end
        i += 1
    return folds


def validate(folds: list[dict]) -> list[str]:
    """Return a list of blocking errors. Empty means the plan may execute (FR-002, SC-003)."""
    errors: list[str] = []
    if not folds:
        return ["plan is empty"]
    seen_cls: dict[str, str] = {}
    seen_dir: dict[str, str] = {}
    prev_test_start = None
    for f in folds:
        fid = f["fold_id"]
        tr_s, tr_e = _d(f["train_start"]), _d(f["train_end"])
        te_s, te_e = _d(f["test_start"]), _d(f["test_end"])
        # THE critical rule: a fold may not be tested on data it was trained on.
        if te_s <= tr_e:
            errors.append(
                f"{fid}: test range starts {te_s} but training ends {tr_e} — "
                f"test range intersects its own training range")
        if te_e <= te_s:
            errors.append(f"{fid}: test_end {te_e} is not after test_start {te_s}")
        if tr_e <= tr_s:
            errors.append(f"{fid}: train_end {tr_e} is not after train_start {tr_s}")
        # Identity collisions would make folds overwrite each other's weights (research R4).
        if f["strategy_class"] in seen_cls:
            errors.append(f"{fid}: strategy_class {f['strategy_class']} already used by "
                          f"{seen_cls[f['strategy_class']]}")
        seen_cls[f["strategy_class"]] = fid
        if f["model_dir"] in seen_dir:
            errors.append(f"{fid}: model_dir collides with {seen_dir[f['model_dir']]}")
        seen_dir[f["model_dir"]] = fid
        if prev_test_start is not None and te_s <= prev_test_start:
            errors.append(f"{fid}: test_start {te_s} does not advance")
        prev_test_start = te_s
    return errors


def snapshot_params(src: Path, run_id: str) -> tuple[Path, str] | tuple[None, None]:
    """Snapshot the strategy params file ONCE, at plan time (T006a).

    NNNC/NNNC_MLX.json is UNTRACKED working state that the user edits. Re-reading it per fold would
    let a mid-run edit hand later folds a different entry gate, making the pooled trade set
    silently heterogeneous with no error anywhere.
    """
    if not src.is_file():
        return None, None            # e.g. NNMT_MLX, which legitimately has no params file
    blob = src.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()[:16]
    dst = FOLDS_DIR / f"params_{run_id}.json"
    dst.write_bytes(blob)
    return dst, sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="NNNC")
    ap.add_argument("--base-strategy", default="NNNC_MLX")
    ap.add_argument("--anchored", action="store_true", default=True)
    ap.add_argument("--min-train-days", type=int, default=730)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--data-start", default="2021-01-01")
    ap.add_argument("--data-end", default="2026-09-07")
    ap.add_argument("--min-final-days", type=int, default=45,
                    help="keep a final shorter fold if it has at least this many days")
    ap.add_argument("--config", default="user_data/strategies/config/config_wf.json",
                    help="config used for BOTH training and evaluation, so a fold's train and "
                         "test see the same universe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    folds = build(a.family, a.data_start, a.data_end, a.min_train_days, a.test_days,
                  a.min_final_days)
    errors = validate(folds)

    print(f"fold plan — family {a.family}, ANCHORED training, "
          f"min {a.min_train_days}d train, {a.test_days}d test")
    print(f"  data {a.data_start} .. {a.data_end}\n")
    print(f"  {'fold':<6}{'train':<26}{'test':<26}{'class'}")
    for f in folds:
        print(f"  {f['fold_id']:<6}{f['train_start']}..{f['train_end']:<13}"
              f"{f['test_start']}..{f['test_end']:<13}{f['strategy_class']:<14}"
              f"{f['test_days']}d")

    if errors:
        print(f"\nREFUSED: {len(errors)} blocking error(s)")
        for e in errors:
            print(f"  {e}")
        return 1

    src = STRAT_DIR / a.family / f"{a.base_strategy}.json"
    snap, sha = snapshot_params(src, run_id)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    plan = {
        "run_id": run_id,
        "family": a.family,
        "base_strategy": a.base_strategy,
        "anchored": True,
        "min_train_days": a.min_train_days,
        "test_days": a.test_days,
        "data_start": a.data_start,
        "data_end": a.data_end,
        "config": a.config,
        "params_snapshot": str(snap) if snap else None,
        "params_sha": sha,
        "validated": True,
        "validation_errors": [],
        "folds": folds,
    }
    Path(a.out).write_text(json.dumps(plan, indent=2) + "\n")
    test_days_total = sum(f["test_days"] for f in folds)
    print(f"\nVALIDATED: {len(folds)} folds, {test_days_total} test days "
          f"({test_days_total/365:.2f} yr-equivalent)")
    print(f"params snapshot: {snap}  sha={sha}" if snap
          else "params snapshot: NONE (base strategy has no .json — correct for NNMT)")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
