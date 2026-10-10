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
# Absolute anchors for `derive` (feature 023), which must work from any cwd -- tests run from
# user_data/strategies, the harness from the repo root. Plans keep storing ROOT-relative paths.
ROOT_ABS = Path(__file__).resolve().parents[3]
FOLDS_DIR_ABS = ROOT_ABS / "user_data/strategies/evaluation/folds"


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


def validate(folds: list[dict], no_model: bool = False) -> list[str]:
    """Return a list of blocking errors. Empty means the plan may execute (FR-002, SC-003).

    `no_model` (feature 023): a model-free arm legitimately runs the SAME class with NO model
    dir in every fold, so the two identity-collision checks -- which exist to stop folds
    overwriting each other's weights -- do not apply. The exemption holds only when every fold
    really has no model dir; a no-model plan that names one is still an error.
    """
    errors: list[str] = []
    if not folds:
        return ["plan is empty"]
    if no_model:
        for f in folds:
            if f.get("model_dir") is not None:
                errors.append(f"{f['fold_id']}: no_model plan has model_dir {f['model_dir']!r}")
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
        if not no_model and f["strategy_class"] in seen_cls:
            errors.append(f"{fid}: strategy_class {f['strategy_class']} already used by "
                          f"{seen_cls[f['strategy_class']]}")
        seen_cls[f["strategy_class"]] = fid
        if not no_model and f["model_dir"] in seen_dir:
            errors.append(f"{fid}: model_dir collides with {seen_dir[f['model_dir']]}")
        seen_dir[f["model_dir"]] = fid
        if prev_test_start is not None and te_s <= prev_test_start:
            errors.append(f"{fid}: test_start {te_s} does not advance")
        prev_test_start = te_s
    return errors


def validate_plan(plan: dict) -> list[str]:
    """Plan-level validation: the fold rules plus the no-model rules (feature 023 C2)."""
    no_model = bool(plan.get("no_model"))
    errors = validate(plan.get("folds", []), no_model=no_model)
    if no_model and plan.get("thresholds"):
        errors.append("no_model plan carries `thresholds`: a model-free arm has no prediction "
                      "threshold, so they would be silently dropped")
    return errors


def derive(src_plan: dict, family: str, base: str, tag: str, *,
           model_source: str | None = None, class_prefix: str | None = None,
           params_file: Path | None = None, snapshot_dir: Path | None = None,
           run_id: str | None = None) -> dict:
    """A new plan whose fold WINDOWS are copied verbatim from `src_plan` (feature 023 C2).

    Rebuilding from dates would need `--data-end` and friends matched by hand; copying makes the
    windows identical by construction, which is what makes the new arm comparable fold-for-fold.

    Two shapes:
      * no model (`model_source` None): every fold runs `base` itself, `model_dir` null,
        `no_model: true`, no params snapshot, no thresholds.
      * NNNC on borrowed models (`model_source` e.g. "NNNC_ARO"): per-fold classes
        `{class_prefix}_fNN` whose model dir is `{model_source}_fNN` -- the source arm's trained
        weights, so nothing is trained. The base's params JSON is snapshotted as `build` does.
    """
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    no_model = model_source is None
    keep = ("fold_id", "train_start", "train_end", "test_start", "test_end",
            "train_timerange", "test_timerange", "test_days")
    folds = []
    for f in src_plan["folds"]:
        g = {k: f[k] for k in keep}
        fid = f["fold_id"]
        if no_model:
            g["strategy_class"] = base
            g["model_dir"] = None
        else:
            if not class_prefix:
                raise ValueError("class_prefix is required when model_source is given")
            g["strategy_class"] = f"{class_prefix}_{fid}"
            g["model_dir"] = f"user_data/strategies/saved_data/{model_source}_{fid}"
        g["arm_tag"] = f"{tag}_{fid}"
        g["status"] = "planned"
        folds.append(g)

    snap = sha = None
    if not no_model:
        src = params_file or (ROOT_ABS / "user_data/strategies" / family / f"{base}.json")
        if not Path(src).is_file():
            raise FileNotFoundError(f"{src}: an NNNC derive needs the base's params JSON -- "
                                    f"without it freqtrade runs the class defaults")
        snap, sha = snapshot_params(Path(src), run_id, out_dir=snapshot_dir or FOLDS_DIR_ABS)
        try:
            snap = Path(snap).resolve().relative_to(ROOT_ABS)
        except ValueError:
            pass

    plan = {
        "run_id": run_id,
        "family": family,
        "base_strategy": base,
        "derived_from": src_plan.get("run_id"),
        "no_model": no_model,
        "model_source": model_source,
        "anchored": src_plan.get("anchored", True),
        "min_train_days": src_plan.get("min_train_days"),
        "test_days": src_plan.get("test_days"),
        "data_start": src_plan.get("data_start"),
        "data_end": src_plan.get("data_end"),
        "config": src_plan["config"],
        "params_snapshot": str(snap) if snap else None,
        "params_sha": sha,
        "folds": folds,
    }
    errors = validate_plan(plan)
    plan["validated"] = not errors
    plan["validation_errors"] = errors
    return plan


def derive_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="foldplan derive", description=derive.__doc__)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-source")
    ap.add_argument("--class-prefix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src_plan = json.loads(Path(a.src).read_text())
    plan = derive(src_plan, a.family, a.base, a.tag, model_source=a.model_source,
                  class_prefix=a.class_prefix)
    if plan["validation_errors"]:
        print(f"REFUSED: {len(plan['validation_errors'])} blocking error(s)")
        for e in plan["validation_errors"]:
            print(f"  {e}")
        return 1
    Path(a.out).write_text(json.dumps(plan, indent=2) + "\n")
    print(f"VALIDATED: {len(plan['folds'])} folds derived from run {plan['derived_from']} "
          f"({'no model' if plan['no_model'] else 'models from ' + plan['model_source']})")
    print(f"written: {a.out}")
    return 0


def snapshot_params(src: Path, run_id: str,
                    out_dir: Path | None = None) -> tuple[Path, str] | tuple[None, None]:
    """Snapshot the strategy params file ONCE, at plan time (T006a).

    NNNC/NNNC_MLX.json is UNTRACKED working state that the user edits. Re-reading it per fold would
    let a mid-run edit hand later folds a different entry gate, making the pooled trade set
    silently heterogeneous with no error anywhere.
    """
    if not src.is_file():
        return None, None            # e.g. NNMT_MLX, which legitimately has no params file
    blob = src.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()[:16]
    # Keyed by the SOURCE FILE STEM as well as run_id. run_id is a timestamp to the
    # second, so two plans built in the same second (a for-loop over two arms) wrote to
    # the SAME path: the second overwrote the first, and the first plan then pointed at
    # the other arm's entry gate while still recording its own sha. Found 2026-10-07
    # after an arm silently ran the control's guards.
    dst = (out_dir or FOLDS_DIR) / f"params_{run_id}_{src.stem}.json"
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
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "derive":
        raise SystemExit(derive_main(_sys.argv[2:]))
    raise SystemExit(main())
