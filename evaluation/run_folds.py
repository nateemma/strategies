"""Train and evaluate each walk-forward fold, in order (FR-003..FR-005, FR-016).

Per fold: generate the fold class and its params -> train -> VERIFY THE ARTEFACT -> evaluate on the
fold's own out-of-sample range via run_arm.py -> record.

Three hazards this module exists to make impossible:

1. `BaseNNStrategy.get_model_path` keys off `self.__class__.__name__`. Folds sharing a class
   overwrite each other's weights, and a class with no model of its own TRAINS INSIDE ITS OWN
   BACKTEST -- fitting the evaluation window, silently, with returncode 0.
2. `scripts/test_strat.sh` returns 0 even when training crashed. The only reliable check is that
   `saved_data/<CLS>/*.safetensors` exists.
3. File params beat class params, so a fold class without its own .json silently inherits
   NNNCStrategy's ALL-GUARDS-OPEN defaults. The .json's strategy_name must equal the class name or
   freqtrade aborts with "Invalid parameter file provided".

Generated classes are written DIRECTLY into the family directory: run_arm.py resolves a fold class
to its family dir and symlinks only top-level *.py, so a subdirectory would never be found.

OPTIONAL plan key `thresholds`: `{fold_id: prediction_threshold}`. Absent -> unchanged
OPTIONAL plan key `guards` (feature 022): `{fold_id: {guard threshold: value}}`. Absent ->
unchanged. Requires a `trackable` trackability verdict -- see `_require_trackable_verdict`.
behaviour (every fold's .json carries the snapshot's own threshold, as before). Present ->
each fold's GENERATED .json carries the given value instead, so an arm whose model's
confidence distribution doesn't match the control's borrowed constant can be measured at a
matched entry rate rather than at zero trades. See evaluation/pick_threshold.py, which
measures the value that belongs here; this file only applies it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# parents: [0]=evaluation [1]=strategies [2]=user_data [3]=repo root (contains user_data/).
# Was parents[4] when this lived at regime/wf/; the 2026-09-14 promotion to evaluation/
# (a3c75bc) moved it one level shallower and the index was not decremented, so every
# path resolved one directory ABOVE the repo. Fixed 2026-10-07.
ROOT = Path(__file__).resolve().parents[3]
# parents: [0]=wf [1]=regime [2]=strategies [3]=user_data [4]=repo root
STRAT_DIR = ROOT / "user_data/strategies"
SAVED = STRAT_DIR / "saved_data"

#: Feature 030: a fold class pins the model key's training window and seed, so a backtest that
#: only LOADS the fold's model resolves the same key as the run that trained it. The template's
#: body was `pass` before 030 (test_run_folds checks the rest is byte-identical).
PIN_BLOCK = '''    MODEL_TRAIN_TIMERANGE = "{train_timerange}"
    MODEL_SEED = {seed}
'''

TEMPLATE = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""{cls} — walk-forward fold {fold_id} (feature 007). GENERATED, do not edit.

Train {train_start}..{train_end}   Test {test_start}..{test_end} ({test_days}d, OUT OF SAMPLE)

Its own class identity gives it its own saved_data/ directory. Two folds sharing a class would
overwrite each other's weights, and a class with no model of its own would train inside its own
backtest. Its .json is generated from the run's params SNAPSHOT, never re-read from the live
working file, so a mid-run edit cannot hand later folds a different entry gate.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from {base} import {base}


class {cls}({base}):
''' + PIN_BLOCK



def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


FOLDS_DIR = Path(__file__).resolve().parent / "folds"


def _require_trackable_verdict(plan: dict) -> None:
    """Feature 022 FR-006: refuse a per-fold-guard run unless trackability was DEMONSTRATED.

    Constitution Principle VII is non-negotiable: "for any proposed adaptive or regime-
    conditional parameter this gate is specific and mandatory: build the rolling-optimum series
    and test whether a trailing-best forecast beats a CONSTANT. A parameter whose optimum merely
    VARIES is not a candidate for adaptivity." Per-fold guard tuning is exactly such a scheme.

    Until this function existed the gate was a prose note in the feature's task list, which
    nothing prevented an operator or an agent from reading past -- and feature 022's own
    inertness check exists because "a guard that cannot fail is worthless". So the gate is code.

    BINDS ONLY ON A PLAN CARRYING A `guards` KEY. Every pre-existing plan has none and is
    returned untouched, which is what keeps this change inert for other features.

    FAILS CLOSED. A missing, unparseable or non-`trackable` verdict all refuse; only an
    explicit `"verdict": "trackable"` proceeds. Failing open on a truncated write would make
    the gate worthless precisely when it matters.
    """
    if not plan.get("guards"):
        return
    run_id = plan.get("run_id", "unknown")
    vf = FOLDS_DIR / f"trackability_{run_id}.json"
    if not vf.is_file():
        print(f"*** REFUSING to run: plan carries a `guards` key but no trackability verdict "
              f"at {vf}.\n    Feature 022 FR-006 and Constitution Principle VII require the "
              f"gate to PASS before per-fold guard tuning is evaluated. Run "
              f"evaluation/guard_trackability.py first.", flush=True)
        raise SystemExit(2)
    try:
        verdict = json.loads(vf.read_text()).get("verdict")
    except ValueError as exc:
        print(f"*** REFUSING to run: trackability verdict at {vf} is unparseable ({exc}). "
              f"Failing closed -- a per-fold guard run is not licensed by a broken verdict.",
              flush=True)
        raise SystemExit(2) from exc
    if verdict != "trackable":
        print(f"*** REFUSING to run: trackability verdict is {verdict!r}, not 'trackable'.\n"
              f"    FR-006: arm C MUST NOT run on a failed gate, and that is a COMPLETE "
              f"finding, not a blocked task -- per-fold guard tuning is not a candidate for "
              f"adaptivity. Write it up (feature 022 T036) rather than working around this.",
              flush=True)
        raise SystemExit(2)


def generate_fold(fold: dict, base: str, family: str, params: dict | None,
                  threshold: float | None = None,
                  guards: dict | None = None, seed: int = 1) -> tuple[Path, Path | None]:
    """Write the fold's class and its params file (FR-003; T006/T006a).

    `threshold`, when given, overrides `params["buy"]["prediction_threshold"]` on the
    in-memory deep copy only -- it is never written back to the params SNAPSHOT file, so the
    invariant that every fold's .json traces to the plan's recorded `params_sha` still holds;
    only the threshold figure (feature pick_threshold.py exists to choose, per fold, in
    place of one constant borrowed from a differently-calibrated model) differs from the
    snapshot. Requires `params` -- a fold with no params file has nowhere to put a threshold.

    `guards` (feature 022) behaves identically and for the same reason: a per-fold guard
    configuration chosen on that fold's OWN training window by `pick_guards.py` is merged into
    the in-memory copy only, never into the snapshot. `entry_enable_guards` is forced True
    regardless of what is passed, because that flag ALSO gates the `volume > 0` data-sanity
    check (`Framework/BaseStrategy.py:295`) and 34.26% of bars in this dataset have
    `volume <= 0`; guards are opened by neutralising thresholds, never by this flag.

    Absent, both arguments leave the output byte-identical -- pinned by
    `test_generate_fold_without_guards_matches_the_PRE_CHANGE_fixture`, whose fixture was
    captured before this argument existed.
    """
    cls = fold["strategy_class"]
    py = STRAT_DIR / family / f"{cls}.py"
    py.write_text(TEMPLATE.format(cls=cls, base=base, family=family, seed=int(seed), **fold))
    js = None
    if params is not None:
        js = STRAT_DIR / family / f"{cls}.json"
        d = json.loads(json.dumps(params))
        d["strategy_name"] = cls              # MUST match the class or freqtrade aborts
        if threshold is not None:
            d.setdefault("params", {}).setdefault("buy", {})["prediction_threshold"] = threshold
        if guards:
            buy = d.setdefault("params", {}).setdefault("buy", {})
            buy.update(guards)
            buy["entry_enable_guards"] = True       # FR-008 -- forced, never taken from input
        js.write_text(json.dumps(d, indent=4) + "\n")
    return py, js


def fold_has_model(fold: dict, base: str, config: str, seed: int) -> bool:
    """Feature 030 FR-017: does the fold's MODEL KEY have weights? The fold's class is `base`
    pinned to the fold's training window and seed (what generate_fold writes), resolved under
    the plan's config -- a variant that changes only post-model settings finds its source
    model's weights, and a missing model is found here, before any backtest could train one."""
    from evaluation.fingerprint import load_config
    from Framework import ModelKey as MK

    cls = _load_strategy_class(base)
    pinned = type(fold["strategy_class"], (cls,), {
        "MODEL_TRAIN_TIMERANGE": fold["train_timerange"], "MODEL_SEED": int(seed),
        "__module__": cls.__module__})
    cfg = Path(config)
    cfg = cfg if cfg.is_absolute() else ROOT / cfg
    return MK.locate_for(pinned, load_config(cfg), fold["strategy_class"]).kind != "absent"


def model_exists(cls: str) -> bool:
    """FR-004 — the ONLY reliable training check; the exit code lies."""
    d = SAVED / cls
    return d.is_dir() and any(d.glob("*.safetensors"))


def model_key(fold: dict) -> str:
    """The saved_data/ directory a fold's model lives in.

    Normally the fold's own class (build() sets model_dir to saved_data/<class>). A derived plan
    may borrow another arm's trained weights (feature 023: NNNC_TriggerGate folds run on
    NNNC_ARO_fNN's models), so the check must follow `model_dir`, not the class -- checking the
    class would find nothing and TRAIN a new model inside the twin experiment.
    """
    md = fold.get("model_dir")
    return Path(md).name if md else fold["strategy_class"]


def _load_strategy_class(name: str):
    from evaluation.fingerprint import load_strategy_class

    return load_strategy_class(name)


def _trigger_runnable(name: str | None) -> tuple[bool, str]:
    """Causality preflight (feature 023): a no-model arm may run only a measured-causal trigger.

    FAILS CLOSED: no checker, no verdict file, or a non-PASS/stale row all refuse.
    """
    try:
        from research.outcomes.causality import check_runnable
    except Exception as exc:  # noqa: BLE001
        return False, f"causality checker unavailable ({type(exc).__name__}: {exc})"
    return check_runnable(name)


def train_fold(fold: dict, family: str, seed: int, config: str) -> bool:
    cls = fold["strategy_class"]
    log = Path(f"/tmp/wf_train_{cls}.log")
    env_prefix = f'PATH="{ROOT}/.venv/bin:$PATH" MLX_SEED={seed}'
    cmd = (f'cd {ROOT} && {env_prefix} zsh user_data/strategies/scripts/test_strat.sh '
           f'-c {config} -f -t {fold["train_timerange"]} {family} {cls} > {log} 2>&1')
    subprocess.run(["zsh", "-c", cmd], check=False)     # exit code deliberately ignored
    return model_exists(cls)


def evaluate_fold(fold: dict, config: str) -> int:
    """Evaluate a fold on its own out-of-sample range via run_arm.py (FR-005).

    `run_arm.py --timerange` takes an arbitrary literal range, so a fold is passed straight
    through with `--label` carrying the fold id into the arm record's naming slot.

    HISTORY, because the old shape is the reason `--timerange` exists. run_arm.py used to
    require `--window`, whose `choices=` came from `regime.regimedata.WINDOWS`. Adding fold
    keys to that dict on disk was NOT safe -- `regime/phase_c_trackability.py` does
    `for w in WINDOWS`, so 15 extra keys would silently change another tool. Each fold
    therefore ran in its own subprocess that injected its window into the IN-MEMORY dict
    before importing run_arm. That hack worked and was honest about itself, but it welded
    this harness to `regime/` more tightly than anything else here, which is why the flag was
    added before the harness was promoted out.

    One strategy per process is still preserved (research R10): each fold is its own
    subprocess.
    """
    start, end = fold["test_timerange"].split("-")
    cmd = (f'cd {ROOT} && PYTHONPATH=user_data/strategies '
           f'FT_MODEL_EVAL_STRICT=1 .venv/bin/python -m regime.run_arm '
           f'--strategy {fold["strategy_class"]} '
           f'--timerange {start}-{end} --label {fold["fold_id"].upper()} '
           f'--tag {fold["arm_tag"]} --config {config} '
           f'> /tmp/wf_arm_{fold["arm_tag"]}.log 2>&1')
    return subprocess.run(["zsh", "-c", cmd], check=False).returncode


def arm_record(fold: dict) -> dict | None:
    """`{arm_tag}_{FOLD}.json` by EXACT name -- the label `evaluate_fold` passes to run_arm.
    A `{tag}_*.json` glob also matched seed-suffixed siblings in directory order (023 R6)."""
    p = STRAT_DIR / "regime/arms" / f"{fold['arm_tag']}_{fold['fold_id'].upper()}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--folds", help="comma-separated fold_ids; default all")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate nothing, train nothing; print the sequence and identities")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())
    if not plan.get("validated"):
        print("REFUSED: plan is not validated"); return 1
    family, base = plan["family"], plan["base_strategy"]
    no_model = bool(plan.get("no_model"))
    if no_model:
        # Feature 023. Re-validate (a hand-edited plan could add thresholds), then two
        # BLOCKING preflights before anything runs: the class must be model-free, and its
        # trigger must be measured causal. Both are refusals of the whole plan, never per fold.
        from evaluation.foldplan import validate_plan

        errors = validate_plan(plan)
        if errors:
            print("REFUSED: " + "; ".join(errors)); return 1
        cls_obj = _load_strategy_class(base)
        cfg = getattr(cls_obj, "strategy_config", None)
        if cfg is None or getattr(cfg, "needs_training", True):
            print(f"REFUSED: {base} is not model-free (strategy_config.needs_training is not "
                  f"False) -- a no_model plan would train inside its own backtest"); return 1
        # 024: a combined arm declares TRIGGERS; every ENABLED one must pass (the strategy
        # refuses the whole arm on any failure, so the preflight must agree with it).
        names = tuple(
            getattr(cls_obj, "TRIGGERS", None) or (getattr(cls_obj, "TRIGGER_NAME", None),)
        )
        disabled = tuple(getattr(cls_obj, "DISABLED_TRIGGERS", ()) or ())
        names = tuple(t for t in names if t not in disabled)
        for name in names or (None,):
            ok, why = _trigger_runnable(name)
            if not ok:
                print(f"REFUSED: {base} trigger {name!r} is not runnable: {why}"); return 1
    params = None
    if plan.get("params_snapshot"):
        snap = Path(plan["params_snapshot"])
        # BLOCKING. The plan records the sha of the params it was built from; if the file
        # on disk no longer hashes to it, something overwrote the snapshot and this run
        # would evaluate a different entry gate than the plan describes. That happened
        # on 2026-10-07 (two plans, same run_id, shared snapshot path) and cost a full
        # 15-fold run, because nothing checked.
        blob = snap.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()[:16]
        expected = plan.get("params_sha")
        if expected and actual != expected:
            print(f"ABORT: params snapshot {snap.name} hashes to {actual}, but the plan\n"
                  f"       was built from {expected}. The snapshot was overwritten.\n"
                  f"       Rebuild the plan with foldplan.py before running.", file=sys.stderr)
            return 2
        params = json.loads(blob)

    folds = plan["folds"]
    if a.folds:
        want = set(a.folds.split(","))
        folds = [f for f in folds if f["fold_id"] in want]

    # OPTIONAL per-fold threshold override: {fold_id: prediction_threshold}. Absent (the
    # default) -> behaviour is byte-identical to before this key existed. Keyed by fold_id,
    # so it survives the seed-namespacing rewrite below unchanged. See pick_threshold.py,
    # which measures the value that belongs here; this file only applies it.
    _require_trackable_verdict(plan)        # FR-006 -- before anything is generated or run
    thresholds = plan.get("thresholds", {})
    guards_map = plan.get("guards", {})

    # SEED NAMESPACING. get_model_path keys off the class name, so running a second seed against
    # the SAME class would silently overwrite the first seed's weights -- the same collision the
    # plan validator forbids between folds, which it cannot see across runs. Seed 1 keeps the
    # canonical identity; any other seed gets its own class, model dir and arm tag.
    if a.seed != 1 and no_model:
        # Nothing is trained, so only the record needs its own name.
        folds = [dict(f, arm_tag=f"{f['arm_tag']}_s{a.seed}") for f in folds]
    elif a.seed != 1:
        folds = [dict(f,
                      strategy_class=f"{f['strategy_class']}_s{a.seed}",
                      model_dir=f"{f['model_dir']}_s{a.seed}",
                      arm_tag=f"{f['arm_tag']}_s{a.seed}") for f in folds]

    if a.dry_run:
        print(f"DRY RUN — {len(folds)} folds, family {family}, base {base}, seed {a.seed}")
        print(f"params snapshot: {plan.get('params_snapshot')} sha={plan.get('params_sha')}")
        seen_c, seen_d = set(), set()
        for f in folds:
            c, d = f["strategy_class"], f["model_dir"]
            dup = ("  *** COLLISION" if not no_model and (c in seen_c or d in seen_d)
                   else "")
            seen_c.add(c); seen_d.add(d)
            print(f"  {f['fold_id']}  train {f['train_timerange']}  test {f['test_timerange']}"
                  f"  -> {STRAT_DIR/family/(c+'.py')}{dup}")
        print(f"\nunique classes {len(seen_c)}/{len(folds)}, unique model dirs {len(seen_d)}/{len(folds)}")
        print("params keys that would be written to every fold .json:",
              sorted((params or {}).get("params", {}).get("buy", {})) or "(none — base has no .json)")
        if no_model:
            return 0
        return 0 if (len(seen_c) == len(folds) and len(seen_d) == len(folds)) else 1

    t0 = time.time()
    results = []
    for f in folds:
        cls = f["fold_id"], f["strategy_class"]
        rec = arm_record(f)
        # `not rec.get("error")` is load-bearing: run_arm.py writes a record with
        # returncode 0 AND an `error` when the backtest exits clean but produces no
        # new result file (its stale-result guard). Resuming on returncode alone
        # treated such a fold as evaluated and skipped it permanently -- exactly the
        # failure mode FR-012 exists to prevent, one layer up.
        if (rec and rec.get("returncode") == 0 and not rec.get("error")
                and (no_model or fold_has_model(f, base, plan["config"], a.seed))):
            _log(f"{f['fold_id']}: already evaluated — skipping (resumable)")
            results.append({**f, "status": "evaluated", "n_trades": rec.get("n_trades")})
            continue
        if no_model:
            _log(f"{f['fold_id']}: no model -- evaluating {f['strategy_class']} on "
                 f"{f['test_timerange']}")
            rc = evaluate_fold(f, plan["config"])
            rec = arm_record(f)
            n = rec.get("n_trades") if rec else None
            status = ("evaluated" if rc == 0 and rec and rec.get("returncode") == 0
                      and not rec.get("error") else "failed")
            if status == "evaluated" and not n:
                status = "zero_trade"
            _log(f"{f['fold_id']}: {status}  n_trades={n}")
            results.append({**f, "status": status, "model_artifact_verified": None,
                            "n_trades": n, "arm_returncode": rc})
            continue
        _log(f"{f['fold_id']}: generating {f['strategy_class']}")
        generate_fold(f, base, family, params, threshold=thresholds.get(f["fold_id"]),
                      guards=guards_map.get(f["fold_id"]), seed=a.seed)
        has_model = fold_has_model(f, base, plan["config"], a.seed)
        if not has_model and plan.get("model_source"):
            # Borrowed weights (feature 023) must already exist. Training here would fit a NEW
            # model under this class's name -- a different experiment -- so fail the fold.
            _log(f"{f['fold_id']}: *** borrowed model {model_key(f)} is missing -- refusing to "
                 f"train a replacement")
            results.append({**f, "status": "failed", "model_artifact_verified": False})
            continue
        if has_model:
            # Resume at a finer grain: a trained artefact from an earlier attempt is reused rather
            # than retrained. Saves ~12 min per fold when only the evaluation needs re-running.
            _log(f"{f['fold_id']}: model already present — skipping training")
        else:
            _log(f"{f['fold_id']}: training on {f['train_timerange']}")
            train_fold(f, family, a.seed, plan["config"])
            if not fold_has_model(f, base, plan["config"], a.seed):
                _log(f"{f['fold_id']}: *** NO MODEL ARTEFACT — fold FAILED "
                     f"(exit code is not evidence)")
                results.append({**f, "status": "failed", "model_artifact_verified": False})
                continue
        _log(f"{f['fold_id']}: model OK; evaluating {f['test_timerange']}")
        rc = evaluate_fold(f, plan["config"])
        rec = arm_record(f)
        n = rec.get("n_trades") if rec else None
        status = ("evaluated" if rc == 0 and rec and rec.get("returncode") == 0 else "failed")
        if status == "evaluated" and not n:
            status = "zero_trade"            # FR-012: zero-trade, never zero-return
        _log(f"{f['fold_id']}: {status}  n_trades={n}")
        results.append({**f, "status": status, "model_artifact_verified": True,
                        "n_trades": n, "arm_returncode": rc})

    out = STRAT_DIR / f"evaluation/folds/results_{plan['run_id']}_seed{a.seed}.json"
    elapsed = (time.time() - t0) / 60
    complete = sum(1 for r in results if r["status"] in ("evaluated", "zero_trade"))
    out.write_text(json.dumps({
        "run_id": plan["run_id"], "seed": a.seed, "params_sha": plan.get("params_sha"),
        "elapsed_minutes": round(elapsed, 1),
        "folds_planned": len(plan["folds"]), "folds_complete": complete,
        "status": "complete" if complete == len(plan["folds"]) else "partial",
        "folds": results}, indent=2) + "\n")
    _log(f"DONE {complete}/{len(plan['folds'])} folds in {elapsed:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
