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
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # repo root (contains user_data/)
# parents: [0]=wf [1]=regime [2]=strategies [3]=user_data [4]=repo root
STRAT_DIR = ROOT / "user_data/strategies"
SAVED = STRAT_DIR / "saved_data"

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
    pass
'''


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def generate_fold(fold: dict, base: str, family: str, params: dict | None) -> tuple[Path, Path | None]:
    """Write the fold's class and its params file (FR-003; T006/T006a)."""
    cls = fold["strategy_class"]
    py = STRAT_DIR / family / f"{cls}.py"
    py.write_text(TEMPLATE.format(cls=cls, base=base, family=family, **fold))
    js = None
    if params is not None:
        js = STRAT_DIR / family / f"{cls}.json"
        d = json.loads(json.dumps(params))
        d["strategy_name"] = cls              # MUST match the class or freqtrade aborts
        js.write_text(json.dumps(d, indent=4) + "\n")
    return py, js


def model_exists(cls: str) -> bool:
    """FR-004 — the ONLY reliable training check; the exit code lies."""
    d = SAVED / cls
    return d.is_dir() and any(d.glob("*.safetensors"))


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

    run_arm.py has NO --timerange flag: --window is required and its choices come from
    regime.regimedata.WINDOWS. Adding fold keys to that dict on disk is NOT safe --
    regime/phase_c_trackability.py does `for w in WINDOWS`, so 15 extra keys would silently change
    another tool. Instead each fold runs in its OWN subprocess that injects its window into the
    in-memory dict before importing run_arm. regimedata.py on disk is never modified, run_arm.py is
    never modified (FR-018), and one-strategy-per-process is preserved (research R10).
    """
    wname = fold["fold_id"].upper()
    start, end = fold["test_timerange"].split("-")
    inject = (
        "import sys; "
        "import regime.regimedata as rd; "
        f"rd.WINDOWS['{wname}']=('{start}','{end}'); "
        "import regime.run_arm as ra; "
        f"sys.argv=['run_arm.py','--strategy','{fold['strategy_class']}',"
        f"'--window','{wname}','--tag','{fold['arm_tag']}','--config','{config}']; "
        "raise SystemExit(ra.main())"
    )
    cmd = (f'cd {ROOT} && PYTHONPATH=user_data/strategies .venv/bin/python -c "{inject}" '
           f'> /tmp/wf_arm_{fold["strategy_class"]}.log 2>&1')
    return subprocess.run(["zsh", "-c", cmd], check=False).returncode


def arm_record(fold: dict) -> dict | None:
    for p in (STRAT_DIR / "regime/arms").glob(f"{fold['arm_tag']}_*.json"):
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
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
    params = (json.loads(Path(plan["params_snapshot"]).read_text())
              if plan.get("params_snapshot") else None)

    folds = plan["folds"]
    if a.folds:
        want = set(a.folds.split(","))
        folds = [f for f in folds if f["fold_id"] in want]

    # SEED NAMESPACING. get_model_path keys off the class name, so running a second seed against
    # the SAME class would silently overwrite the first seed's weights -- the same collision the
    # plan validator forbids between folds, which it cannot see across runs. Seed 1 keeps the
    # canonical identity; any other seed gets its own class, model dir and arm tag.
    if a.seed != 1:
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
            dup = "  *** COLLISION" if (c in seen_c or d in seen_d) else ""
            seen_c.add(c); seen_d.add(d)
            print(f"  {f['fold_id']}  train {f['train_timerange']}  test {f['test_timerange']}"
                  f"  -> {STRAT_DIR/family/(c+'.py')}{dup}")
        print(f"\nunique classes {len(seen_c)}/{len(folds)}, unique model dirs {len(seen_d)}/{len(folds)}")
        print("params keys that would be written to every fold .json:",
              sorted((params or {}).get("params", {}).get("buy", {})) or "(none — base has no .json)")
        return 0 if (len(seen_c) == len(folds) and len(seen_d) == len(folds)) else 1

    t0 = time.time()
    results = []
    for f in folds:
        cls = f["fold_id"], f["strategy_class"]
        rec = arm_record(f)
        if rec and rec.get("returncode") == 0 and model_exists(f["strategy_class"]):
            _log(f"{f['fold_id']}: already evaluated — skipping (resumable)")
            results.append({**f, "status": "evaluated", "n_trades": rec.get("n_trades")})
            continue
        _log(f"{f['fold_id']}: generating {f['strategy_class']}")
        generate_fold(f, base, family, params)
        if model_exists(f["strategy_class"]):
            # Resume at a finer grain: a trained artefact from an earlier attempt is reused rather
            # than retrained. Saves ~12 min per fold when only the evaluation needs re-running.
            _log(f"{f['fold_id']}: model already present — skipping training")
        else:
            _log(f"{f['fold_id']}: training on {f['train_timerange']}")
            if not train_fold(f, family, a.seed, plan["config"]):
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

    out = STRAT_DIR / f"regime/wf/folds/results_{plan['run_id']}_seed{a.seed}.json"
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
