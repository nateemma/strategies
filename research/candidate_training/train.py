"""Train the candidate-only folds, one at a time (032 T013, research R7).

`run_folds` trains and evaluates in one loop and needs each fold's threshold BEFORE it evaluates,
but a candidate-only threshold can only be picked from the trained model. So this driver trains
only: per fold, write the fold class from the params snapshot (no threshold), train if the fold's
model key has no weights, verify, and print the fold's row_filter.json. Resumable; refuses to start
while another backtest, training or harness process runs (one chain at a time). The live dry-run
bot (`freqtrade trade`) is not a chain and is ignored.

    PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.train --model ARO [--folds f00]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "evaluation"))

FOLDS = STRAT / "evaluation" / "folds"
PLANS = {"ARO": "co_aro", "GBR": "co_gbr", "AROV": "cs_arov"}  # AROV: research/aro_volume
BUSY = ("freqtrade backtesting", "freqtrade hyperopt", "test_strat.sh", "evaluation.run_folds",
        "regime.run_arm", "research.candidate_training.train ")


def other_chains(ps_lines: list[str], own_pid: int) -> list[str]:
    """Command lines of other training/backtest/harness processes (not the dry-run bot)."""
    out = []
    for line in ps_lines:
        pid, _, cmd = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) == own_pid:
            continue
        tok = cmd.split()
        if len(tok) > 1 and Path(tok[0]).name in ("zsh", "bash", "sh") and tok[1] == "-c":
            continue  # a wrapper shell (e.g. the one that launched us); its child is what counts
        if any(b in cmd for b in BUSY):
            out.append(cmd)
    return out


def _ps() -> list[str]:
    return subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                          text=True, check=True).stdout.splitlines()


def _location(fold: dict, base: str, config: str, seed: int):
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cls = load_strategy_class(base)
    pinned = type(fold["strategy_class"], (cls,), {
        "MODEL_TRAIN_TIMERANGE": fold["train_timerange"], "MODEL_SEED": int(seed),
        "__module__": cls.__module__})
    cfg = Path(config)
    cfg = cfg if cfg.is_absolute() else STRAT.parents[1] / cfg
    return MK.locate_for(pinned, load_config(cfg), fold["strategy_class"])


def run(model: str, folds: set[str] | None, seed: int = 1, *, rf=None, ps=_ps,
        own_pid: int | None = None) -> list[dict]:
    import os

    rf = rf or __import__("run_folds")
    busy = other_chains(ps(), own_pid if own_pid is not None else os.getpid())
    if busy:
        raise SystemExit("REFUSED: another chain is running:\n  " + "\n  ".join(busy))
    plan = json.loads((FOLDS / f"plan_{PLANS[model]}.json").read_text())
    snap = Path(plan["params_snapshot"])
    snap = snap if snap.is_absolute() else STRAT.parents[1] / snap
    blob = snap.read_bytes()
    if plan.get("params_sha") and hashlib.sha256(blob).hexdigest()[:16] != plan["params_sha"]:
        raise SystemExit(f"ABORT: params snapshot {snap.name} does not hash to the plan's params_sha")
    params = json.loads(blob)
    out = []
    for f in plan["folds"]:
        if folds and f["fold_id"] not in folds:
            continue
        rf.generate_fold(f, plan["base_strategy"], plan["family"], params, seed=seed)
        if rf.fold_has_model(f, plan["base_strategy"], plan["config"], seed):
            print(f"{f['fold_id']}: model present -- skipping", flush=True)
            out.append({"fold": f["fold_id"], "trained": False})
            continue
        print(f"{f['fold_id']}: training {f['strategy_class']} on {f['train_timerange']}", flush=True)
        rf.train_fold(f, plan["family"], seed, plan["config"])
        if not rf.fold_has_model(f, plan["base_strategy"], plan["config"], seed):
            raise SystemExit(f"{f['fold_id']}: *** NO MODEL ARTEFACT after training -- stopping")
        report = Path(_location(f, plan["base_strategy"], plan["config"], seed).path).parent \
            / "row_filter.json"
        rep = json.loads(report.read_text()) if report.is_file() else None
        print(f"{f['fold_id']}: trained; row_filter {json.dumps(rep)}", flush=True)
        out.append({"fold": f["fold_id"], "trained": True, "row_filter": rep})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(PLANS))
    ap.add_argument("--folds", help="comma-separated fold ids; default all")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.model, set(a.folds.split(",")) if a.folds else None, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
