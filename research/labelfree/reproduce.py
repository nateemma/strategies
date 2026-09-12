"""Re-run a logged configuration in a FRESH process and assert it reproduces (T032b, SC-006).

Brief 01 planned this module and never built it, because its Phase 6 was unreached.

Reproducibility here is not a nicety. Every verdict in this study is a difference between
two numbers of order 0.1-1.0 log return; if a configuration does not re-emit the same
number from the same seed, no delta it participates in means anything. MLX seeds through
MLX_SEED, which is read at import time -- so the re-run MUST happen in a new interpreter,
not in the calling process. Doing it in-process would silently pass by reusing state.

Usage:
    python research/labelfree/reproduce.py --trial-id 'armC_auxvol-s1-f07@20260912T144503Z'
    python research/labelfree/reproduce.py --latest-run 20260912T144503Z
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.labelfree import trials            # noqa: E402

# Fields that define the run. A mismatch in any of these is a different configuration,
# not a reproduction failure, so they are rebuilt rather than compared.
CONFIG_KEYS = ("arm", "seed", "fold_id", "turnover_control", "gamma", "epochs",
               "head_lr", "encoder_lr", "blend_alpha", "label_threshold",
               "min_exposure", "participation_weight", "warm_start_ref")

# What reproduction means. Exact equality on floats across processes is not guaranteed by
# MLX on Metal (kernel scheduling can reorder reductions), so each metric carries the
# tolerance at which a difference would change a verdict. test_log_return's tolerance is
# two orders of magnitude below the smallest delta this study reports.
TOLERANCES = {"test_log_return": 1e-4, "test_mean_exposure": 1e-4,
              "test_flat_share": 1e-3, "test_turnover": 1e-2}


def find(trial_id: str) -> tuple[dict, dict]:
    """(pre-registered config, logged result) for one trial id."""
    pre = res = None
    for r in trials.read():
        if r.get("trial_id") != trial_id:
            continue
        if "written_at" in r:
            pre = r
        if "completed_at" in r:
            res = r.get("result")
    if pre is None:
        raise SystemExit(f"no pre-registered record for {trial_id}")
    if res is None:
        raise SystemExit(f"{trial_id} has no completed result to reproduce")
    return pre, res


def rerun(cfg: dict) -> dict:
    """Re-train in a FRESH interpreter so MLX_SEED is read at import, and return metrics."""
    spec = {k: cfg[k] for k in CONFIG_KEYS if cfg.get(k) is not None}
    fold_id = spec.pop("fold_id", "f08")
    seed = spec.pop("seed")
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(ROOT)!r});"
        "from research.labelfree.arms import ArmConfig, load_panel, train_arm;"
        "from research.labelfree import folds as F;"
        f"spec={spec!r};fold_id={fold_id!r};seed={seed!r};"
        "fold={f.fold_id:f for f in F.folds()}[fold_id];"
        "pairs=list(fold.pair_universe);"
        "thr=spec.get('label_threshold');"
        "tr=load_panel(fold,pairs,True,thr);te=load_panel(fold,pairs,False,thr);"
        "cfg=ArmConfig(seed=seed,fold_id=fold_id,**spec);"
        "r=train_arm(cfg,tr,te);"
        "print('___RESULT___'+json.dumps(r))"
    )
    env = dict(os.environ, MLX_SEED=str(seed), PYTHONPATH=str(ROOT))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"re-run failed:\n{p.stderr[-2000:]}")
    for line in p.stdout.splitlines():
        if line.startswith("___RESULT___"):
            return json.loads(line[len("___RESULT___"):])
    raise SystemExit(f"re-run produced no result:\n{p.stdout[-2000:]}")


def compare(original: dict, repro: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for k, tol in TOLERANCES.items():
        a, b = original.get(k), repro.get(k)
        if a is None or b is None:
            lines.append(f"  {k:<22} MISSING (original={a}, repro={b})")
            ok = False
            continue
        d = abs(a - b)
        hit = d <= tol
        ok &= hit
        lines.append(f"  {k:<22} {a:+.6f} -> {b:+.6f}   |d|={d:.2e} "
                     f"(tol {tol:.0e})  {'OK' if hit else 'MISMATCH'}")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial-id")
    ap.add_argument("--latest-run", help="reproduce the FIRST completed trial of this run")
    a = ap.parse_args()

    tid = a.trial_id
    if not tid and a.latest_run:
        done = [r["trial_id"] for r in trials.read()
                if "completed_at" in r and r["trial_id"].endswith("@" + a.latest_run)
                and (r.get("result") or {}).get("status") == "ok"]
        if not done:
            raise SystemExit(f"no completed trials in run {a.latest_run}")
        tid = sorted(done)[0]
    if not tid:
        raise SystemExit("pass --trial-id or --latest-run")

    pre, original = find(tid)
    print(f"REPRODUCE {tid}")
    print(f"  config: " + ", ".join(f"{k}={pre[k]}" for k in CONFIG_KEYS if k in pre))
    repro = rerun(pre)
    ok, lines = compare(original, repro)
    print("\n".join(lines))
    print(f"  VERDICT: {'REPRODUCED' if ok else 'NOT REPRODUCED'} (SC-006)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
