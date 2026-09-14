"""Regenerate per-fold TRAINING-range probability side-cars (supports T012, T020, T035).

WHY THIS RUNS AT ALL. The search recomputes entries at candidate guard thresholds, so it
needs p(buy) on EVERY bar of a fold's training range -- not just the bars the incumbent
happened to enter on. The archive holds 378 admitted trades; a looser guard would have
nothing to admit from it.

WHY A BACKTEST AND NOT OFFLINE INFERENCE. The 15 fold models are saved under saved_data/ and
could be run directly, which would take seconds per fold instead of ~44 minutes. That was
weighed and rejected: it means a SECOND inference path, and this study already has a case
where predict-time scaling silently diverged from the production path -- the single-task GAN
scaler bug, worth 1.53pp and undetected until it was hunted deliberately. Running the
production path costs wall-clock and buys exactness, and the search's whole validity rests on
its entries being the ones the strategy would really take.

Cost: the training ranges are ANCHORED, from 730 days (f00) to 1,990 (f14), against a 90-day
test range. This is the feature's single largest compute item and it runs once.

The two traps below are inherited verbatim from metalabel/reinfer.py, where they were
measured:
  1. freqtrade EXITS 0 ON A FATAL CONFIG ERROR, so `rc` is not a failure signal. Result
     freshness is the only trustworthy check.
  2. ~/.freqtrade_env is sourced by ~/.zshrc for INTERACTIVE shells only, so a
     non-interactive `zsh -c` must source it explicitly or config validation fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from research.labelfree.folds import folds              # noqa: E402

CONFIG = "user_data/strategies/config/config_wf.json"
SINK_ROOT = Path("/tmp/lf/es_sink")
LOG_ROOT = Path("/tmp/lf/es_logs")


def _timerange(start: str, end: str) -> str:
    return f"{start.replace('-', '')}-{end.replace('-', '')}"


def run_fold(fold, which: str, sink_root: Path = SINK_ROOT,
             log_root: Path = LOG_ROOT) -> Path:
    """One backtest over the fold's train or test range, writing the probability side-car."""
    if which not in ("train", "test"):
        raise ValueError(which)
    start, end = ((fold.train_start, fold.train_end) if which == "train"
                  else (fold.test_start, fold.test_end))
    sink = sink_root / f"{fold.fold_id}_{which}"
    sink.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / f"sink_{fold.fold_id}_{which}.log"

    results_dir = REPO / "user_data/backtest_results"
    started = time.time()
    cmd = ("[ -f \"$HOME/.freqtrade_env\" ] && . \"$HOME/.freqtrade_env\"; "
           f"cd {REPO} && FT_PREDICT_PROB_SINK={sink} "
           f"PYTHONPATH=user_data/strategies .venv/bin/freqtrade backtesting "
           f"--strategy NNNC_WF_{fold.fold_id} --strategy-path user_data/strategies/NNNC "
           f"--config {CONFIG} --timerange {_timerange(start, end)} "
           f"--export signals --cache none")
    with log.open("w") as fh:
        fh.write(cmd + "\n\n")
        fh.flush()
        subprocess.run(["zsh", "-c", cmd], stdout=fh, stderr=fh)

    # TRAP 1: rc is not a failure signal. Freshness is.
    latest = results_dir / ".last_result.json"
    if not latest.exists():
        raise SystemExit(f"{fold.fold_id}/{which}: no .last_result.json; see {log}")
    zp = results_dir / json.loads(latest.read_text())["latest_backtest"]
    if not zp.exists() or zp.stat().st_mtime < started:
        raise SystemExit(
            f"{fold.fold_id}/{which}: STALE RESULT — {zp.name} predates this run, so the "
            f"backtest produced nothing. freqtrade exits 0 even on a fatal config error, so "
            f"this is the only reliable failure signal. See {log}")

    written = sorted(sink.glob("*.feather"))
    if not written:
        raise SystemExit(
            f"{fold.fold_id}/{which}: no side-car written to {sink}. Without it the entry "
            f"guards cannot be recomputed and the search space collapses to exits only. "
            f"See {log}")
    return sink


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="f00..f14")
    ap.add_argument("--which", default="train", choices=("train", "test", "both"))
    ap.add_argument("--sink-root", default=str(SINK_ROOT))
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    want = set(parse_span(a.folds))
    todo = [f for f in folds() if f.fold_id in want]
    kinds = ("train", "test") if a.which == "both" else (a.which,)

    print(f"{len(todo)} folds x {len(kinds)} range(s), SEQUENTIAL", flush=True)
    t0 = time.time()
    for i, fd in enumerate(todo, 1):
        for which in kinds:
            s = time.time()
            sink = run_fold(fd, which, Path(a.sink_root))
            n = len(list(sink.glob("*.feather")))
            print(f"[{i}/{len(todo)}] {fd.fold_id}/{which}: {n} pairs, "
                  f"{time.time() - s:.0f}s  (elapsed {(time.time() - t0) / 60:.1f}m)",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} minutes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
