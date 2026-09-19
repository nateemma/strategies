# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
`research/clfstudy/phase.py` -- run every not-yet-completed (config, window, seed) in a
phase file (T018, research R9's resumability guarantee).

Loads `configs/phase_N.yaml` (registry.load_phase), expands each config's `seeds` x its
split set's windows into run keys, skips keys already present in `results/results.jsonl`
with status `ok`/`collapsed`/`class_absent`, and retries `failed` keys only with
`--retry-failed`.

By default (`--isolate`, the default) each (config, window, seed) is run in a *fresh
subprocess* (`python -m research.clfstudy.run --config <inline JSON> ...`) rather than
in-process: `run.run_one`'s own cleanup (T029) frees MLX's Metal buffer cache after every
run, but a long-lived process still accumulates some Metal allocations run over run: a
subprocess is the only way to hand every byte back to the OS between runs. `--no-isolate`
runs everything in this process via `run.run_one` directly (the old behaviour) -- useful
for `--dry-run`-style debugging where subprocess overhead isn't worth it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

from research.clfstudy import CONFIGS_DIR, RESULTS_PATH, assert_interpreter
from research.clfstudy import registry
from research.clfstudy.registry import ExperimentConfig
from research.clfstudy.run import load_or_build_splitset, run_one

_SKIP_STATUSES = {"ok", "collapsed", "class_absent"}

RunKey = Tuple[str, str, str, int]  # (config_id, split_set, window_id, seed)


def _load_existing_statuses() -> Dict[RunKey, str]:
    """`RunKey -> status` for every row currently in `results/results.jsonl`."""
    existing: Dict[RunKey, str] = {}
    if not RESULTS_PATH.exists():
        return existing
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key: RunKey = (row["config_id"], row["split_set"], row["window_id"], row["seed"])
            existing[key] = row["status"]
    return existing


def _should_skip(status: Optional[str], retry_failed: bool) -> bool:
    if status is None:
        return False
    if status in _SKIP_STATUSES:
        return True
    if status == "failed":
        return not retry_failed
    return False


def _matches_only(cfg: ExperimentConfig, cid: str, only: Optional[Set[str]]) -> bool:
    if only is None:
        return True
    return cfg.arch in only or cid in only


def expand_runs(
    configs: List[ExperimentConfig],
    existing: Dict[RunKey, str],
    only: Optional[Set[str]] = None,
    retry_failed: bool = False,
) -> Tuple[List[Tuple[ExperimentConfig, str, int]], int]:
    """Expand `configs` x windows(cfg.split_set) x cfg.seeds into run keys not already
    completed. Returns `(runs, skipped_count)`."""
    runs: List[Tuple[ExperimentConfig, str, int]] = []
    skipped = 0
    for cfg in configs:
        cid = registry.config_id(cfg)
        if not _matches_only(cfg, cid, only):
            continue
        ss = load_or_build_splitset(cfg.split_set)
        for w in ss.windows:
            for seed in cfg.seeds:
                key: RunKey = (cid, cfg.split_set, w.window_id, seed)
                status = existing.get(key)
                if _should_skip(status, retry_failed):
                    skipped += 1
                    continue
                runs.append((cfg, w.window_id, seed))
    return runs, skipped


def _isolated_cmd(
    cfg: ExperimentConfig, window_id: str, seed: int, save_probs: bool = False,
) -> List[str]:
    """`run.py`'s CLI command for one (config, window, seed), as a fresh subprocess.

    The inline `--config` JSON carries every `ExperimentConfig` field (not just the
    `config_id`-bearing ones) so `run._load_config` reconstructs an identical config;
    `cfg.train` is already merged with `TRAIN_DEFAULTS` (`ExperimentConfig.__post_init__`),
    and re-merging an already-complete dict with the same defaults is a no-op, so the
    child's `config_id` matches the parent's exactly.
    """
    payload = {
        "arch": cfg.arch,
        "hp": cfg.hp,
        "input_set": cfg.input_set,
        "seq_len": cfg.seq_len,
        "split_set": cfg.split_set,
        "train": cfg.train,
        "phase": cfg.phase,
        "seeds": cfg.seeds,
        "tags": cfg.tags,
    }
    cmd = [
        sys.executable, "-m", "research.clfstudy.run",
        "--config", json.dumps(payload, sort_keys=True),
        "--window", window_id,
        "--seed", str(seed),
    ]
    if save_probs:
        cmd.append("--save-probs")
    return cmd


def _run_isolated(
    cfg: ExperimentConfig, window_id: str, seed: int, save_probs: bool = False,
) -> Tuple[int, Optional[str]]:
    """Run one (config, window, seed) in a fresh subprocess (streaming its stdout/stderr
    through unchanged) and return `(returncode, status)`, where `status` is read back from
    `results/results.jsonl` afterwards (`None` if the child never appended a row, e.g. it
    was killed mid-run)."""
    cmd = _isolated_cmd(cfg, window_id, seed, save_probs=save_probs)
    result = subprocess.run(cmd, check=False)
    key: RunKey = (registry.config_id(cfg), cfg.split_set, window_id, seed)
    status = _load_existing_statuses().get(key)
    return result.returncode, status


def main(argv: Optional[List[str]] = None) -> int:
    try:
        assert_interpreter()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.phase")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--only", default=None, help="comma-separated arch names or config_ids")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--save-probs", action="store_true",
        help="forward --save-probs to every run.py invocation (isolated or in-process)",
    )
    parser.add_argument(
        "--isolate", action=argparse.BooleanOptionalAction, default=True,
        help="run each (config, window, seed) in a fresh subprocess so MLX/Metal memory "
        "is returned to the OS between runs (default: on; --no-isolate runs in-process)",
    )
    args = parser.parse_args(argv)

    phase_path = CONFIGS_DIR / f"phase_{args.phase}.yaml"
    try:
        _header, configs = registry.load_phase(phase_path)
    except Exception as exc:
        print(f"invalid phase file {phase_path}: {exc}", file=sys.stderr)
        return 2

    only = set(args.only.split(",")) if args.only else None

    existing = _load_existing_statuses()
    runs, skipped = expand_runs(configs, existing, only=only, retry_failed=args.retry_failed)

    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    n = len(runs)
    ran = 0
    failed = 0
    for i, (cfg, window_id, seed) in enumerate(runs, start=1):
        print(
            f"[{i}/{n}] config_id={registry.config_id(cfg)} arch={cfg.arch} "
            f"window={window_id} seed={seed}"
        )
        if args.isolate:
            returncode, status = _run_isolated(cfg, window_id, seed, save_probs=args.save_probs)
            ran += 1
            if returncode not in (0, 1):
                print(f"  child exited with unexpected code {returncode}", file=sys.stderr)
            if returncode != 0 or status not in _SKIP_STATUSES:
                failed += 1
        else:
            row = run_one(cfg, window_id, seed, save_probs=args.save_probs)
            ran += 1
            if row.get("status") == "failed":
                failed += 1

    print(f"phase {args.phase}: skipped={skipped} ran={ran} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
