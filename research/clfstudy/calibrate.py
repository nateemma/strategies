# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
`research/clfstudy/calibrate.py` -- compute-calibration and the 40-hour budget (T025,
research R10).

Runs one arch (default `LSTM`, the production type) once per window at the study's train
defaults, seed 0, and times each run wall-clock. Prints minutes/run per window and the
mean, then projects total hours for every phase's planned run count (research R10) using
that mean. Writes the same to `report/calibration.md`.

The `ExperimentConfig` built here (`hp={}`, `ind24`, `seq_len=16`, `split_set=v1`, train
defaults, `phase=3`) has the identical id fields `phase_3.yaml`'s entry for the same arch
will have, so its `config_id` -- and therefore its `results.jsonl` rows -- are the same;
running `calibrate.py` first means `phase --phase 3` will skip these three windows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from research.clfstudy import REPORT_DIR, RESULTS_PATH, assert_interpreter
from research.clfstudy import registry
from research.clfstudy.registry import ExperimentConfig
from research.clfstudy.run import load_or_build_splitset, run_one

#: planned run counts per phase (research R10); phase 2 is baselines only (~instant).
PHASE_RUN_COUNTS = {2: 18, 3: 144, 4: 186, 5: 36, 6: 24, 7: 120}


def _load_existing_keys() -> set:
    keys = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") in ("ok", "collapsed", "class_absent"):
                    keys.add(
                        (row["config_id"], row["split_set"], row["window_id"], row["seed"])
                    )
    return keys


def main(argv: Optional[List[str]] = None) -> int:
    try:
        assert_interpreter()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.calibrate")
    parser.add_argument("--arch", default="LSTM")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        arch=args.arch, hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=3, seeds=[0],
    )
    cfg.validate()
    cid = registry.config_id(cfg)

    ss = load_or_build_splitset(cfg.split_set)
    existing = _load_existing_keys()

    minutes: dict[str, float] = {}
    for w in ss.windows:
        key = (cid, cfg.split_set, w.window_id, 0)
        if key in existing:
            print(f"window {w.window_id}: already in results.jsonl, skipping timing")
            continue
        t0 = time.perf_counter()
        run_one(cfg, w.window_id, 0)
        minutes[w.window_id] = (time.perf_counter() - t0) / 60.0
        print(f"window {w.window_id}: {minutes[w.window_id]:.2f} min")

    if not minutes:
        print("all windows already calibrated; nothing timed this run")
        return 0

    mean_min = sum(minutes.values()) / len(minutes)
    print(f"mean minutes/run: {mean_min:.2f}")

    lines = [
        "# Compute calibration",
        "",
        f"arch: {args.arch}",
        f"config_id: {cid}",
        "",
        "| window | minutes |",
        "|---|---|",
    ]
    for window_id, m in minutes.items():
        lines.append(f"| {window_id} | {m:.2f} |")
    lines += ["", f"mean minutes/run: {mean_min:.2f}", "", "| phase | runs | projected hours |", "|---|---|---|"]

    total_hours = 0.0
    for phase in sorted(PHASE_RUN_COUNTS):
        count = PHASE_RUN_COUNTS[phase]
        hours = 0.0 if phase == 2 else count * mean_min / 60.0
        total_hours += hours
        lines.append(f"| {phase} | {count} | {hours:.2f} |")
        print(f"phase {phase}: {count} runs -> {hours:.2f} h")
    lines.append(f"| **total** |  | **{total_hours:.2f}** |")
    print(f"total projected hours: {total_hours:.2f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "calibration.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
