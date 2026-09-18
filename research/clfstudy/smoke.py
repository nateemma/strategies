# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
`research/clfstudy/smoke.py` -- real-data end-to-end smoke check for the harness.

Trains ONE architecture on the real, production split set ("v1") windows A, B, C, on a
caller-chosen subset of real pairs (default one pair, for a fast/low-memory check), and
appends the resulting rows to the REAL `results/results.jsonl` -- but under a config that
is, by construction, distinct from every real study config: `tags=["smoke"]`, `phase=0`,
and a `stride`/`max_epochs` combination the study phases don't use (`config_id` is a hash
of `arch`/`hp`/`input_set`/`seq_len`/`split_set`/`train`, so it never collides with a real
row keyed on the study's own hyperparameters).

This is a fast way to confirm the harness (registry -> splits -> windows -> train -> save)
works end to end on real data without waiting hours for a full phase: it is the same
`run.run_one()` a phase run calls, just for one arch across the three windows.

IMPORTANT -- one trainer at a time: do not run this while a study `phase.py` run is
active. Both write into the same `CACHE_DIR/ckpt/` directory and the same
`results/results.jsonl`; concurrent writers can cross-contaminate each other's checkpoint
files or interleave `results.jsonl` lines (see AGENT_GUIDE.md "Operational rules": run
everything sequentially).

Usage::

    python -m research.clfstudy.smoke --cpu --arch LSTM_FAM --stride 32 --epochs 2 \\
        --pairs LINK/USDT
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from typing import List

MAJORITY_BASELINE_MCC = 0.0


def _run_smoke(arch: str, stride: int, epochs: int, pairs: List[str]) -> int:
    # Imported lazily so `--cpu` (handled in `main()` before this is called) can set the
    # MLX default device before any mlx.core array is created.
    from research.clfstudy import data
    from research.clfstudy import run as run_mod
    from research.clfstudy.registry import ExperimentConfig, config_id

    cfg = ExperimentConfig(
        arch=arch,
        hp={},
        input_set="ind24",
        seq_len=16,
        split_set="v1",
        train={"max_epochs": epochs, "patience": 5, "batch_size": 256, "stride": stride},
        phase=0,
        seeds=[0],
        tags=["smoke"],
    )
    cfg.validate()
    cid = config_id(cfg)
    print(f"smoke: arch={arch} config_id={cid} stride={stride} epochs={epochs} pairs={pairs}")

    # Prime the frame cache with only the requested pairs (real feathers, real indicators,
    # real labels -- `data.load_all` itself is untouched, just called with a pair subset)
    # so `run_one`'s internal `_load_frames()` never loads the full 11-pair universe.
    run_mod._FRAMES_CACHE = data.load_all(pairs=tuple(pairs))

    ss = run_mod.load_or_build_splitset(cfg.split_set)
    window_ids = [w.window_id for w in ss.windows]

    rows = []
    captured_all = []
    for window_id in window_ids:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            row = run_mod.run_one(cfg, window_id, seed=0)
        text = buf.getvalue()
        print(text, end="")
        captured_all.append(text)
        rows.append(row)

    combined_output = "\n".join(captured_all)

    # ---- verdict ------------------------------------------------------------------
    checks: List[tuple] = []

    ckpt_names = [r.get("ckpt_name") for r in rows]
    distinct = len(set(ckpt_names)) == len(ckpt_names) == len(window_ids)
    contains_window = all(
        wid in (name or "") for wid, name in zip(window_ids, ckpt_names)
    )
    checks.append(("distinct ckpt names containing the window id", distinct and contains_window))

    no_warm_start = "MLX model loaded" not in combined_output
    checks.append(('no "MLX model loaded" in captured output', no_warm_start))

    all_ok_status = all(r.get("status") in ("ok", "collapsed", "class_absent") for r in rows)
    checks.append(("no run failed", all_ok_status))

    mccs = [r.get("mcc") for r in rows]
    above_majority = all(
        (m is not None and m == m and m > MAJORITY_BASELINE_MCC) for m in mccs
    )
    checks.append((f"mcc > majority baseline ({MAJORITY_BASELINE_MCC}) on every window", above_majority))

    print("\nsmoke verdict:")
    header = f"{'window':<8}{'ckpt_name':<40}{'epochs_run':<12}{'best_epoch':<12}{'mcc':<10}{'stale_files_removed':<20}"
    print(header)
    for wid, row in zip(window_ids, rows):
        print(
            f"{wid:<8}{str(row.get('ckpt_name')):<40}{row.get('epochs_run'):<12}"
            f"{row.get('best_epoch'):<12}{_fmt_mcc(row.get('mcc')):<10}"
            f"{row.get('stale_files_removed'):<20}"
        )

    print()
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    return 0 if all_pass else 1


def _fmt_mcc(m) -> str:
    if m is None or m != m:  # NaN
        return "NaN"
    return f"{float(m):.4f}"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.smoke")
    parser.add_argument("--arch", default="LSTM_FAM")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pairs", nargs="+", default=["LINK/USDT"])
    parser.add_argument(
        "--cpu", action="store_true",
        help="mx.set_default_device(mx.cpu) before anything else (use when a GPU job "
        "is running concurrently)",
    )
    args = parser.parse_args(argv)

    if args.cpu:
        import mlx.core as mx

        mx.set_default_device(mx.cpu)

    from research.clfstudy import assert_interpreter

    try:
        assert_interpreter()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    return _run_smoke(args.arch, args.stride, args.epochs, args.pairs)


if __name__ == "__main__":
    sys.exit(main())
