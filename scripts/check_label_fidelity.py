#!/usr/bin/env python3
"""Fidelity check for research/clfstudy/labels.py (research R2, task T024).

Loads NNNC_MLX through freqtrade's own StrategyResolver (config/config.json, dry-run,
no exchange calls) and compares its `get_training_labels` output against
`research.clfstudy.labels.ternary_labels` on the SAME populated dataframe. This is the
independent check that the study's shim never drifts from production: if it ever
diverges (a mixin override, an attribute change, a new label method), this script is
the thing that catches it.

Usage (from user_data/strategies, with the study's venv):
    PYTHONPATH=. python scripts/check_label_fidelity.py --pair LINK/USDT [--rows 60000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STRATEGIES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STRATEGIES_ROOT))

from research.clfstudy import data, labels  # noqa: E402


def _load_strategy(strategy_name: str = "NNNC_MLX"):
    """Load `strategy_name` via StrategyResolver, dry-run mode, no exchange calls.

    `strategy_path` points at NNNC/ directly: the default (non-recursive) strategy
    search only looks in `<user_data_dir>/strategies` top-level, and NNNC_MLX lives
    one directory down -- the same `--strategy-path` argument `scripts/test_strat.sh`
    passes for every NNNC run.
    """
    from freqtrade.configuration import Configuration
    from freqtrade.resolvers import StrategyResolver

    args = {
        "config": [str(STRATEGIES_ROOT / "config" / "config.json")],
        "strategy": strategy_name,
        "user_data_dir": STRATEGIES_ROOT.parent,
        "strategy_path": str(STRATEGIES_ROOT / "NNNC"),
    }
    cfg = Configuration(args, None).get_config()
    strat = StrategyResolver.load_strategy(cfg)

    # StrategyResolver never sets `strat.dp` (that's wired up by the live bot loop),
    # and `dbg_verbose` defaults to True on BaseStrategy -- with it True,
    # `get_training_labels` -> `print_distribution_compact` -> `debug_print` would
    # touch `self.dp.runmode` and crash. The study shim always runs with
    # dbg_verbose=False for the same reason; match it here so the comparison is
    # apples-to-apples and doesn't crash on the strategy side.
    strat.dbg_verbose = False
    return strat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, help='e.g. "LINK/USDT"')
    parser.add_argument("--rows", type=int, default=None, help="use only the last N rows")
    args = parser.parse_args()

    strat = _load_strategy("NNNC_MLX")

    df = data.populate(data.load_pair(args.pair))
    if args.rows is not None:
        df = df.iloc[-args.rows :].reset_index(drop=True)

    shim_labels = labels.ternary_labels(df)
    strat_labels = np.asarray(strat.get_training_labels(df.copy()), dtype=np.int8)

    if np.array_equal(shim_labels, strat_labels):
        print(f"IDENTICAL (n={len(shim_labels)})")
        return 0

    mismatches = np.where(shim_labels != strat_labels)[0]
    print(f"MISMATCH: {len(mismatches)} / {len(shim_labels)} rows differ")
    for idx in mismatches[:10]:
        print(f"  idx={idx}  shim={int(shim_labels[idx])}  strategy={int(strat_labels[idx])}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
