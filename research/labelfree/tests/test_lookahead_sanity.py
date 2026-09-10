"""Lookahead sanity ceiling — SC-002, paired form (T020).

The criterion is PAIRED on identical data: leaky must exceed clean by >= 2.0 log units of
cumulative net return on the same range and seed, and the clean run must not beat
buy-and-hold over that range by more than the resolvable floor.

Why paired: a fixed multiple of a single window's return is not commensurable across ranges.
Measured 2026-09-10 — on f08's 4-year TRAINING range a flat 0.28-exposure policy shows +289%
from beta alone, while on its 90-day TEST range a good policy shows -16.9%. No absolute bar
works across both; a paired difference is immune to horizon and regime.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from regime import regimedata as rd                                     # noqa: E402
from research.labelfree import features as FT                           # noqa: E402
from research.labelfree import folds as F                               # noqa: E402
from research.labelfree import train as T                               # noqa: E402
from research.labelfree.costs import CostModel                          # noqa: E402
from research.labelfree.stats import SINGLE_WINDOW_FLOOR_PP             # noqa: E402

MIN_LOG_SEPARATION = 2.0
PAIR = "SOL/USDT"
SEED = 1
EPOCHS = 8


@pytest.fixture(scope="module")
def paired():
    fold = {f.fold_id: f for f in F.folds()}["f08"]
    X, px, vq, _ = FT.build_features(PAIR, fold.test_start, fold.test_end)
    df = rd.load_ohlcv(PAIR.split("/")[0])
    df = df[(df.index >= pd.Timestamp(fold.test_start, tz="UTC")) &
            (df.index <= pd.Timestamp(fold.test_end, tz="UTC"))]
    cost = CostModel().per_unit(df, evaluation=True).to_numpy()[:len(px)]
    out = {}
    for leaky in (True, False):
        cfg = T.TrainConfig(objective="utility", turnover_control="penalty",
                            seed=SEED, epochs=EPOCHS, leaky=leaky)
        _, h = T.train_fold(X, px, cost, vq, cfg)
        k = "leaky" if leaky else "clean"
        out[k] = h["total_log_return"]
        out[k + "_mean_w"] = h["mean_exposure"]
    out["buy_and_hold"] = float(np.log(px[-1] / px[1]))
    return out


def test_leaky_separates_from_clean(paired):
    """The gradient path can exploit deliberate lookahead."""
    sep = paired["leaky"] - paired["clean"]
    assert sep >= MIN_LOG_SEPARATION, (
        f"leaky-clean separation {sep:.3f} log < {MIN_LOG_SEPARATION}; "
        "the learning path cannot exploit lookahead and is defective")


def test_clean_does_not_beat_its_exposure_matched_benchmark(paired):
    """The clean run is honest: no absurd return without the leak.

    The benchmark is buy-and-hold scaled to the policy's OWN mean exposure. A long-only
    policy that is only partly invested beats a fully-invested benchmark in any falling
    market purely by holding less, which is beta, not skill. Measured 2026-09-10: the clean
    policy returned -16.9% against a -34.0% market and so "beat" it by 17.2pp, while its
    mean exposure was 0.408 - exposure-matched, it is slightly BEHIND its benchmark.
    """
    mw = paired["clean_mean_w"]
    bench = mw * paired["buy_and_hold"]
    excess_pp = 100.0 * (np.exp(paired["clean"]) - np.exp(bench))
    assert excess_pp <= SINGLE_WINDOW_FLOOR_PP, (
        f"clean run beat its exposure-matched benchmark (mean_w={mw:.3f}) by "
        f"{excess_pp:.1f}pp, above the {SINGLE_WINDOW_FLOOR_PP}pp floor — suspect leakage")
