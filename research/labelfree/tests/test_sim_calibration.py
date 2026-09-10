"""SC-001 calibration test (T010) — the blocking gate.

Passes only when all three hold: per-trade profit within 1 bp, total return within 0.10pp,
and identical trade counts.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from research.labelfree import archive                                  # noqa: E402
from research.labelfree.calibrate import (TOL_PER_TRADE_BPS,            # noqa: E402
                                          TOL_TOTAL_PP, calibrate)

STRATEGY, WINDOW = "NNNC_MLX", "W1"


@pytest.fixture(scope="module")
def report():
    return calibrate(archive.latest_zip(STRATEGY), STRATEGY, WINDOW)


def test_trade_counts_identical(report):
    assert report["n_trades_arbiter"] == report["n_trades_sim"] == report["n_matched"]


def test_per_trade_within_one_bp(report):
    assert report["max_per_trade_diff_bps"] <= TOL_PER_TRADE_BPS


def test_total_within_tenth_of_a_point(report):
    assert abs(report["total_diff_pp"]) <= TOL_TOTAL_PP


def test_verdict_pass(report):
    assert report["verdict"] == "pass"
