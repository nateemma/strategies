# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""Characterisation of the LIVE `custom_exit` decision (specs/021-label-exit-coordination,
FR-001 amendment / T021 follow-up).

WHY THIS EXISTS, AND WHY THE ORDER MATTERS -- same discipline as
`test_exit_policy_characterisation.py` (T004, the stop fixture). T021's reconciliation found
the outcome table optimistic by 10.7% on mean profit/trade because `simulate_forward` only
modelled stop/take-profit/hold-cap while the live `custom_exit` ALSO fires on
`metric_overbought`, `unclog_12h`, `unclog_1d` and `max_hold` (FR-001, amended). Before any of
that logic is extracted into `utils.ExitPolicy.custom_exit_decision` and `custom_exit` is made
a thin caller over it, this fixture records the CURRENT, unrefactored function's decision over
a grid spanning every documented branch. A fixture generated AFTER the refactor would prove
nothing (same reasoning as T004's docstring).

Regenerate ONLY if `custom_exit`'s decision logic is deliberately changed (FR-019 forbids
changing any exit PARAMETER value; this fixture is not expected to need regeneration for this
feature at all):

    GENERATE_CUSTOM_EXIT_FIXTURE=1 .venv/bin/python -m pytest \
        user_data/strategies/Framework/tests/test_custom_exit_characterisation.py

HOW IT CALLS THE LIVE CODE. `custom_exit` touches `self.dp`, `self.cexit_enable_profit_checks`,
`self.cexit_take_profit`, `self.cexit_max_days`, and (via `strategy_custom_exit`, bound from the
real class) nothing else -- so, as with the stop fixture, the real unbound method is invoked
against a minimal stub rather than instantiating a freqtrade strategy.

GRID. Per the task brief: trade age straddling the 12h / 1d / max_hold(10d=240h) boundaries,
`current_profit` straddling 0 / 0.005 / 0.022(cexit_take_profit) / 0.05(the ROI level -- not
read by `custom_exit` itself, included only so the grid's profit axis covers every named
threshold), and `guard_metric` both below and above the 0.98 `metric_overbought` threshold.
`guard_metric` is injected directly onto each pair's real last-candle row (rather than hunting
for naturally-occurring rare >0.98 bars) so both branches are covered deterministically --
research.md documents the real occurrence rate (0.3-2.6% of bars) elsewhere; this fixture's
job is to characterise the function given its inputs, not to re-measure that rate.
"""
from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "custom_exit_characterisation.json"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "binanceus"

PAIRS = ["SOL_USDT", "XRP_USDT", "ZEC_USDT"]
BARS_PER_PAIR = 2
BAR_STRIDE = 9_973  # same prime stride as the stop fixture, deliberately uncorrelated

#: Straddles 12h, 24h (1d) and 240h (cexit_max_days=10d) on both sides.
AGE_HOURS = (0.0, 6.0, 11.9, 12.1, 23.9, 24.1, 239.9, 240.1)

#: Straddles 0, 0.005 (unclog_12h), 0.022 (cexit_take_profit) and 0.05 (the ROI level --
#: not read by custom_exit, kept for grid completeness per the task brief).
PROFITS = (-0.01, 0.0, 0.003, 0.005, 0.006, 0.02, 0.022, 0.023, 0.05, 0.06)

#: Below / above the metric_overbought threshold.
GUARD_METRICS = (0.5, 0.99)


class _P:
    """Shim for a freqtrade `Parameter` -- `custom_exit` reads `.value`."""

    def __init__(self, value):
        self.value = value


class _StubTrade:
    def __init__(self, open_date_utc: datetime, is_short: bool = False):
        self.open_date_utc = open_date_utc
        self.is_short = is_short


class _StubDataProvider:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return self._df, {}


class _StubStrategy:
    """Carries exactly the attributes the live `custom_exit` path reads."""

    timeframe = "15m"

    def __init__(self, df: pd.DataFrame):
        self.dp = _StubDataProvider(df)
        self.cexit_enable_profit_checks = _P(True)
        self.cexit_take_profit = _P(0.022)   # Open operating point (NNNC_MLX_Open.json)
        self.cexit_max_days = _P(10)
        # `custom_exit` now calls `self._exit_params()` (feature 021 FR-001's amendment),
        # which reads these too, even though none of them affect custom_exit's own decision
        # (they feed the native stop, evaluated separately by freqtrade). Required only so
        # `_exit_params()`/`ExitParams.__post_init__` do not raise.
        self.stoploss = -0.05
        self.atr_stoploss_multiplier = 2.5
        self.atr_stoploss_floor = -0.04
        self.atr_stoploss_cap = -0.02
        self.use_atr_adaptive_stoploss = True
        self.use_volume_confirmation_stoploss = True
        self._exit_params = types.MethodType(BaseStrategy._exit_params, self)
        self.strategy_custom_exit = types.MethodType(BaseStrategy.strategy_custom_exit, self)


def _load(pair: str) -> pd.DataFrame:
    f = DATA_DIR / f"{pair}-15m.feather"
    if not f.exists():
        pytest.skip(f"missing data: {f}")
    df = pd.read_feather(f).rename(columns=str.lower)
    return DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)


def _cases():
    for pair in PAIRS:
        df = _load(pair)
        n = len(df)
        for i in range(BARS_PER_PAIR):
            idx = (2_000 + i * BAR_STRIDE) % n
            if idx < 50:
                continue
            window = df.iloc[: idx + 1].copy()
            for guard_metric in GUARD_METRICS:
                for age_h in AGE_HOURS:
                    for profit in PROFITS:
                        yield {
                            "pair": pair,
                            "bar": int(idx),
                            "guard_metric": guard_metric,
                            "age_h": age_h,
                            "profit": profit,
                        }, window


def _evaluate(case, window: pd.DataFrame) -> "str | None":
    window = window.copy()
    window.iloc[-1, window.columns.get_loc("guard_metric")] = case["guard_metric"]
    strat = _StubStrategy(window)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trade = _StubTrade(now - timedelta(hours=case["age_h"]))
    return BaseStrategy.custom_exit(
        strat,
        pair=case["pair"].replace("_", "/"),
        trade=trade,
        current_time=now,
        current_rate=100.0,
        current_profit=case["profit"],
    )


def _build():
    return [{**c, "reason": _evaluate(c, w)} for c, w in _cases()]


def test_characterisation():
    recorded = _build()
    assert len(recorded) > 800, f"grid collapsed to {len(recorded)} cases"

    if os.environ.get("GENERATE_CUSTOM_EXIT_FIXTURE"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"fixture regenerated: {len(recorded)} cases -> {FIXTURE}")

    assert FIXTURE.exists(), (
        f"{FIXTURE} missing. It must be generated from PRE-refactor code:\n"
        "  GENERATE_CUSTOM_EXIT_FIXTURE=1 python -m pytest <this file>"
    )
    expected = json.loads(FIXTURE.read_text())
    assert len(recorded) == len(expected), (
        f"case count changed: {len(expected)} -> {len(recorded)}. The grid itself "
        "moved, so the comparison is meaningless until it is restored."
    )

    diffs = [
        (e, r["reason"]) for e, r in zip(expected, recorded)
        if e["reason"] != r["reason"]
    ]
    assert not diffs, (
        f"{len(diffs)} of {len(recorded)} custom_exit decisions CHANGED. The extraction is "
        f"wrong -- fix it, do NOT regenerate the fixture.\nFirst 5: "
        + "; ".join(
            f"{e['pair']}@{e['bar']} age={e['age_h']} profit={e['profit']} "
            f"guard_metric={e['guard_metric']}: {e['reason']!r} -> {got!r}"
            for e, got in diffs[:5]
        )
    )


def test_grid_covers_every_branch():
    """A fixture that silently stopped exercising a branch would pass forever."""
    recorded = _build()
    reasons = {r["reason"] for r in recorded}
    expected_reasons = {
        None,
        "metric_overbought",
        "take_profit",
        "unclog_12h",
        "unclog_1d",
        "max_hold",
    }
    assert reasons == expected_reasons, (
        f"grid produced reasons {sorted(str(r) for r in reasons)}, expected "
        f"{sorted(str(r) for r in expected_reasons)} -- a branch stopped firing or an "
        "unexpected one appeared"
    )
