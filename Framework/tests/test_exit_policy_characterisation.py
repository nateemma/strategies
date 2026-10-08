# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T004 / SC-001 — characterisation of the LIVE stop decision.

WHY THIS EXISTS, AND WHY THE ORDER MATTERS. Feature 021 extracts the ATR stop
arithmetic out of three places into one (`utils/ExitPolicy.py`). FR-005 demands
PROOF that live behaviour is unchanged, not inspection. So this fixture is
generated from the PRE-refactor code and committed; the refactor must then
reproduce it exactly.

A fixture generated AFTER the refactor would record whatever the new code does
and then assert the new code does it -- it would pass unconditionally and prove
nothing. If a case ever mismatches, the extraction is wrong: fix the extraction,
do NOT regenerate the fixture.

Regenerate ONLY when the exit policy is deliberately changed (which feature 021
forbids -- FR-019 holds it fixed):

    GENERATE_EXIT_FIXTURE=1 .venv/bin/python -m pytest \
        user_data/strategies/Framework/tests/test_exit_policy_characterisation.py

HOW IT CALLS THE LIVE CODE. `custom_stoploss` touches a closed set of 8 `self`
attributes and 2 `trade` fields, so the real method is invoked UNBOUND against a
stub rather than instantiating a freqtrade strategy. That keeps the test fast and
hermetic while still exercising the genuine code path, including
`_resolve_normal_stop`, which is bound onto the stub from the real class.
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

FIXTURE = Path(__file__).parent / "fixtures" / "exit_policy_characterisation.json"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "binanceus"

#: Multi-pair and multi-year by construction. SOL/XRP are the liquid pairs and ZEC
#: the thin one, so the sampled atr_pct_roll spans the real range rather than one
#: pair's idiosyncrasy.
PAIRS = ["SOL_USDT", "XRP_USDT", "ZEC_USDT"]

#: Sampled at a fixed stride from a fixed offset -- deterministic, and spread
#: across the whole 2021-2026 span rather than clustered in one regime.
BAR_STRIDE = 9_973          # prime, so it does not alias with any periodicity
BARS_PER_PAIR = 12


class _StubTrade:
    def __init__(self, open_date_utc: datetime, open_rate: float):
        self.open_date_utc = open_date_utc
        self.open_rate = open_rate


class _StubDataProvider:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return self._df, {}


class _StubStrategy:
    """Carries exactly the attributes the live stop path reads -- nothing more.

    If the refactor makes `custom_stoploss` read something new, this stub raises
    AttributeError and the test fails loudly. That is deliberate: a widened
    dependency surface is itself a behaviour change worth catching.
    """

    timeframe = "15m"

    def __init__(self, df: pd.DataFrame, **overrides):
        self.dp = _StubDataProvider(df)
        # Live values, read off BaseStrategy / BaseNNStrategy. FR-019 holds these fixed.
        self.stoploss = -0.05
        self.atr_stoploss_multiplier = 2.5
        self.atr_stoploss_floor = -0.04
        self.atr_stoploss_cap = -0.02
        self.use_atr_adaptive_stoploss = True
        self.use_volume_confirmation_stoploss = True
        self.stoploss_grace_hours = 0.0
        self.stoploss_grace_level = -0.15
        for k, v in overrides.items():
            setattr(self, k, v)
        # Bind the real methods the live stop path uses. The refactor added
        # `_exit_params` / `_atr_bar_inputs` as internal helpers; binding them from the
        # real class (rather than reimplementing them here) keeps this test an exercise
        # of production code. An AttributeError here means the dependency surface grew
        # again, which is itself worth failing on.
        for _name in ("_resolve_normal_stop", "_exit_params", "_atr_bar_inputs"):
            if hasattr(BaseStrategy, _name):
                setattr(self, _name, types.MethodType(getattr(BaseStrategy, _name), self))


def _load(pair: str) -> pd.DataFrame:
    f = DATA_DIR / f"{pair}-15m.feather"
    if not f.exists():
        pytest.skip(f"missing data: {f}")
    df = pd.read_feather(f).rename(columns=str.lower)
    return DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)


def _cases():
    """Every (pair, bar, config, age, after_fill) combination, deterministically.

    Covers both branches of the grace gate, both `after_fill` values, the ATR path
    and the static fallback, volume confirmation on and off, and -- because the
    bars are real -- all three regions of the cap/floor clamp.
    """
    for pair in PAIRS:
        df = _load(pair)
        n = len(df)
        for i in range(BARS_PER_PAIR):
            idx = (1_000 + i * BAR_STRIDE) % n
            if idx < 50:
                continue
            window = df.iloc[: idx + 1]
            rate = float(window.iloc[-1]["close"])
            if not rate > 0:
                continue
            for cfg_name, cfg in (
                ("atr_vol", {}),
                ("atr_novol", {"use_volume_confirmation_stoploss": False}),
                ("static", {"use_atr_adaptive_stoploss": False}),
                ("grace", {"stoploss_grace_hours": 6.5, "stoploss_grace_level": -0.27}),
            ):
                for age_h in (0.0, 3.0, 6.4, 6.6, 48.0):
                    for after_fill in (True, False):
                        yield {
                            "pair": pair,
                            "bar": int(idx),
                            "cfg": cfg_name,
                            "age_h": age_h,
                            "after_fill": after_fill,
                        }, (window, rate, cfg)


def _evaluate(case, payload) -> float:
    window, rate, cfg = payload
    strat = _StubStrategy(window, **cfg)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trade = _StubTrade(now - timedelta(hours=case["age_h"]), rate)
    return float(
        BaseStrategy.custom_stoploss(
            strat,
            pair=case["pair"].replace("_", "/"),
            trade=trade,
            current_time=now,
            current_rate=rate,
            current_profit=0.0,
            after_fill=case["after_fill"],
        )
    )


def _build():
    return [{**c, "stop": _evaluate(c, p)} for c, p in _cases()]


def test_characterisation():
    recorded = _build()
    assert len(recorded) > 400, f"grid collapsed to {len(recorded)} cases"

    if os.environ.get("GENERATE_EXIT_FIXTURE"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"fixture regenerated: {len(recorded)} cases -> {FIXTURE}")

    assert FIXTURE.exists(), (
        f"{FIXTURE} missing. It must be generated from PRE-refactor code:\n"
        "  GENERATE_EXIT_FIXTURE=1 python -m pytest <this file>"
    )
    expected = json.loads(FIXTURE.read_text())
    assert len(recorded) == len(expected), (
        f"case count changed: {len(expected)} -> {len(recorded)}. The grid itself "
        "moved, so the comparison is meaningless until it is restored."
    )

    diffs = [
        (e, r["stop"]) for e, r in zip(expected, recorded)
        if abs(e["stop"] - r["stop"]) > 1e-12
    ]
    assert not diffs, (
        f"{len(diffs)} of {len(recorded)} stop decisions CHANGED. The extraction is "
        f"wrong -- fix it, do NOT regenerate the fixture.\nFirst 5: "
        + "; ".join(
            f"{e['pair']}@{e['bar']} cfg={e['cfg']} age={e['age_h']} "
            f"after_fill={e['after_fill']}: {e['stop']:.6f} -> {got:.6f}"
            for e, got in diffs[:5]
        )
    )


def test_grid_covers_every_branch():
    """A fixture that silently stopped exercising a branch would pass forever."""
    cases = [c for c, _ in _cases()]
    assert {c["cfg"] for c in cases} == {"atr_vol", "atr_novol", "static", "grace"}
    assert {c["after_fill"] for c in cases} == {True, False}
    grace = [c for c in cases if c["cfg"] == "grace"]
    assert any(c["age_h"] < 6.5 for c in grace), "no bar inside the grace window"
    assert any(c["age_h"] > 6.5 for c in grace), "no bar outside the grace window"
