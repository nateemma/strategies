# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Causality (truncation-invariance) guard for the MomentumRegimeBasket15m family,
plus an executable explanation of why `freqtrade lookahead-analysis` reports
`has_bias: Yes` for these strategies anyway.

WHY THIS FILE EXISTS
--------------------
`freqtrade lookahead-analysis` flags MomentumRegimeBasket15m / ...15mFast. That
report is a FALSE POSITIVE, but the reason is subtle enough that it was
previously recorded only as prose in the strategy docstring, backed by a
throwaway script in /tmp that has since been deleted -- i.e. an unverifiable
claim. These tests replace both.

THE REAL MECHANISM (see test_single_pair_whitelist_collapses_membership)
------------------------------------------------------------------------
freqtrade/optimize/analysis/lookahead.py builds every comparison run per trade:

    self.prepare_data(entry_varHolder, [result_row["pair"]])   # ONE pair
    ...
    prepare_data_config["exchange"]["pair_whitelist"] = pairs_to_load

So the reference run sees the full whitelist but each cut run is re-run with a
SINGLE-pair whitelist. Under that substitution `_compute_xs` degenerates:

  * `mom.rank(axis=1, ...) <= TOP_N` ranks across one column -> always rank 1,
    so the top-N constraint is vacuously satisfied on every candle; and
  * `known.get(REGIME_REF)` finds no BTC/USDT column, so the code falls to its
    `ron_d = pd.Series(True, ...)` branch and the risk-on gate switches OFF.

`hold` therefore MUST differ between the two runs no matter how causal the
strategy is. The tool cannot validate any cross-sectional or regime-gated
strategy, with any config.

WHAT ACTUALLY NEEDS PROVING (see test_membership_is_truncation_invariant)
-------------------------------------------------------------------------
That `hold[t]` depends only on data at or before `t`. We prove it directly:
compute the membership matrix on full data, recompute it with all data after a
cut point removed, and assert ZERO changed cells on the overlap. Two scenarios:

  "cut_all"  -- daily AND 15m truncated (the strict causality question)
  "ft_exact" -- daily full / 15m truncated, mirroring production, where
                `_daily_closes` reads the feathers straight off disk and so is
                NOT truncated by a timerange

Data is synthetic and seeded, so this is hermetic and fast -- it tests the
ALGORITHM, which is where causality lives. It deliberately exercises the real
`_compute_xs`, not a reimplementation; a copy would rot exactly the way the
/tmp script did.

Run from repo root:

    PYTHONPATH=. .venv/bin/pytest user_data/strategies/Basket/test_momentum_regime_bias.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGIES_ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "Basket"):
    p = str(_STRATEGIES_ROOT / sub) if sub else str(_STRATEGIES_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from MomentumRegimeBasket15m import MomentumRegimeBasket15m
from MomentumRegimeBasket15mFast import MomentumRegimeBasket15mFast

STRATEGIES = [MomentumRegimeBasket15m, MomentumRegimeBasket15mFast]
STRATEGY_IDS = [c.__name__ for c in STRATEGIES]

# BTC/USDT must be present -- it is REGIME_REF, and its absence silently
# disables the risk-on gate (which is half of the false positive above).
PAIRS = [
    "BTC/USDT", "ETH/USDT", "ZEC/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT",
    "NEAR/USDT", "AAVE/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "SUI/USDT",
]
DAILY_HISTORY_DAYS = 400   # must exceed MOM_LOOKBACK_DAYS + REGIME_SMA
TEST_WINDOW_DAYS = 60      # length of the 15m panel under test
CANDLES_PER_DAY = 96       # 15m
_TF_MIN = 15               # timeframe in minutes
CUT_FRACTIONS = (0.50, 0.65, 0.80, 0.95)


class _FakeDP:
    """Minimal DataProvider stand-in: whitelist + per-pair 15m frames."""

    def __init__(self, frames: dict):
        self._frames = frames

    def current_whitelist(self):
        return sorted(self._frames)

    def get_pair_dataframe(self, pair, timeframe):   # noqa: ARG002
        return self._frames.get(pair)


def _make_data(seed: int = 0):
    """Seeded geometric walks on 15m, with the daily panel resampled from them.

    Returns (daily_close_panel, {pair: 15m OHLCV-ish frame}) where the 15m
    frames cover only the last TEST_WINDOW_DAYS, exactly as freqtrade would
    load them, while the daily panel spans the full history read off disk.
    """
    rng = np.random.default_rng(seed)
    n_full = DAILY_HISTORY_DAYS * CANDLES_PER_DAY
    index = pd.date_range("2024-01-01", periods=n_full, freq="15min", tz="UTC")

    daily = {}
    frames = {}
    window = TEST_WINDOW_DAYS * CANDLES_PER_DAY
    for i, pair in enumerate(PAIRS):
        steps = rng.normal(loc=0.0, scale=0.004, size=n_full)
        close = 100.0 * np.exp(np.cumsum(steps))
        s = pd.Series(close, index=index)
        daily[pair] = s.resample("1D").last()
        tail = s.iloc[-window:]
        frames[pair] = pd.DataFrame(
            {
                "date": tail.index,
                "close": tail.values,
                "volume": rng.uniform(1e3, 1e6, size=window),
            }
        ).reset_index(drop=True)
    return pd.DataFrame(daily).sort_index(), frames


_KEEP = object()   # sentinel: leave the class default alone (None is a real value)


def _membership(strategy_cls, daily: pd.DataFrame, frames: dict,
                rebalance_hourly=None, exit_rank_n=_KEEP) -> pd.DataFrame:
    """Run the REAL _compute_xs against injected daily/15m panels."""
    strat = object.__new__(strategy_cls)      # skip IStrategy.__init__
    if rebalance_hourly is not None:
        strat.REBALANCE_HOURLY = rebalance_hourly
    if exit_rank_n is not _KEEP:
        strat.EXIT_RANK_N = exit_rank_n
    strat.dp = _FakeDP(frames)
    strat._xs = None                           # defeat the class-level cache
    strat._xs_key = None
    strat._daily_closes = lambda pairs: daily[[p for p in pairs if p in daily.columns]]
    return strat._compute_xs()


def _truncate(daily, frames, fraction, cut_daily: bool):
    """Drop everything after `fraction` through the 15m window."""
    any_frame = next(iter(frames.values()))
    k = int(len(any_frame) * fraction)
    # Snap so the LAST retained candle sits exactly on the hour. Otherwise the
    # `want[want.index.minute == 0]` + ffill floor overwrites the boundary rows
    # and a one-candle peek is invisible -- a real blind spot, caught by
    # mutation-testing this file (P15.shift(-1) passed until this snap existed).
    k -= (k - 1) % (60 // _TF_MIN)
    cut_ts = any_frame["date"].iloc[k - 1]
    assert cut_ts.minute == 0, f"cut must land on the hour, got {cut_ts}"
    cut_frames = {p: df.iloc[:k].copy() for p, df in frames.items()}
    cut_daily_panel = daily.loc[daily.index <= cut_ts] if cut_daily else daily
    return cut_daily_panel, cut_frames, cut_ts


def _changed_cells(full: pd.DataFrame, cut: pd.DataFrame) -> int:
    overlap = full.loc[cut.index, cut.columns]
    return int((overlap.values != cut.values).sum())


@pytest.mark.parametrize("strategy_cls", STRATEGIES, ids=STRATEGY_IDS)
@pytest.mark.parametrize("fraction", CUT_FRACTIONS)
@pytest.mark.parametrize("scenario", ["cut_all", "ft_exact"])
@pytest.mark.parametrize("hourly", [True, False], ids=["hourly", "per_candle"])
@pytest.mark.parametrize("exit_rank_n", [None, 9], ids=["no_hyst", "hyst9"])
def test_membership_is_truncation_invariant(strategy_cls, fraction, scenario, hourly,
                                            exit_rank_n):
    """hold[t] must not change when data after t is removed.

    `hourly=False` disables the rebalance floor so every candle's membership is
    compared directly -- without it the ffill masks boundary-only divergence.
    """
    daily, frames = _make_data()
    full = _membership(strategy_cls, daily, frames, rebalance_hourly=hourly,
                       exit_rank_n=exit_rank_n)

    cut_daily_panel, cut_frames, cut_ts = _truncate(
        daily, frames, fraction, cut_daily=(scenario == "cut_all")
    )
    cut = _membership(strategy_cls, cut_daily_panel, cut_frames, rebalance_hourly=hourly,
                      exit_rank_n=exit_rank_n)

    assert len(cut) > 0, "truncated run produced no rows"
    changed = _changed_cells(full, cut)
    assert changed == 0, (
        f"{strategy_cls.__name__} [{scenario}/{'hourly' if hourly else 'per_candle'}] "
        f"cut at {cut_ts}: "
        f"{changed} of {cut.size} membership cells changed when future data was "
        f"removed -- this would be genuine lookahead bias"
    )


@pytest.mark.parametrize("strategy_cls", STRATEGIES, ids=STRATEGY_IDS)
def test_membership_actually_selects(strategy_cls):
    """Guard the guard: an all-False matrix would pass invariance vacuously."""
    daily, frames = _make_data()
    want = _membership(strategy_cls, daily, frames)
    held = want.values.sum(axis=1)
    assert held.max() > 0, "no pair was ever selected -- invariance is vacuous"
    assert held.max() <= strategy_cls.TOP_N, (
        f"selected {held.max()} pairs at once, TOP_N is {strategy_cls.TOP_N}"
    )
    assert not want.all().all(), "every pair held on every candle -- degenerate"


@pytest.mark.parametrize("strategy_cls", STRATEGIES, ids=STRATEGY_IDS)
def test_single_pair_whitelist_collapses_membership(strategy_cls):
    """Executable explanation of the lookahead-analysis false positive.

    lookahead.py re-runs each comparison backtest with a single-pair whitelist.
    That makes the cross-sectional rank vacuous AND drops BTC/USDT (REGIME_REF)
    from the panel, so `hold` can only widen. The tool reads that widening as
    bias. Nothing here depends on the strategy being non-causal.
    """
    subject = "ZEC/USDT"
    daily, frames = _make_data()

    full = _membership(strategy_cls, daily, frames)[subject]
    solo_frames = {subject: frames[subject]}
    solo = _membership(strategy_cls, daily, solo_frames)[subject]

    # single-pair membership is a strict superset: rank is vacuous, regime gate off
    assert (full & ~solo).sum() == 0, (
        "single-pair run dropped candles the full run held -- the collapse is "
        "supposed to only ever ADD"
    )
    extra = int((solo & ~full).sum())
    assert extra > 0, (
        "expected the single-pair whitelist to widen `hold`; if this ever fails, "
        "re-derive the false-positive explanation in MomentumRegimeBasket15m's "
        "docstring before trusting a clean lookahead-analysis report"
    )


@pytest.mark.parametrize("strategy_cls", STRATEGIES, ids=STRATEGY_IDS)
def test_exit_hysteresis_default_is_a_noop(strategy_cls):
    """EXIT_RANK_N None/TOP_N must reproduce plain top-N membership exactly.

    The default is now 9, but the None path must stay exactly equivalent to plain
    top-N: it is the control arm every sweep in the docstring is measured against. The slot scan is written so that stay_ok ==
    enter_ok collapses to "the enter_ok set", but that is an argument, not a
    guarantee -- pin it.
    """
    daily, frames = _make_data()
    none_ = _membership(strategy_cls, daily, frames, exit_rank_n=None)

    strat = object.__new__(strategy_cls)
    strat.EXIT_RANK_N = strategy_cls.TOP_N                        # explicit, same thing
    strat.dp = _FakeDP(frames)
    strat._xs = None
    strat._xs_key = None
    strat._daily_closes = lambda pairs: daily[[p for p in pairs if p in daily.columns]]
    explicit = strat._compute_xs()

    assert _changed_cells(none_, explicit) == 0, "None and TOP_N must agree"


@pytest.mark.parametrize("strategy_cls", STRATEGIES, ids=STRATEGY_IDS)
@pytest.mark.parametrize("exit_n", [5, 7])
def test_exit_hysteresis_widens_and_respects_slots(strategy_cls, exit_n):
    """A wider exit band may only LENGTHEN holds, never exceed TOP_N slots."""
    daily, frames = _make_data()

    strat = object.__new__(strategy_cls)
    strat.EXIT_RANK_N = exit_n
    strat.dp = _FakeDP(frames)
    strat._xs = None
    strat._xs_key = None
    strat._daily_closes = lambda pairs: daily[[p for p in pairs if p in daily.columns]]
    wide = strat._compute_xs()

    held = wide.values.sum(axis=1)
    assert held.max() <= strategy_cls.TOP_N, (
        f"hysteresis held {held.max()} positions, TOP_N is {strategy_cls.TOP_N} -- "
        "slot accounting is broken and the equal-weight target (pv/TOP_N) would be wrong"
    )
    base = _membership(strategy_cls, daily, frames, exit_rank_n=None)
    assert wide.values.sum() >= base.values.sum(), "wider exit band held FEWER candle-slots"
