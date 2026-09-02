"""Phase-instrument catalogue for the market-phase study.

An instrument is a named, causal recipe mapping price/volume history to a
discrete state per bar (data-model.md section 1). Three levels:

  asset  - computed on the traded pair
  market - computed cross-sectionally over the universe (coverage rule applies)
  btc    - computed on BTC only

Every instrument MUST be causal, MUST include UNKNOWN in its state vocabulary,
and MUST NOT reference a specific trading pair (FR-025, Constitution X).

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/instruments.py
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import talib

from regime.regimedata import UNKNOWN, MIN_PAIRS, MIN_HISTORY_BARS

# Estimator defaults (research R1). Fixed, never adaptive - making the window
# adaptive would itself require a trackability test (Constitution VII).
WINDOW = 480
Q_GRID = (2, 4, 8, 24)


@dataclass
class PhaseInstrument:
    name: str
    level: str                      # asset | market | btc
    states: tuple                   # state vocabulary, always includes UNKNOWN
    window_bars: int
    fn: Callable                    # (inputs) -> pd.Series of state labels
    params: dict = field(default_factory=dict)
    is_baseline: bool = False
    status: str = "PENDING"         # PENDING | SURVIVING | ELIMINATED
    status_reason: str = ""

    def __post_init__(self):
        if UNKNOWN not in self.states:
            raise ValueError(f"{self.name}: state vocabulary must include {UNKNOWN}")
        if self.level not in ("asset", "market", "btc"):
            raise ValueError(f"{self.name}: bad level {self.level!r}")


# ---------------------------------------------------------------------------
# BTC level - the exp4 baseline, reproduced VERBATIM (FR-004)
# ---------------------------------------------------------------------------

EXP4_STATES = ("Strong Down", "Weak Down", "Sideways", "Weak Up", "Strong Up", UNKNOWN)


def exp4_regime(ohlcv: pd.DataFrame) -> pd.Series:
    """The btc_corr Exp-4 taxonomy, unchanged.

    ADX strength x EMA-slope direction. Reproduced exactly as
    btc_corr/exp4_regime.py:btc_regime() so every new instrument is measured
    against a known reference rather than in isolation. The only change is the
    NaN sentinel: exp4 used np.nan, we use the UNKNOWN label so the state
    vocabulary is closed.
    """
    c = ohlcv["close"].values
    h = ohlcv["high"].values
    l = ohlcv["low"].values

    ema = talib.EMA(c, 50)
    slope = pd.Series(ema, index=ohlcv.index).pct_change(24)   # 24-bar slope, causal
    adx = pd.Series(talib.ADX(h, l, c, 14), index=ohlcv.index)

    reg = pd.Series("Sideways", index=ohlcv.index, dtype=object)
    up, dn = slope > 0, slope < 0
    reg[(adx >= 20) & (adx < 30) & up] = "Weak Up"
    reg[(adx >= 30) & up] = "Strong Up"
    reg[(adx >= 20) & (adx < 30) & dn] = "Weak Down"
    reg[(adx >= 30) & dn] = "Strong Down"
    reg[adx < 20] = "Sideways"
    reg[adx.isna() | slope.isna()] = UNKNOWN
    return reg.rename("exp4")


def btc_vol_state(ohlcv: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    """BTC realized-volatility tercile.

    Exp 2 found BTC's predictive content sits in its volatility/trend STATE,
    not its returns, so this carries the vol axis explicitly rather than
    letting the exp4 direction cut stand in for it.
    """
    r = np.log(ohlcv["close"]).diff()
    vol = r.rolling(96).std()                       # 24h realized vol
    lo = vol.rolling(window).quantile(0.33)
    hi = vol.rolling(window).quantile(0.67)

    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    out[vol <= lo] = "Low Vol"
    out[(vol > lo) & (vol < hi)] = "Mid Vol"
    out[vol >= hi] = "High Vol"
    out[vol.isna() | lo.isna() | hi.isna()] = UNKNOWN
    return out.rename("btc_vol")


# ---------------------------------------------------------------------------
# Asset level - the direct trend-vs-reversion measures (FR-003)
# ---------------------------------------------------------------------------

def _tercile_state(x: pd.Series, window: int, labels: tuple) -> pd.Series:
    """Discretise a continuous measure into terciles of its own rolling history.

    Self-referencing terciles keep the state definition scale-free across pairs,
    which is what lets one instrument definition serve every pair without
    per-pair tuning (Constitution X).
    """
    lo = x.rolling(window).quantile(0.33)
    hi = x.rolling(window).quantile(0.67)
    out = pd.Series(UNKNOWN, index=x.index, dtype=object)
    out[x <= lo] = labels[0]
    out[(x > lo) & (x < hi)] = labels[1]
    out[x >= hi] = labels[2]
    out[x.isna() | lo.isna() | hi.isna()] = UNKNOWN
    return out


VR_STATES = ("Reverting", "Neutral", "Trending", UNKNOWN)


def vr_state(ohlcv: pd.DataFrame, q: int = 4, window: int = WINDOW) -> pd.Series:
    """Variance-ratio LEVEL, cut at the theoretical random-walk value of 1.

    Absolute cuts rather than terciles: VR has a meaningful zero point (VR=1 is
    a random walk), so a tercile split would manufacture three states even in a
    series that is a random walk throughout.
    """
    from utils.custom_indicators import variance_ratio
    vr = variance_ratio(ohlcv["close"], q=q, window=window)
    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    out[vr < 0.9] = "Reverting"
    out[(vr >= 0.9) & (vr <= 1.1)] = "Neutral"
    out[vr > 1.1] = "Trending"
    out[vr.isna()] = UNKNOWN
    return out.rename(f"vr{q}")


HURST_STATES = ("Anti-persistent", "Random", "Persistent", UNKNOWN)


def hurst_state(ohlcv: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    """Hurst LONG-MEMORY state, cut around 0.5.

    NOT a substitute for vr_state: on short-memory processes Hurst reads ~0.5
    while VR level correctly shows reversion (research R1 finding). The two are
    complementary instruments and are expected to disagree.
    """
    from utils.custom_indicators import hurst_from_vr
    h = hurst_from_vr(ohlcv["close"], q_grid=Q_GRID, window=window)
    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    out[h < 0.45] = "Anti-persistent"
    out[(h >= 0.45) & (h <= 0.55)] = "Random"
    out[h > 0.55] = "Persistent"
    out[h.isna()] = UNKNOWN
    return out.rename("hurst")


AC_STATES = ("Neg Autocorr", "No Autocorr", "Pos Autocorr", UNKNOWN)


def autocorr_state(ohlcv: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    from utils.custom_indicators import rolling_autocorr
    ac = rolling_autocorr(ohlcv["close"], lag=1, window=window)
    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    out[ac < -0.05] = "Neg Autocorr"
    out[(ac >= -0.05) & (ac <= 0.05)] = "No Autocorr"
    out[ac > 0.05] = "Pos Autocorr"
    out[ac.isna()] = UNKNOWN
    return out.rename("autocorr")


ADX_STATES = ("Choppy", "Transitional", "Trending", UNKNOWN)


def asset_adx_state(ohlcv: pd.DataFrame) -> pd.Series:
    """Per-asset ADX - the free proxy the direct measures are compared against."""
    adx = pd.Series(
        talib.ADX(ohlcv["high"].values, ohlcv["low"].values, ohlcv["close"].values, 14),
        index=ohlcv.index,
    )
    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    out[adx < 20] = "Choppy"
    out[(adx >= 20) & (adx < 30)] = "Transitional"
    out[adx >= 30] = "Trending"
    out[adx.isna()] = UNKNOWN
    return out.rename("asset_adx")


TREND_STATES = ("Below EMA", "Above EMA", UNKNOWN)


def asset_trend_state(ohlcv: pd.DataFrame, period: int = 200) -> pd.Series:
    """close > EMA(period) - the ONE regime intervention already validated here.

    The EMA-slope check is deliberately absent: it was removed 2026-05-30 for
    cutting entries during normal mid-uptrend pullbacks.
    """
    ema = ohlcv["close"].ewm(span=period, adjust=False).mean()
    out = pd.Series(UNKNOWN, index=ohlcv.index, dtype=object)
    ok = ohlcv["close"].notna() & ema.notna()
    out[ok & (ohlcv["close"] > ema)] = "Above EMA"
    out[ok & (ohlcv["close"] <= ema)] = "Below EMA"
    # EMA is defined from bar 0, so warm it up explicitly.
    out.iloc[:period] = UNKNOWN
    return out.rename("asset_trend")


# ---------------------------------------------------------------------------
# Market level - breadth and dispersion (FR-005), under the coverage rule
# ---------------------------------------------------------------------------

BREADTH_STATES = ("Narrow", "Mixed", "Broad", UNKNOWN)


def breadth_state(panel: pd.DataFrame, cov_ok: pd.Series, period: int = 200) -> pd.Series:
    """Share of the universe trading above its own EMA(period).

    Cross-sectional, so this is the highest lookahead risk in the study: any
    operation touching future rows of the panel leaks silently. Everything here
    is column-wise and backward-looking by construction.
    """
    ema = panel.ewm(span=period, adjust=False).mean()
    above = (panel > ema) & panel.notna() & ema.notna()
    share = above.sum(axis=1) / panel.notna().sum(axis=1).replace(0, np.nan)

    out = pd.Series(UNKNOWN, index=panel.index, dtype=object)
    out[share < 0.40] = "Narrow"
    out[(share >= 0.40) & (share <= 0.60)] = "Mixed"
    out[share > 0.60] = "Broad"
    out[share.isna() | ~cov_ok] = UNKNOWN
    return out.rename("breadth")


DISPERSION_STATES = ("Low Dispersion", "Mid Dispersion", "High Dispersion", UNKNOWN)


def dispersion_state(panel: pd.DataFrame, cov_ok: pd.Series, window: int = WINDOW) -> pd.Series:
    """Cross-sectional dispersion of returns - how differently coins are moving."""
    r = np.log(panel).diff()
    disp = r.std(axis=1)
    out = _tercile_state(disp, window, DISPERSION_STATES[:3])
    out[~cov_ok] = UNKNOWN
    return out.rename("dispersion")


CORR_STATES = ("Decoupled", "Mid Correlation", "Coupled", UNKNOWN)


def correlation_state(panel: pd.DataFrame, cov_ok: pd.Series, window: int = 96) -> pd.Series:
    """Median pairwise correlation, proxied by cross-sectional R-squared.

    A true rolling pairwise correlation matrix over 77 columns x 198K bars is
    prohibitively slow. The share of cross-sectional return variance explained
    by the equal-weight market return is a monotone proxy for average pairwise
    correlation and costs one pass.
    """
    r = np.log(panel).diff()
    mkt = r.mean(axis=1)
    # Fraction of each bar's cross-sectional variance explained by the market move.
    num = (mkt ** 2) * panel.notna().sum(axis=1)
    den = (r ** 2).sum(axis=1)

    # A bar where NOTHING moved (den == 0) has undefined correlation - it is
    # skipped, not treated as a value. min_periods stops one such bar from
    # blanking the whole averaging window: without it a single all-flat bar
    # (observed 2026-08-31 03:45, all 75 pairs unchanged) propagated NaN
    # through 96 bars and produced UNKNOWN labels inside W1.
    frac = (num / den.replace(0, np.nan)).rolling(window, min_periods=window // 2).mean()

    out = pd.Series(UNKNOWN, index=panel.index, dtype=object)
    lo = frac.rolling(WINDOW, min_periods=WINDOW // 2).quantile(0.33)
    hi = frac.rolling(WINDOW, min_periods=WINDOW // 2).quantile(0.67)
    out[frac <= lo] = "Decoupled"
    out[(frac > lo) & (frac < hi)] = "Mid Correlation"
    out[frac >= hi] = "Coupled"
    out[frac.isna() | lo.isna() | hi.isna() | ~cov_ok] = UNKNOWN
    return out.rename("correlation")


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOGUE = [
    # BTC level - baseline first
    PhaseInstrument("btc_exp4", "btc", EXP4_STATES, 50, exp4_regime,
                    {"ema": 50, "slope": 24, "adx": 14}, is_baseline=True),
    PhaseInstrument("btc_vol", "btc", ("Low Vol", "Mid Vol", "High Vol", UNKNOWN),
                    WINDOW, btc_vol_state, {"vol_window": 96}),
    # Asset level
    PhaseInstrument("asset_vr4", "asset", VR_STATES, WINDOW,
                    lambda d: vr_state(d, q=4), {"q": 4}),
    PhaseInstrument("asset_vr24", "asset", VR_STATES, WINDOW,
                    lambda d: vr_state(d, q=24), {"q": 24}),
    PhaseInstrument("asset_hurst", "asset", HURST_STATES, WINDOW,
                    hurst_state, {"q_grid": list(Q_GRID)}),
    PhaseInstrument("asset_autocorr", "asset", AC_STATES, WINDOW,
                    autocorr_state, {"lag": 1}),
    PhaseInstrument("asset_adx", "asset", ADX_STATES, 14, asset_adx_state, {"adx": 14}),
    PhaseInstrument("asset_trend", "asset", TREND_STATES, 200,
                    asset_trend_state, {"ema": 200}),
    # Market level
    PhaseInstrument("market_breadth", "market", BREADTH_STATES, 200,
                    breadth_state, {"ema": 200}),
    PhaseInstrument("market_dispersion", "market", DISPERSION_STATES, WINDOW,
                    dispersion_state, {}),
    PhaseInstrument("market_correlation", "market", CORR_STATES, WINDOW,
                    correlation_state, {"corr_window": 96}),
]


def validate_catalogue(catalogue=CATALOGUE) -> None:
    """Enforce the data-model invariants. Raises on violation."""
    names = [i.name for i in catalogue]
    if len(names) != len(set(names)):
        raise ValueError("instrument names must be unique")

    baselines = [i.name for i in catalogue if i.is_baseline]
    if len(baselines) != 1:
        raise ValueError(f"exactly one baseline required, found {baselines}")

    max_horizon = max(Q_GRID)
    for i in catalogue:
        if UNKNOWN not in i.states:
            raise ValueError(f"{i.name}: missing {UNKNOWN} state")
        # Only the VR-derived instruments carry an estimator horizon to compare
        # against; the rest set window_bars to their own natural lookback.
        if i.window_bars == WINDOW and WINDOW < 20 * max_horizon:
            raise ValueError(
                f"{i.name}: window_bars {i.window_bars} < 20x largest horizon {max_horizon}"
            )


def to_frame(catalogue=CATALOGUE) -> pd.DataFrame:
    """The instrument_catalogue.csv schema (contracts/artifacts.md)."""
    import json
    return pd.DataFrame([{
        "name": i.name,
        "level": i.level,
        "states": "|".join(i.states),
        "window_bars": i.window_bars,
        "params": json.dumps(i.params),
        "is_baseline": i.is_baseline,
        "status": i.status,
        "status_reason": i.status_reason,
    } for i in catalogue])


if __name__ == "__main__":
    validate_catalogue()
    df = to_frame()
    print(f"catalogue validated: {len(df)} instruments\n")
    print(df[["name", "level", "window_bars", "is_baseline", "status"]].to_string(index=False))
    print(f"\nlevels: {df.level.value_counts().to_dict()}")
    print(f"baseline: {df[df.is_baseline].name.iloc[0]}")
    print(f"coverage rule: >={MIN_PAIRS} pairs with >={MIN_HISTORY_BARS} bars")
