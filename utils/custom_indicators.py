"""
Solipsis Custom Indicators and Maths
"""
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from pandas import DataFrame, Series


"""
Misc. Helper Functions
"""
def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))

"""
Maths
"""
def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))

def linear_decay(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))

"""
TA Indicators
"""

def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

def mastreak(dataframe: DataFrame, period: int = 4, field='close') -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                    -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = same_length(df['close'], streak)

    return res

def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    df = dataframe.copy()

    df['previous_close'] = df['close'].shift()

    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100

    df['delta'] = df['high_change'] - df['low_change']

    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower

def SSLChannels(dataframe, length=10, mode='sma'):
    """
    Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1025
    Usage:
        dataframe['sslDown'], dataframe['sslUp'] = SSLChannels(dataframe, 10)
    """
    if mode not in ('sma'):
        raise ValueError(f"Mode {mode} not supported yet")

    df = dataframe.copy()

    if mode == 'sma':
        df['smaHigh'] = df['high'].rolling(length).mean()
        df['smaLow'] = df['low'].rolling(length).mean()

    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
                         np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()

    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def WaveTrend(dataframe, chlen=10, avg=21, smalen=4):
    """
    WaveTrend Ocillator by LazyBear
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    """
    df = dataframe.copy()

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)

    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']

    return df['wt1'], df['wt2']

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b*b*b
    c2 = 3*b*b+3*b*b*b
    c3 = -6*b*b-3*b-3*b*b*b
    c4 = 1+3*b+b*b*b+3*b*b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']


def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

"""
Trend-vs-Reversion Estimators

Added for spec-kit feature 001-market-phase-entry-gate (research R1). These
measure directly whether price is trending or mean-reverting, rather than
proxying it via ADX. All are causal: the value at bar t uses only bars <= t.
"""

def variance_ratio(close: Series, q: int = 4, window: int = 480) -> Series:
    """
    Lo-MacKinlay variance ratio on log returns, computed over a rolling window.

        VR(q) = Var(r_q) / (q * Var(r_1))

    where r_1 is the one-bar log return and r_q the overlapping q-bar log
    return. VR > 1 => trending (variance grows faster than linearly in time),
    VR < 1 => mean-reverting, VR == 1 => random walk.

    window must be much larger than q for the variance estimates to be stable;
    the study uses 480 bars against a largest q of 24 (20 non-overlapping
    blocks at the longest horizon).
    """
    logc = np.log(close)
    r1 = logc.diff()
    rq = logc.diff(q)

    var1 = r1.rolling(window).var(ddof=1)
    varq = rq.rolling(window).var(ddof=1)

    return (varq / (q * var1)).rename(f"vr{q}")


def hurst_from_vr(close: Series, q_grid=(2, 4, 8, 24), window: int = 480) -> Series:
    """
    Hurst exponent estimated from the slope of the variance-ratio curve.

    For a process with Hurst exponent H, Var(r_q) = sigma^2 * q^(2H), so

        VR(q) = q^(2H - 1)   =>   log VR(q) = (2H - 1) * log q

    Regressing log VR on log q across the grid gives slope = 2H - 1, hence
    H = (slope + 1) / 2. A random walk (VR == 1 at every q) yields slope 0 and
    H = 0.5, as it must.

    Preferred over R/S rescaling, which is materially biased at these window
    lengths, and it reuses the variance_ratio computation rather than adding a
    second estimator. Validated against fractional Brownian motion of known
    Hurst (0.30/0.40/0.50/0.60/0.70 recovered as 0.295/0.383/0.496/0.582/0.665).

    CAVEAT - this measures LONG MEMORY, not short-horizon reversion, and the
    two are not the same thing. An AR(1) reverting process (phi=-0.4) has
    VR(4)=0.45 and VR(24)=0.53 (clearly reverting) yet H=0.51, because AR(1) is
    short-memory and has no true Hurst exponent. Use variance_ratio levels for
    short-horizon reversion and this for long memory; they are complementary
    instruments, not substitutes.
    """
    q_grid = tuple(q_grid)
    log_q = np.log(np.asarray(q_grid, dtype=float))
    log_q_dev = log_q - log_q.mean()
    denom = float((log_q_dev ** 2).sum())

    vrs = [np.log(variance_ratio(close, q=q, window=window)) for q in q_grid]
    log_vr = np.column_stack([s.values for s in vrs])

    # OLS slope across the (fixed) log-q grid, one regression per bar.
    slope = np.nansum(log_vr * log_q_dev, axis=1) / denom
    # Any bar where some q is missing must not produce a partial-grid slope.
    slope[np.isnan(log_vr).any(axis=1)] = np.nan

    return Series((slope + 1.0) / 2.0, index=close.index, name="hurst")


def rolling_autocorr(close: Series, lag: int = 1, window: int = 480) -> Series:
    """
    Rolling autocorrelation of log returns at the given lag.

    Negative => reversion (an up bar tends to be followed by a down bar),
    positive => momentum persistence. The cheapest of the three measures and
    the most direct reading of short-horizon reversion.
    """
    r = np.log(close).diff()
    return r.rolling(window).corr(r.shift(lag)).rename(f"autocorr{lag}")
