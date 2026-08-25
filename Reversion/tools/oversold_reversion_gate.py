"""Gate for an absolute-oversold reversion book: decay curve + signal selection.

Answers two questions before any strategy is written:
  1. how fast does the signal decay (=> hold length => cost tolerance), and is the
     edge an illiquidity artifact (the failure mode that killed study1_xsec)?
  2. does any causal filter materially raise the forward return per signal?

See ../README.md for the verdict. Run:
    PYTHONPATH=. .venv/bin/python user_data/strategies/Reversion/tools/oversold_reversion_gate.py
"""
import numpy as np, pandas as pd, rapidjson
from pathlib import Path

DATA = Path('user_data/data/binanceus')
CFG = 'user_data/strategies/config/config_mom_15m.json'
HORIZONS = [4, 8, 12, 24, 48, 72, 96]
N_LIQUID = 15


def rsi(c, n=14):
    d = c.diff()
    ru = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    rd = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def hourly(sym):
    d = pd.read_feather(DATA / f'{sym}_USDT-15m.feather')
    d['date'] = pd.to_datetime(d.date, utc=True)
    return d.set_index('date').resample('1h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()


def collect(pairs):
    """Per-signal features (all causal) + forward returns."""
    recs = []
    for p in pairs:
        f = DATA / f"{p.split('/')[0]}_USDT-15m.feather"
        if not f.exists():
            continue
        h = hourly(p.split('/')[0])
        if len(h) < 3000:
            continue
        r = rsi(h.close)
        sma50 = h.close.rolling(24 * 50).mean()
        sig = (r < 30) & (r.shift(1) >= 30)          # fresh cross, closed bars only
        d = pd.DataFrame({'rsi': r, 'd_sma50': h.close / sma50 - 1,
                          'qv': h.volume * h.close}, index=h.index)
        for hz in HORIZONS:
            d[f'f{hz}'] = h.close.shift(-hz) / h.close - 1
        d = d[sig].copy(); d['pair'] = p
        recs.append(d)
    X = pd.concat(recs).reset_index().rename(columns={'index': 'ts', 'date': 'ts'})
    X['ts'] = pd.to_datetime(X['ts'], utc=True)
    return X


def main():
    cfg = rapidjson.load(open(CFG), parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
    wl = cfg['exchange']['pair_whitelist']
    X = collect(wl)
    liq = X.groupby('pair').qv.median().sort_values(ascending=False)
    liquid = set(liq.head(N_LIQUID).index)
    print(f"signals {len(X):,} | liquid-{N_LIQUID}: "
          f"{sorted(p.split('/')[0] for p in liquid)}\n")
    print(f"{'horizon':>8}{'ALL (bp)':>14}{'LIQUID (bp)':>16}")
    for hz in HORIZONS:
        print(f"{hz:>6}h{X[f'f{hz}'].mean()*1e4:>14.1f}"
              f"{X[X.pair.isin(liquid)][f'f{hz}'].mean()*1e4:>16.1f}")
    L = X[X.pair.isin(liquid)]
    print("\nfwd-48h by distance-below-SMA50 quintile (median bp):")
    q = pd.qcut(L.d_sma50, 5, duplicates='drop')
    print((L.groupby(q, observed=True).f48.median() * 1e4).to_string(float_format=lambda x: f"{x:,.0f}"))
    deep = L[L.d_sma50 < -0.20]
    print(f"\nFILTER RSI<30 AND >20% below SMA50: n={len(deep):,} "
          f"median f48 {deep.f48.median()*1e4:+.0f}bp  win {(deep.f48>0).mean():.0%}")


if __name__ == '__main__':
    main()
