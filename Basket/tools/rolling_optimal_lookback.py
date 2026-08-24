"""Gate for a DYNAMIC MOM_LOOKBACK_DAYS: is the optimal lookback trackable?

Vectorised daily proxy of the cross-sectional strategy (same causal structure as
_compute_xs: known=shift(1), regime=BTC>SMA100, trend=>SMA50, top-N rank) evaluated
for every lookback 5..60. Then, over rolling windows, ask which lookback WOULD have
been best, and whether that is predictable.

VERDICT (2026-08-24): NO. See the strategy docstring. Kept so the negative result
is reproducible and nobody re-proposes a dynamic lookback without re-running it.

    PYTHONPATH=. .venv/bin/python user_data/strategies/Basket/tools/rolling_optimal_lookback.py
"""
import pandas as pd, numpy as np, rapidjson
from pathlib import Path

CFG = 'user_data/strategies/config/config_mom_15m.json'
DATA = Path('user_data/data/binanceus')
TOP_N, LBS = 3, list(range(5, 61))


def proxy_returns():
    cfg = rapidjson.load(open(CFG), parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
    out = {}
    for p in cfg['exchange']['pair_whitelist']:
        f = DATA / f"{p.split('/')[0]}_USDT-1d.feather"
        if f.exists():
            d = pd.read_feather(f); d['date'] = pd.to_datetime(d.date, utc=True)
            out[p] = d.set_index('date')['close']
    Pd = pd.DataFrame(out).sort_index()
    known = Pd.shift(1); btc = known['BTC/USDT']
    regime = btc > btc.rolling(100).mean()
    trend = known > known.rolling(50).mean()
    fwd = Pd.pct_change().shift(-1)
    rets = {}
    for lb in LBS:
        mom = known / known.shift(lb) - 1
        mem = (mom.rank(axis=1, ascending=False, method='first') <= TOP_N) & trend
        mem = mem.apply(lambda c: c & regime)
        w = mem.div(mem.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        rets[lb] = (w * fwd).sum(axis=1)
    R = pd.DataFrame(rets)
    return R.loc[R.index >= Pd.index[0] + pd.Timedelta(days=200)].fillna(0.0)


def main():
    R = proxy_returns()
    print(f"{'window':>7} {'n':>6} {'corr(now,next)':>15} {'MAE adaptive':>13} {'MAE const':>10}  verdict")
    for win in (60, 90, 180, 365):
        cum = R.rolling(win).sum().dropna(how='all')
        cum = cum[cum.notna().all(axis=1)]
        b = cum.idxmax(axis=1).astype(float)
        nxt = b.shift(-win).dropna(); cur = b.reindex(nxt.index)
        ma = (nxt - cur).abs().mean(); mc = (nxt - b.median()).abs().mean()
        print(f"{win:>7} {len(nxt):>6} {cur.corr(nxt):>+15.3f} {ma:>13.1f} {mc:>10.1f}  "
              f"{'adaptive wins' if ma < mc else 'CONSTANT WINS'}")


if __name__ == '__main__':
    main()
