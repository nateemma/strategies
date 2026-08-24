"""Generate synthetic pump-then-die coins to bound survivorship bias.

Profile calibrated on real P1-era pumps (SHIB 12.8x then to 9% of peak; APE 2.5x
then 14%; GALA to 6%). Crucially VOLUME TRACKS THE PUMP -- a dead coin that never
trades is harmless to the strategy, so a toothless volume profile would make this
test prove nothing. Volume rises through the run-up, spikes at the collapse, then
decays to near-zero, which is what traps a position.
"""
import sys, shutil
import numpy as np, pandas as pd
from pathlib import Path

REAL = Path('/Users/philprice95/projects/freqtrade/user_data/data/binanceus')

def make_coin(rng, index, med_qv, mode='soft'):
    n = len(index)
    list_i   = int(rng.uniform(0.02, 0.45) * n)      # when it lists
    runup    = int(rng.uniform(0.06, 0.22) * n)      # weeks of pump
    # 'soft' = decay over weeks (exit signal can fire). 'hard' = rug: 1-3 candles,
    # which momentum CANNOT outrun -- the adversarial bound.
    collapse = (int(rng.uniform(0.04, 0.15) * n) if mode == 'soft'
                else int(rng.integers(1, 4)))
    peak_x   = float(rng.uniform(4.0, 25.0))         # run-up multiple
    floor_x  = float(rng.uniform(0.02, 0.12))        # fraction of peak it dies at

    price = np.full(n, np.nan)
    p0 = float(rng.uniform(1e-5, 5.0))
    a, b = list_i, min(list_i + runup, n)
    price[a:b] = p0 * np.exp(np.linspace(0, np.log(peak_x), b - a))
    if mode == 'hard':
        # single-candle gap straight to the floor: momentum cannot exit ahead of it
        c = min(b + 1, n)
        if c > b:
            price[b:c] = price[b-1] * floor_x
    else:
        c = min(b + collapse, n)
        if c > b:
            price[b:c] = price[b-1] * np.exp(np.linspace(0, np.log(floor_x), c - b))
    if c < n:
        price[c:] = price[c-1]
    noise = np.exp(rng.normal(0, 0.02, n)); noise[:list_i] = 1.0
    price = price * noise

    # volume: tracks the pump, spikes on collapse, decays to dust afterwards
    vol_q = np.zeros(n)
    vol_q[a:b] = med_qv * np.linspace(0.5, 8.0, b - a)
    if c > b: vol_q[b:c] = med_qv * np.linspace(12.0, 1.5, c - b)
    if c < n: vol_q[c:] = med_qv * np.exp(np.linspace(np.log(1.0), np.log(0.02), n - c))
    vol_q *= np.exp(rng.normal(0, 0.6, n))
    vol_q[:list_i] = 0.0

    df = pd.DataFrame({'date': index, 'close': price})
    df['open'] = df.close.shift(1).fillna(df.close)
    df['high'] = df[['open','close']].max(axis=1) * (1 + np.abs(rng.normal(0, .004, n)))
    df['low']  = df[['open','close']].min(axis=1) * (1 - np.abs(rng.normal(0, .004, n)))
    df['volume'] = np.where(df.close > 0, vol_q / df.close.replace(0, np.nan), 0.0)
    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df[['date','open','high','low','close','volume']]

def main(n_dead, seed, outdir, mode='soft'):
    import json
    carriers = json.load(open(f'{Path(outdir).parent}/safe_carriers.json'))
    out = Path(outdir); 
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in REAL.glob('*.feather'):          # symlink real data, don't copy
        (out / f.name).symlink_to(f)
    idx = pd.to_datetime(pd.read_feather(REAL/'BTC_USDT-15m.feather', columns=['date']).date, utc=True)
    med_qv = 5000.0
    rng = np.random.default_rng(seed)
    names = []
    for i in range(n_dead):
        pair = carriers[i]                      # a REAL exchange market (passes validation)
        nm = pair.split('/')[0]
        d15 = make_coin(rng, idx, med_qv * float(rng.uniform(0.3, 4.0)), mode)
        t15 = out / f'{nm}_USDT-15m.feather'
        # NEVER write through a symlink -- that would clobber the real data file
        assert not t15.is_symlink(), f'{t15} is a symlink to real data; refusing to write'
        d15.to_feather(t15)
        dd = d15.set_index('date').resample('1D').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
        t1d = out / f'{nm}_USDT-1d.feather'
        assert not t1d.is_symlink(), f'{t1d} is a symlink to real data; refusing to write'
        dd.to_feather(t1d)
        names.append(pair)
    print(' '.join(names))

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else 'soft')
