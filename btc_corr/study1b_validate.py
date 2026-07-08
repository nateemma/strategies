"""Study 1b — validate the cross-sectional reversion edge (Study 1).

No trained params, so the risks are: (1) temporal stability across sub-periods,
(2) survival under REALISTIC turnover-scaled costs (H=4 rebalances a lot),
(3) parameter robustness (L, H), (4) not driven by one coin. Long-only bottom-
quintile rotation throughout.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from btcdata import log_returns, ALTS_1H

TF = "1h"
PERIODS_YR = 24 * 365


def returns_matrix(alts=ALTS_1H):
    return pd.DataFrame({a: log_returns(a, TF) for a in alts}).dropna(how="all")


def rotate(rets, L=24, H=4, cost_side_bp=10.0, ret_idx=None):
    """Long-only bottom-quintile rotation with turnover-scaled cost.
    Returns per-rebalance net returns + mean one-way turnover."""
    sig = rets.rolling(L).sum()
    sig_dm = sig.sub(sig.mean(axis=1), axis=0)
    fwd = rets.rolling(H).sum().shift(-H)
    c = cost_side_bp * 1e-4
    T = len(rets)
    cols = list(rets.columns)
    w_prev = pd.Series(0.0, index=cols)
    nets, turns = [], []
    rebal = range(L, T - H, H) if ret_idx is None else ret_idx
    for t in rebal:
        s = sig_dm.iloc[t].dropna()
        if len(s) < 5:
            continue
        lo = s.index[s.rank(pct=True) <= 0.2]
        w = pd.Series(0.0, index=cols)
        if len(lo):
            w[lo] = 1.0 / len(lo)
        gross = float(fwd.iloc[t][lo].mean()) if len(lo) else 0.0
        traded = float((w - w_prev).abs().sum())      # notional traded (2x one-way turnover)
        nets.append(gross - traded * c)
        turns.append(traded / 2.0)
        w_prev = w
    return np.array(nets), (np.mean(turns) if turns else np.nan)


def stats(r, H):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) < 3:
        return dict(tot=np.nan, sharpe=np.nan, maxdd=np.nan, cal=np.nan)
    eq = np.cumsum(r)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    tot = float(eq[-1]) * 100
    return dict(tot=round(tot, 1), sharpe=round(r.mean() / (r.std() + 1e-12) * np.sqrt(PERIODS_YR / H), 2),
                maxdd=round(dd * 100, 1), cal=round(tot / abs(dd * 100), 2) if dd else np.nan)


if __name__ == "__main__":
    rets = returns_matrix()
    print("===== Study 1b: validation of cross-sectional reversion (bottom-Q rotation) =====\n")

    print("[1] TEMPORAL STABILITY — L24/H4, cost 10bp/side, 6 sequential sub-periods:")
    nets, turn = rotate(rets, 24, 4, 10.0)
    k = len(nets) // 6
    for i in range(6):
        seg = nets[i*k:(i+1)*k] if i < 5 else nets[i*k:]
        st = stats(seg, 4)
        print(f"   period {i+1}: tot={st['tot']:+7}%  sharpe={st['sharpe']:+.2f}  maxdd={st['maxdd']}%")
    full = stats(nets, 4)
    print(f"   FULL     : tot={full['tot']:+}%  sharpe={full['sharpe']}  maxdd={full['maxdd']}%  "
          f"calmar={full['cal']}  (mean 1-way turnover/rebal={turn:.2f})")

    print("\n[2] REALISTIC COST SWEEP — L24/H4 (per-side bp; turnover-scaled):")
    for c in [0, 5, 10, 20, 30, 50, 80]:
        st = stats(rotate(rets, 24, 4, float(c))[0], 4)
        print(f"   cost={c:2d}bp: tot={st['tot']:+8}%  sharpe={st['sharpe']:+.2f}  calmar={st['cal']}")

    print("\n[3] PARAMETER ROBUSTNESS — net@10bp, tot% (sharpe):")
    print("        H=2        H=4        H=8")
    for L in [12, 24, 48]:
        row = f"   L={L:2d} "
        for H in [2, 4, 8]:
            st = stats(rotate(rets, L, H, 10.0)[0], H)
            row += f" {st['tot']:+7}({st['sharpe']:+.2f})"
        print(row)

    print("\n[4] LEAVE-ONE-OUT — L24/H4 net@10bp total% dropping each alt (is one coin driving it?):")
    base = stats(rotate(rets, 24, 4, 10.0)[0], 4)["tot"]
    loo = []
    for a in ALTS_1H:
        sub = returns_matrix([x for x in ALTS_1H if x != a])
        loo.append((a, stats(rotate(sub, 24, 4, 10.0)[0], 4)["tot"]))
    for a, t in sorted(loo, key=lambda x: x[1]):
        print(f"   drop {a:5s}: tot={t:+8}%   (Δ vs {base:+.0f} = {t-base:+.0f})")
