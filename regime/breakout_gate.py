"""FEASIBILITY GATE - does momentum pay in trending phases? (no training)

Constitution VII: validate cheaply before building. Before adding a breakout
TrainingSignal and a training cycle, ask the cheaper question first.

THE HYPOTHESIS (mirror of the Phase C diagnostic). For the DIP model, reverting
states produced the best marginal returns in all three windows. The mirror claim
is that BREAKOUT entries should do best in TRENDING states. If forward returns
after a breakout are no better in trending phases than in reverting ones, the
mirror is false and no label change can rescue it.

WHY THIS IS NOT THE PRIOR BREAKOUT WORK. Breakout labels were built and killed in
2026-07 ("no tradeable long edge at ANY timeframe, with ANY exit", best -1.11%).
But that work gated on BtcRegimeGate uptrend - and Phase B of this study
eliminated exactly that class of instrument (btc_exp4: 48.9% of real trades span
one of its regime changes; asset_adx: 42.2%). Breakout labels have never been
paired with an instrument that survived actionability screening.

METHOD. Price-based, deliberately: this tests the PHENOMENON (does momentum pay
when the market is trending) rather than any particular label form. A label can
be redesigned; a phenomenon that is absent cannot.

    breakout   close > max(close[t-L .. t-1])      new L-bar high
    dip        close < min(close[t-L .. t-1])      new L-bar low
    outcome    forward return at H, and MFE over H (what the labeler scores)

conditioned on the surviving direct instruments (asset_vr24, asset_hurst,
asset_autocorr) at the signal bar, per evaluation window.

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/breakout_gate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime.regimedata import TRADED_PAIRS, UNKNOWN, WINDOWS, load_ohlcv, window_slice

LABELS = Path(__file__).parent / "labels"
OUT_CSV = Path(__file__).parent / "breakout_gate.csv"

L = 48          # breakout lookback, bars (12h at 15m)
H = 48          # forward horizon, matches TrainingConfig.HORIZON
INSTRUMENTS = ["asset_vr24", "asset_hurst", "asset_autocorr"]


def signals_and_outcomes(pair: str) -> pd.DataFrame:
    d = load_ohlcv(pair)
    c = d["close"]
    prior_high = c.shift(1).rolling(L).max()
    prior_low = c.shift(1).rolling(L).min()
    fwd = c.shift(-H) / c - 1.0                      # forward return at H
    mfe = c.shift(-1).rolling(H).max().shift(-(H - 1)) / c - 1.0   # max favourable excursion
    return pd.DataFrame({
        "breakout": c > prior_high,
        "dip": c < prior_low,
        "fwd": fwd,
        "mfe": mfe,
    }, index=c.index)


def main() -> int:
    rows = []
    for pair in TRADED_PAIRS:
        so = signals_and_outcomes(pair)
        lab = pd.read_feather(LABELS / f"asset_{pair}.feather").set_index("date")
        for inst in INSTRUMENTS:
            if inst not in lab.columns:
                continue
            st = lab[inst].astype(str)
            df = so.join(st.rename("state"), how="inner").dropna(subset=["fwd", "mfe"])
            for w in WINDOWS:
                x = window_slice(df, w)
                x = x[x.state != UNKNOWN]
                for kind in ("breakout", "dip"):
                    sel = x[x[kind]]
                    for state, g in sel.groupby("state"):
                        rows.append({"pair": pair, "instrument": inst, "window": w,
                                     "signal": kind, "state": state, "n": len(g),
                                     "fwd_bp": g.fwd.mean() * 1e4,
                                     "mfe_bp": g.mfe.mean() * 1e4})

    df = pd.DataFrame(rows)
    # Pool across pairs, weighting by signal count.
    agg = (df.assign(fw=df.fwd_bp * df.n, mw=df.mfe_bp * df.n)
             .groupby(["instrument", "window", "signal", "state"])
             .agg(n=("n", "sum"), fw=("fw", "sum"), mw=("mw", "sum")).reset_index())
    agg["fwd_bp"] = (agg.fw / agg.n).round(1)
    agg["mfe_bp"] = (agg.mw / agg.n).round(1)
    agg = agg.drop(columns=["fw", "mw"])
    agg.to_csv(OUT_CSV, index=False)

    for inst in INSTRUMENTS:
        sub = agg[agg.instrument == inst]
        if sub.empty:
            continue
        print(f"\n=== {inst} — mean FORWARD RETURN at H={H} bars, in bp ===")
        for kind in ("breakout", "dip"):
            piv = sub[sub.signal == kind].pivot(index="state", columns="window",
                                                values="fwd_bp")
            cnt = sub[sub.signal == kind].pivot(index="state", columns="window", values="n")
            print(f"\n  -- {kind.upper()} signals --")
            print("  " + piv.to_string().replace("\n", "\n  "))
            print("  n: " + " ".join(f"{w}={int(cnt[w].sum())}" for w in cnt.columns))

    # The decisive contrast, on the primary instrument.
    print("\n" + "=" * 70)
    print("GATE: does BREAKOUT pay more in trending states than reverting ones?")
    sub = agg[(agg.instrument == "asset_vr24") & (agg.signal == "breakout")]
    piv = sub.pivot(index="state", columns="window", values="fwd_bp")
    if {"Trending", "Reverting"} <= set(piv.index):
        d = piv.loc["Trending"] - piv.loc["Reverting"]
        print(f"\n  Trending - Reverting (bp), breakout signals: "
              + "  ".join(f"{w}={d[w]:+.1f}" for w in d.index))
        consistent = bool((d > 0).all())
        print(f"\n  VERDICT: {'PASS' if consistent else 'FAIL'} — "
              f"{'positive in all three windows' if consistent else 'not positive in all three windows'}")
        if not consistent:
            print("  The mirror hypothesis is not supported: breakout entries do not\n"
                  "  reliably do better when the phase says the market is trending.")
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
