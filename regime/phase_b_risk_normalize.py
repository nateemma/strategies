"""Phase B - is the exp4 risk discrimination real, or a time-in-state artifact? (T016)

Contract: specs/001-market-phase-entry-gate/contracts/study-cli.md (FR-012, SC-004)

The prior study reported per-state max drawdown for a passive alt basket:

    Sideways  -118.3%   (29.8% of bars)
    Weak Down  -44.3%
    Strong Down-56.9%
    Strong Up  -38.7%
    Weak Up    -28.1%   (19.2% of bars)

a roughly 4x spread that motivated treating phase as a RISK signal. But the
worst state also holds 1.55x the bars of the best, and drawdown grows with
sample length under any process - roughly as sqrt(t) for a random walk. So part
or all of that spread may be nothing but time-in-state.

METHOD. Reproduce the original figure, then equalise exposure two ways:

  1. sqrt-normalised   maxDD / sqrt(n_bars) - the closed-form correction for
                       random-walk drawdown scaling.
  2. bootstrap         resample a FIXED number of bars from each state, many
                       times, and compare the drawdown distributions. This is
                       the honest version: it holds n identical by construction
                       and needs no assumption about how DD scales.

A CAVEAT ON WHAT THIS CAN NOW DECIDE. Phase B eliminated btc_exp4 (48.9% of real
trades span one of its regime changes), so this analysis can no longer feed the
gate whatever it finds. It settles a question about the PRIOR work - was the
headline risk result real - and that is worth answering on its own terms.

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_b_risk_normalize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "btc_corr"))

from regime.instruments import exp4_regime          # noqa: E402
from regime.regimedata import UNKNOWN               # noqa: E402

OUT_CSV = Path(__file__).parent / "risk_normalized.csv"
ORDER = ["Strong Down", "Weak Down", "Sideways", "Weak Up", "Strong Up"]
N_BOOT = 2000
SEED = 20260902


def max_drawdown(logret: np.ndarray) -> float:
    """Log drawdown of the cumulative return path, as the prior study computed it."""
    eq = np.cumsum(logret)
    return float((eq - np.maximum.accumulate(eq)).min())


def load_1h():
    """BTC regime and the equal-weight alt basket, on the prior study's own 1h data."""
    from btcdata import load_ohlcv, log_returns, ALTS_1H
    reg = exp4_regime(load_ohlcv("BTC", "1h"))
    alt = pd.concat([log_returns(a, "1h").rename(a) for a in ALTS_1H], axis=1)
    fwd = alt.mean(axis=1).shift(-1)                 # next-bar basket return
    df = pd.DataFrame({"regime": reg, "fwd": fwd}).dropna()
    return df[df.regime != UNKNOWN]


def main() -> int:
    rng = np.random.default_rng(SEED)
    df = load_1h()
    groups = {r: df.fwd[df.regime == r].to_numpy() for r in ORDER if (df.regime == r).any()}
    n_min = min(len(v) for v in groups.values())

    rows = []
    for state, v in groups.items():
        raw = max_drawdown(v)
        boot = np.array([max_drawdown(rng.choice(v, n_min, replace=False))
                         for _ in range(N_BOOT)])
        rows.append({
            "state": state,
            "pct_time": round(len(v) / len(df) * 100, 1),
            "n_bars": len(v),
            "maxdd_raw_pct": round(raw * 100, 1),
            "maxdd_sqrt_norm": round(raw / np.sqrt(len(v)) * 100, 4),
            "maxdd_boot_median_pct": round(float(np.median(boot)) * 100, 1),
            "maxdd_boot_p10_pct": round(float(np.quantile(boot, 0.10)) * 100, 1),
            "mean_bp": round(float(v.mean()) * 1e4, 2),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(f"equalised sample size n = {n_min} bars per state, {N_BOOT} bootstrap draws\n")
    print(out.to_string(index=False))

    def spread(col):
        """Worst-state drawdown over best-state drawdown.

        Drawdowns are NEGATIVE, so the WORST state is the column MINIMUM and the
        BEST is the maximum. Taking max/min instead inverts the ratio and turns a
        4.8x spread into 0.21x.
        """
        worst, best = out[col].min(), out[col].max()
        return abs(worst / best) if best else float("inf")

    raw_spread = spread("maxdd_raw_pct")
    boot_spread = spread("maxdd_boot_median_pct")
    print(f"\nraw spread (worst/best)             : {raw_spread:.2f}x")
    print(f"time-equalised spread (bootstrap)   : {boot_spread:.2f}x")

    # CONFIRMED only if the discrimination survives equalising exposure.
    verdict = "CONFIRMED" if boot_spread >= 2.0 else "REFUTED"
    print(f"\nVERDICT: {verdict}")
    if verdict == "REFUTED":
        print("  The apparent risk discrimination is substantially an artifact of\n"
              "  unequal time-in-state. Once every state is given the same number of\n"
              "  bars, the spread collapses.")
    else:
        print("  The risk discrimination survives equalising time-in-state.")
    print(f"\nwrote {OUT_CSV}")
    print("\nNOTE: btc_exp4 was ELIMINATED in Phase B (48.9% of real trades span one\n"
          "of its regime changes), so this result cannot feed the gate either way.\n"
          "It settles whether the prior study's headline risk finding was real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
