"""US1 -- is the repeat signal anything more than "this position is underwater"? (T017-T023)

THE QUESTION. Trades that received a repeat entry signal did worse (win 0.783 vs 0.635,
Fisher p = 0.0049). But if the model re-fires mainly after price has fallen, "received a repeat
signal" is a proxy for unrealised loss, and losing trades ending badly is close to tautological.
Nothing downstream of this module is evidence until that is excluded.

THE MATCHING RULE MATTERS MORE THAN THE TEST. Controls are positions ALIVE AT THE SAME HOLDING
AGE, compared at that same age. Drawing controls from all trades would compare different holding
durations -- and since trades that survive longer are systematically different, that alone would
manufacture an effect.

TWO EFFECT SIZES, ONE GATE. The tests measure a WIN-RATE difference; the study's resolution limit
of 7.633 pp is an ANNUALISED RETURN quantity. The motivating effect is 14.8 win-rate points,
which clears 7.633 numerically while carrying no return information at all. `verdict()` gates on
`effect_return_pp_yr` and RAISES if handed the win-rate figure.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import NOISE_FLOOR_PP, OUT, STAKE_PP_PER_RATIO
from . import paths as P
from . import signals as S

BARS_PER_YEAR = 96 * 365


def repeat_events(fold_id: str, panels: dict) -> pd.DataFrame:
    """Every trade, with the first repeat signal that arrived while it was open."""
    tr = P.trades(fold_id)
    sig = S.signal_bars(fold_id)
    rows = []
    for _, t in tr.iterrows():
        pan = panels.get(t["pair"])
        if pan is None:
            continue
        path = P.unrealised(t.to_dict(), pan)
        if path.empty:
            continue
        bars = sorted(b for b in sig.get(t["pair"], ())
                      if t["open_date"] < b <= t["close_date"])
        rec = {"fold_id": fold_id, "pair": t["pair"], "open_date": t["open_date"],
               "duration_bars": len(path), "profit_ratio": float(t["profit_ratio"]),
               "win": bool(t["profit_ratio"] > 0), "path": path}
        if bars:
            b = bars[0]
            near = path.index[path.index.get_indexer([b], method="nearest")[0]]
            rec.update(repeat=True, repeat_bar=b,
                       repeat_age=int(path.index.get_loc(near)),
                       unrealised_at_repeat=float(path.loc[near]))
        else:
            rec.update(repeat=False, repeat_bar=pd.NaT, repeat_age=-1,
                       unrealised_at_repeat=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def matched(df: pd.DataFrame) -> pd.DataFrame:
    """FR-002 -- for each repeat event at age a, controls are trades ALIVE at age a with no
    repeat by then, measured at their OWN age a."""
    out = []
    reps = df[df.repeat]
    for _, r in reps.iterrows():
        a = r["repeat_age"]
        out.append({"group": "repeat", "age": a, "unrealised": r["unrealised_at_repeat"],
                    "win": r["win"], "profit_ratio": r["profit_ratio"]})
        pool = df[(df.duration_bars > a) & (~df.repeat | (df.repeat_age > a))]
        for _, c in pool.iterrows():
            if c["open_date"] == r["open_date"] and c["pair"] == r["pair"]:
                continue
            out.append({"group": "control", "age": a,
                        "unrealised": float(c["path"].iloc[a]),
                        "win": c["win"], "profit_ratio": c["profit_ratio"]})
    return pd.DataFrame(out)


def strata(m: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    m = m.copy()
    m["stratum"] = pd.qcut(m["unrealised"], q=n_bins, labels=False, duplicates="drop")
    return m


def cmh(m: pd.DataFrame) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel across unrealised-P&L strata (primary test)."""
    num = den = 0.0
    for _, g in m.groupby("stratum"):
        a = float(((g.group == "repeat") & g.win).sum())
        b = float(((g.group == "repeat") & ~g.win).sum())
        c = float(((g.group == "control") & g.win).sum())
        d = float(((g.group == "control") & ~g.win).sum())
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a - (a + b) * (a + c) / n
        den += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 0
    if den <= 0:
        return float("nan"), float("nan")
    chi = (abs(num) - 0.5) ** 2 / den
    return float(chi), float(stats.chi2.sf(chi, 1))


def logit(m: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    X = sm.add_constant(pd.DataFrame({"unrealised": m["unrealised"],
                                      "repeat": (m.group == "repeat").astype(float)}))
    try:
        r = sm.Logit(m["win"].astype(float), X).fit(disp=0)
        return {"coef": float(r.params["repeat"]), "p": float(r.pvalues["repeat"]),
                "ci": [float(x) for x in r.conf_int().loc["repeat"]]}
    except Exception as e:
        return {"coef": float("nan"), "p": float("nan"), "ci": [None, None], "error": str(e)}


def effect_sizes(m: pd.DataFrame, span_years: float) -> dict:
    """FR-007 -- BOTH currencies, explicitly separate."""
    rep, ctl = m[m.group == "repeat"], m[m.group == "control"]
    win_pp = (rep.win.mean() - ctl.win.mean()) * 100.0
    d_ratio = rep.profit_ratio.mean() - ctl.profit_ratio.mean()
    n_affected_per_year = len(rep) / span_years if span_years > 0 else 0.0
    return {"effect_winrate_pp": float(win_pp),
            "effect_return_pp_yr": float(d_ratio * STAKE_PP_PER_RATIO * n_affected_per_year),
            "d_mean_profit_ratio": float(d_ratio),
            "affected_per_year": float(n_affected_per_year)}


def verdict(res: dict, key: str = "effect_return_pp_yr") -> str:
    """Gates on the RETURN consequence only."""
    if key != "effect_return_pp_yr":
        raise ValueError(
            f"the {NOISE_FLOOR_PP} pp floor is an annualised-RETURN quantity and must not be "
            f"applied to {key!r}; win-rate points are a different currency (FR-007)")
    if not np.isfinite(res.get("cmh_p", np.nan)) or res["cmh_p"] >= 0.05:
        return "proxy_for_drawdown"
    if abs(res[key]) < NOISE_FLOOR_PP:
        return "proxy_for_drawdown"          # real but unresolvable is not a finding
    return "survives"
