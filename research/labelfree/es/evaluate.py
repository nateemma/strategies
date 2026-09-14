"""Score one candidate on one fold panel (T017 caller, T022).

Aggregates the per-pair surrogate runs into the single number the search ranks on. Pairs are
pooled by concatenating their trades and compounding, matching how the walk-forward harness
pools the 378 archived trades rather than averaging per-pair returns.
"""
from __future__ import annotations

import numpy as np

from research.labelfree.es import exits as EX
from research.labelfree.es import objective as OB
from research.labelfree.es import panel as PA
from research.labelfree.es import space as SP


def run(panel: dict, params: dict) -> EX.TradeSeries:
    """Every pair in the fold, under one candidate's parameters."""
    pooled = EX.TradeSeries()
    n = max(len(d["open"]) for d in panel["pairs"].values())
    occ = np.zeros(n, dtype=bool)
    for pair in sorted(panel["pairs"]):
        d = panel["pairs"][pair]
        ent = PA.entries(d, params)
        sell = PA.exits(d)
        ts = EX.simulate(
            ent, d["open"], d["high"], d["low"],
            np.nan_to_num(d["atr_pct_roll"]), np.nan_to_num(d["rvol_stop"]),
            d["cost"], params, model_exit=sell)
        pooled.entry_bar += ts.entry_bar
        pooled.exit_bar += ts.exit_bar
        pooled.pnl += ts.pnl
        pooled.reason += ts.reason
        pooled.n_bars_total += ts.n_bars_total
        if ts.occupancy is not None:
            occ[:len(ts.occupancy)] |= ts.occupancy
    pooled.occupancy = occ          # PORTFOLIO occupancy — see TradeSeries.mean_exposure
    return pooled


def score(panel: dict, params: dict, incumbent_tpy: float,
          incumbent_exposure: float | None = None) -> OB.Score:
    """The regularised number the search optimises. Inadmissible candidates score -inf."""
    try:
        SP.validate(params)
    except SP.Inadmissible as e:
        return OB.Score(value=float("-inf"), unregularised=0.0, distance_penalty=0.0,
                        turnover_penalty=0.0, admissible=False, reason=str(e),
                        n_trades=0, trades_per_year=0.0, mean_hold_h=0.0, mean_exposure=0.0)
    ts = run(panel, params)
    return OB.score(ts, params, panel["span_years"], incumbent_tpy, incumbent_exposure)


def incumbent_reference(panel: dict) -> tuple[float, float]:
    """(trades/yr, exposure) for the incumbent ON THIS PANEL.

    Both are references the candidate is judged against: turnover for the regularisation
    (data-model §1a), exposure for the participation floor. Computing them per panel matters
    -- a 90-day test range and a 4-year training range give very different exposures for the
    SAME strategy, so a single absolute number cannot serve both.
    """
    ts = run(panel, dict(SP.INCUMBENT))
    tpy = len(ts) / panel["span_years"] if panel["span_years"] > 0 else 0.0
    return tpy, ts.mean_exposure


def incumbent_trades_per_year(panel: dict) -> float:
    return incumbent_reference(panel)[0]
