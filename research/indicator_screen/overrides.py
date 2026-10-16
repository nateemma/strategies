"""037 raw-signal rules (research R1 table). Every rule calls the strategy's OWN code; none
re-implements a formula (FR-002). Raw = before every family filter (FR-003): the family's
`get_entry_conditions` is never called, and `atr_ok` is forced True by the harvest.

Three classes filter inside their signal code, and get a declared override:
- EhlersBase family: the cycle/trend vote, strict regions and atr_ok are inside
  `get_entry_signals`. Raw signals = the class's own four trigger functions, called after
  `get_entry_signals` has built the Ehlers columns. The as-built (filtered) columns are dropped.
- BBBreakout: `trend_mode` selects which band crossing fires. Raw = its own signal code run with
  the trend_mode column forced to 1 (`tm1`) and to 0 (`tm0`).
- Ichimoku: the cloud-thickness and higher-timeframe switches are turned off on the instance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: as-built Ehlers columns that carry the vote / strict-region / atr_ok filters -- not raw
EHLERS_FILTERED = ("ehler_buy", "ehler_sell", "cycle_buy", "trend_buy", "cycle_sell",
                   "trend_sell", "ehler_cycle", "ehler_trend")


def _arr(x, n: int) -> np.ndarray:
    a = np.asarray(x.to_numpy() if isinstance(x, pd.Series) else x)
    if a.shape != (n,):
        raise ValueError(f"signal shape {a.shape} != ({n},)")
    return (np.nan_to_num(a.astype(float), nan=0.0) > 0).astype(np.int8)


def is_ehlers(strat) -> bool:
    return any(k.__name__ == "EhlersBase" for k in type(strat).__mro__)


def kind(strat) -> str:
    name = type(strat).__name__
    if is_ehlers(strat):
        return "ehlers_triggers"
    if name == "BBBreakout":
        return "force_trend_mode"
    if name == "Ichimoku":
        return "ichimoku_switches"
    return "plain"


def plain(strat, d: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(d)
    buy = _arr(strat.get_entry_signals(d), n)
    sell = _arr(strat.get_exit_signals(d), n)   # exits may read columns the entry call built
    return {"buy": buy, "sell": sell}


def ehlers_triggers(strat, d: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(d)
    strat.get_entry_signals(d)                  # builds the Ehlers columns (its own code)
    out = {"buy:cycle": _arr(strat.get_cycle_entry_trigger(d), n),
           "buy:trend": _arr(strat.get_trend_entry_trigger(d), n),
           "sell:cycle": _arr(strat.get_cycle_exit_trigger(d), n),
           "sell:trend": _arr(strat.get_trend_exit_trigger(d), n)}
    d.drop(columns=[c for c in EHLERS_FILTERED if c in d.columns], inplace=True)
    return out


def force_trend_mode(strat, d: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(d)
    out = {}
    original = d["trend_mode"].copy()
    for v in (1, 0):
        d["trend_mode"] = v
        out[f"buy:tm{v}"] = _arr(strat.get_entry_signals(d), n)
        out[f"sell:tm{v}"] = _arr(strat.get_exit_signals(d), n)
    d["trend_mode"] = original                  # value columns keep the real trend_mode
    return out


def ichimoku_switches(strat, d: pd.DataFrame) -> dict[str, np.ndarray]:
    strat.entry_use_cloud_thickness.value = False
    strat.entry_use_htf_filter.value = False
    return plain(strat, d)


RULES = {"plain": plain, "ehlers_triggers": ehlers_triggers,
         "force_trend_mode": force_trend_mode, "ichimoku_switches": ichimoku_switches}


def raw_signals(strat, d: pd.DataFrame) -> dict[str, np.ndarray]:
    """{"buy[:variant]" | "sell[:variant]": int8 array}. Mutates `d` (adds the class's columns)."""
    return RULES[kind(strat)](strat, d)
