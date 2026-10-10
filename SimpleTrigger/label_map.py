"""Every LabelMethod -> the causal trigger a SimpleTrigger arm can trade (feature 023, FR-004).

A label is `trigger AND outcome` (feature 021). The outcome always looks forward -- it is what a
label is -- so it can never be an entry rule. The TRIGGER is the part that can, when it is causal.
This table records, for every `LabelMethod` member, which registered trigger carries its entry
condition, or why none does. `SimpleTrigger/tests/test_label_map.py` fails if the enum gains a
member this table does not classify.

STATUS
  supported    the label's entry condition is a registered trigger that PASSES the measured
               causality check (`research/outcomes/trigger_causality.json`).
  empty        a pure-outcome label: no entry condition at all, so its SimpleTrigger arm trades
               the `empty` trigger (every bar). Labelled as such in run records so it cannot be
               mistaken for a selective trigger.
  unsupported  no tradeable entry condition; `reason` says why. Never silently mapped to empty.

Source of each row: `research/outcomes/tests/test_labelmethod_inventory.py` (corrected by 023
T021: the three `future_*` labels reduce to `adx`) and the causality verdicts (023 T020).
"""
from __future__ import annotations

from dataclasses import dataclass

from Framework.TrainingSignals import LabelMethod

SUPPORTED, EMPTY, UNSUPPORTED = "supported", "empty", "unsupported"


@dataclass(frozen=True)
class LabelTrigger:
    label_method: LabelMethod
    trigger: str | None
    status: str
    reason: str = ""


_UNSCOREABLE = "unscoreable: the trigger's required columns are absent from the populator"

MAPPING: dict[LabelMethod, LabelTrigger] = {lt.label_method: lt for lt in (
    # --- pure outcome: the no-trigger case ------------------------------------------------
    LabelTrigger(LabelMethod.forward_mae, "empty", EMPTY, "forward MAE-capped return"),
    LabelTrigger(LabelMethod.triple_barrier, "empty", EMPTY, "forward barriers only"),
    LabelTrigger(LabelMethod.quantile_future, "empty", EMPTY, "forward return quantile"),
    LabelTrigger(LabelMethod.mfe_mae_ratio, "empty", EMPTY, "forward MFE/MAE ratio"),
    LabelTrigger(LabelMethod.risk_adj_future, "empty", EMPTY, "risk-adjusted forward return"),
    LabelTrigger(LabelMethod.multi_horizon_vote, "empty", EMPTY, "vote over forward horizons"),
    LabelTrigger(LabelMethod.optimal_signals, "empty", EMPTY, "oracle over the forward window"),
    LabelTrigger(LabelMethod.local_extrema, "empty", EMPTY, "minimum of a forward window"),
    # --- entry condition is a registered, causal trigger -----------------------------------
    LabelTrigger(LabelMethod.future_sharpe, "adx", SUPPORTED, "ends with adx_scaled > -0.2"),
    LabelTrigger(LabelMethod.future_sortino, "adx", SUPPORTED, "ends with adx_scaled > -0.2"),
    LabelTrigger(LabelMethod.future_expectancy, "adx", SUPPORTED, "ends with adx_scaled > -0.2"),
    LabelTrigger(LabelMethod.gbb, "gbb", SUPPORTED),
    LabelTrigger(LabelMethod.gbb_path, "gbb", SUPPORTED, "gbb + a forward path test"),
    LabelTrigger(LabelMethod.gbb_fwdmae, "gbb", SUPPORTED, "gbb AND forward_mae"),
    LabelTrigger(LabelMethod.breakout, "breakout", SUPPORTED),
    LabelTrigger(LabelMethod.breakout_tb, "breakout", SUPPORTED, "breakout + triple barrier"),
    LabelTrigger(LabelMethod.breakout_vol, "breakout_vol", SUPPORTED),
    LabelTrigger(LabelMethod.breakout_squeeze, "breakout_squeeze", SUPPORTED),
    LabelTrigger(LabelMethod.breakout_bb, "breakout_bb", SUPPORTED),
    LabelTrigger(LabelMethod.breakout_gbb, "breakout_gbb", SUPPORTED),
    LabelTrigger(LabelMethod.breakout_consensus, "votes_consensus", SUPPORTED),
    LabelTrigger(LabelMethod.technical_indicators, "votes_tech", SUPPORTED),
    LabelTrigger(LabelMethod.indicators2, "votes_ind2", SUPPORTED),
    LabelTrigger(LabelMethod.indicators3, "votes_ind3", SUPPORTED),
    LabelTrigger(LabelMethod.indicators4, "votes_ind4", SUPPORTED),
    LabelTrigger(LabelMethod.geometry, "geometry", SUPPORTED,
                 "the registered `geometry` trigger is a CAUSAL SUBSTITUTE, not a port: the "
                 "labeler's own condition reads forward"),
    # --- no tradeable entry condition ------------------------------------------------------
    LabelTrigger(LabelMethod.local_min_followthrough, None, UNSUPPORTED,
                 "centred +/-3-bar local minimum reads 3 bars ahead"),
    LabelTrigger(LabelMethod.trends, None, UNSUPPORTED, _UNSCOREABLE + " (flow/regime/risk/momentum)"),
    LabelTrigger(LabelMethod.bands, None, UNSUPPORTED, _UNSCOREABLE + " (6 of 8 band columns)"),
)}


def trigger_for(method: LabelMethod) -> str:
    """The trigger a SimpleTrigger arm for `method` trades. Raises for an unsupported label."""
    row = MAPPING[method]
    if row.status == UNSUPPORTED:
        raise ValueError(f"{method.name} has no tradeable entry condition: {row.reason}")
    return row.trigger
