"""Entry family implied by a training label (feature 029, decision A′; contract I2).

A model's entries are predictions of its training label, so they take the family the label's
POSITIVES imply -- the market state a positive bar is in, exactly as a trigger's family records
the state its condition buys (`research/outcomes/builtin_triggers.py`). Only labels whose
positives encode an entry state are listed; an outcome-only label (it asks whether price rises,
so a bounce and a breakout both qualify) has no family, and its entries stay untagged, on the
strategy's shared exit.

Each row was checked against its labeler in `Framework/TrainingSignals.py` (029 T013). The family
names are `utils.FamilyExits.FAMILIES`. Strategy-agnostic: no strategy names.
"""

from __future__ import annotations

from Framework.TrainingSignals import LabelMethod
from utils.FamilyExits import FAMILIES


_GBB = "positives require guard_metric < 0.3 with a wide band: a WEAK band position"
_BRK = "positives require a close above the prior lookback high: a NEW HIGH (strength)"

#: {LabelMethod: (family, reason)} -- labels whose positives imply an entry state.
LABEL_FAMILY: dict[LabelMethod, tuple[str, str]] = {
    LabelMethod.gbb: ("reversion", _GBB),
    LabelMethod.gbb_path: ("reversion", _GBB + " (and no stop breach before the target)"),
    LabelMethod.gbb_fwdmae: ("reversion", "gbb's positives AND a forward-MAE cap: " + _GBB),
    LabelMethod.geometry: ("reversion", "positives are local price minima (valleys): buys a dip"),
    LabelMethod.breakout: ("momentum", _BRK),
    LabelMethod.breakout_tb: ("momentum", _BRK + ", held to the horizon"),
    LabelMethod.breakout_vol: ("momentum", _BRK + " on raised relative volume"),
    LabelMethod.breakout_squeeze: ("momentum", _BRK + " out of a band squeeze"),
    LabelMethod.breakout_gbb: (
        "momentum", "positives require guard_metric > 0.2 with a wide band: a STRONG band position"),
    LabelMethod.breakout_bb: (
        "momentum", "positives require a high Bollinger-band position with a wide band (strength)"),
    LabelMethod.breakout_consensus: (
        "momentum", "positives need a majority of strength votes (aroon up, guard >= 0.2, above "
        "VWAP, SAR bullish, trending)"),
}

assert all(f in FAMILIES for f, _ in LABEL_FAMILY.values())


def label_family(training_type) -> str | None:
    """The family a label's positives imply, or None (outcome-only, or unknown)."""
    if training_type is None:
        return None
    try:
        method = LabelMethod(int(training_type))
    except (TypeError, ValueError):
        return None
    row = LABEL_FAMILY.get(method)
    return row[0] if row else None
