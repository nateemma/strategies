"""Coupled exit mode: take-profit, max hold and stop DERIVED from the label's targets (023 R8).

Decoupled (the default, today's behaviour): the exit's take-profit and stop are set
independently of the label the trigger came from. Coupled: they are derived from the label's own
gain/loss targets, so the trade is managed toward the outcome the label was defined by. Coupling
has caused real defects here (label/exit disagreement is a recorded reason the label and the
trade diverge), which is why this exists as a switch to MEASURE, not a default.

A PURE PARAMETER MAPPING (FR-016). This module only produces values; they flow through the one
exit definition (`BaseStrategy._exit_params` -> `utils.ExitPolicy.ExitParams`). There is no second
exit implementation to drift.

    exit field                 coupled value
    cexit_take_profit (NET)    net_from_price_ratio(1 + min_gain, fee_per_side)   -- the label's
                               min_gain is a GROSS price move; the exit target is NET of fees,
                               so the unit conversion is explicit, never implied
    cexit_max_days             horizon * timeframe, which must be WHOLE days (the parameter is an
                               integer); a non-integral horizon is refused, not rounded
    stop                       -loss as a static stop with the ATR stop OFF, ONLY for a label
                               whose buy side has a loss barrier (forward_mae's max_drawdown,
                               triple_barrier's sl); otherwise unchanged and recorded as
                               "none-in-label"

The class declares the targets explicitly -- `LABEL_TARGETS = {"min_gain": .., "horizon": ..,
"loss": ..}` -- rather than this module reading a labeler's defaults, so a run record states
exactly which targets were coupled.
"""

from __future__ import annotations

from Framework.TrainingSignals import LabelMethod
from SimpleTrigger.label_map import MAPPING, UNSUPPORTED
from SimpleTrigger.presets import PresetError
from utils.ExitPolicy import net_from_price_ratio


#: Labels whose BUY side has a loss barrier, and the labeler argument that sets it.
LOSS_BARRIER = {LabelMethod.forward_mae: "max_drawdown", LabelMethod.triple_barrier: "sl"}

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

DEFAULT_FEE_PER_SIDE = 0.001  # utils.ExitPolicy.ExitParams.fee_per_side default


def resolve(cls) -> dict:
    """{"sell": {...}, "attributes": {...}, "record": {...}} for a coupled class."""
    method = getattr(cls, "LABEL_METHOD", None)
    if method is None:
        raise PresetError("EXIT_MODE='coupled' requires LABEL_METHOD")
    row = MAPPING[method]
    if row.status == UNSUPPORTED:
        raise PresetError(f"{method.name} is unsupported ({row.reason}); nothing to couple to")
    t = dict(getattr(cls, "LABEL_TARGETS", {}) or {})
    for k in ("min_gain", "horizon"):
        if k not in t:
            raise PresetError(f"coupled mode needs LABEL_TARGETS[{k!r}]")
    fee = float(getattr(cls, "FEE_PER_SIDE", DEFAULT_FEE_PER_SIDE))
    minutes = _TF_MINUTES[getattr(cls, "timeframe", "15m")]
    total = int(t["horizon"]) * minutes
    if total % 1440:
        raise PresetError(
            f"horizon {t['horizon']} x {minutes}m = {total / 1440:.4f} days; "
            f"cexit_max_days is whole days, and rounding would change the target"
        )
    # Rounded to 1e-12 -- far below any price tick -- so a declared round trip (min_gain chosen
    # with price_ratio_for_net) lands EXACTLY on the decoupled value instead of 2e-17 off it,
    # which would change the framework fingerprint while changing no trade.
    sell = {
        "cexit_take_profit": round(net_from_price_ratio(1.0 + float(t["min_gain"]), fee), 12),
        "cexit_max_days": total // 1440,
    }
    attrs: dict = {}
    loss = t.get("loss")
    if loss is not None:
        if method not in LOSS_BARRIER:
            raise PresetError(
                f"{method.name} has no loss barrier on its buy side; a `loss` "
                f"target would be coupled to nothing"
            )
        attrs = {"stoploss": -abs(float(loss)), "use_atr_adaptive_stoploss": False}
    record = {
        "label_method": method.name,
        "targets": t,
        "fee_per_side": fee,
        "take_profit_net": sell["cexit_take_profit"],
        "max_days": sell["cexit_max_days"],
        "stop": attrs.get("stoploss"),
        "coupled_stop": "label" if attrs else "none-in-label",
    }
    return {"sell": sell, "attributes": attrs, "record": record}
