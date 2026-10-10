"""Framework components and their NEUTRAL values (feature 023 US2, FR-010).

Each component can be neutralised alone -- switched off or opened fully -- and the change must
touch ONLY the keys listed in its `overrides` (asserted by tests/test_components.py on the
framework fingerprint). `liveness_metric` names the recorded measurement that must move for the
neutralisation to count as LIVE (constitution IV); an arm whose trades do not change is INERT, a
defect, never "no effect".

"Already neutral" includes OPEN-IN-EFFECT values: close_norm and guard_metric are bounded to
[-1, 1], so LEGACY's 1.1 and ARM_D's 5.0 are both fully open even though they differ.

Override keys were checked against BaseStrategy by name (a misspelt key is silently inert) --
and are re-checked by the fingerprint test, which fails if an override changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ARM_D = "ARM_D_OPEN"
LEGACY = "LEGACY_TRIGGERONLY"


@dataclass(frozen=True)
class Component:
    name: str
    overrides: dict
    liveness_metric: str
    neutral_is_identity_under: tuple = field(default=())


COMPONENTS = (
    # --- entry: the trigger itself and each quality guard ---------------------------------
    Component("trigger", {"trigger": "empty"}, "entry_condition_counts.signal"),
    Component(
        "guard_adx",
        {"buy": {"entry_adx_threshold": -1.0}},
        "entry_condition_counts.adx",
        (ARM_D, LEGACY),
    ),
    Component(
        "guard_atr_pct",
        {"buy": {"entry_atr_pct": -1.0}},
        "entry_condition_counts.atr_pct",
        (ARM_D, LEGACY),
    ),
    Component(
        "guard_bb_width",
        {"buy": {"entry_bb_width_threshold": -1.0}},
        "entry_condition_counts.bb_width",
        (ARM_D, LEGACY),
    ),
    Component(
        "guard_close_norm",
        {"buy": {"entry_close_norm_threshold": 5.0}},
        "entry_condition_counts.close_norm",
        (ARM_D, LEGACY),
    ),
    Component(
        "guard_guard",
        {"buy": {"entry_guard_threshold": 5.0}},
        "entry_condition_counts.guard",
        (ARM_D, LEGACY),
    ),
    Component(
        "guard_rvol",
        {"buy": {"entry_rvol_threshold": -1.0}},
        "entry_condition_counts.rvol",
        (ARM_D, LEGACY),
    ),
    # --- exits ---------------------------------------------------------------------------
    Component(
        "take_profit", {"sell": {"cexit_take_profit": 10.0}}, "exit_reason_counts.take_profit"
    ),
    Component("max_hold", {"sell": {"cexit_max_days": 3650}}, "exit_reason_counts.max_hold"),
    Component("roi", {"attributes": {"minimal_roi": {"0": 10.0}}}, "exit_reason_counts.roi"),
    Component(
        "exit_signal",
        {"sell": {"enable_exit_signal": False}},
        "exit_reason_counts.exit_signal",
        (ARM_D, LEGACY),
    ),
    # --- stop ----------------------------------------------------------------------------
    Component(
        "stop_atr_band",
        {"attributes": {"use_atr_adaptive_stoploss": False}},
        "trades_hash",
        (LEGACY,),
    ),
    Component("stop_grace", {"sell": {"stoploss_grace_hours": 0.0}}, "trades_hash"),
    Component(
        "stop_volume_confirm",
        {"attributes": {"use_volume_confirmation_stoploss": False}},
        "trades_hash",
    ),
)
