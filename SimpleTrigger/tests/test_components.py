"""T043 — each component neutralises exactly its own keys (023 FR-010, data-model Component)."""

from __future__ import annotations

import pytest
from evaluation import fingerprint as fp
from SimpleTrigger.components import COMPONENTS
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


CFG = fp.load_config(fp.STRAT_DIR / "config" / "config_wf.json")
SECTION_ATTR = {"buy": "BUY_OVERRIDES", "sell": "SELL_OVERRIDES", "attributes": "ATTR_OVERRIDES"}


def _variant(base, comp=None):
    body = {"TRIGGER_NAME": base.TRIGGER_NAME, "PRESET": base.PRESET}
    for section, attr in SECTION_ATTR.items():
        body[attr] = {
            **getattr(base, attr),
            **((comp.overrides.get(section) or {}) if comp else {}),
        }
    if comp and "trigger" in comp.overrides:
        body["TRIGGER_NAME"] = comp.overrides["trigger"]
    return type(f"V_{comp.name if comp else 'full'}_{base.PRESET}", (SimpleTriggerStrategy,), body)


def _flat(resolved):
    return {(s, k): v for s, d in resolved.items() for k, v in d.items()}


def test_every_component_is_well_formed():
    names = [c.name for c in COMPONENTS]
    assert len(names) == len(set(names))
    for c in COMPONENTS:
        assert c.overrides and c.liveness_metric, c.name
        assert set(c.overrides) <= {"trigger", *SECTION_ATTR}, c.name


@pytest.mark.parametrize("preset", ["ARM_D_OPEN", "LEGACY_TRIGGERONLY"])
@pytest.mark.parametrize("comp", COMPONENTS, ids=lambda c: c.name)
def test_neutralising_touches_only_its_own_keys(preset, comp):
    base = type(f"B_{preset}", (SimpleTriggerStrategy,), {"TRIGGER_NAME": "gbb", "PRESET": preset})
    full = _flat(fp.resolve(_variant(base), None, CFG)[1])
    neut = _flat(fp.resolve(_variant(base, comp), None, CFG)[1])
    changed = {k for k in full if full[k] != neut[k]}
    allowed = {
        (s if s != "attributes" else "attributes", k)
        for s, d in comp.overrides.items()
        if s in SECTION_ATTR
        for k in d
    }
    assert changed <= allowed, (comp.name, changed - allowed)
    if "trigger" in comp.overrides:
        assert not changed  # the trigger is not part of the framework
    elif preset in comp.neutral_is_identity_under:
        # already neutral: either identical, or a different value that is open in effect
        assert changed <= allowed
    else:
        assert changed, f"{comp.name} changes nothing under {preset}: its override is inert"


def test_every_guard_is_already_neutral_under_arm_d():
    for c in COMPONENTS:
        if c.name.startswith("guard_"):
            assert "ARM_D_OPEN" in c.neutral_is_identity_under, c.name
