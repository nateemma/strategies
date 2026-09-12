"""Every swept parameter must provably change what it claims to control.

Guards the defect class that voided 75+ runs in this study: a parameter that is passed,
logged, and believed active while reaching nothing that matters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import liveness as LV      # noqa: E402


@pytest.mark.parametrize("name", sorted(LV.PROBES))
def test_parameter_is_live(name):
    r = LV.check(name)
    assert r["differs"], (
        f"{name} is DEAD: it controls {r['controls']}, but changing it left that "
        f"quantity unchanged at {r['a']!r}. Any sweep over it would be vacuous.")


def test_probe_can_actually_fail(monkeypatch):
    """NEGATIVE CONTROL. A liveness check that cannot report DEAD certifies nothing.

    Sever gamma from the loss the way the real defects were severed -- the parameter is
    still passed and still logged, it just no longer reaches the computation -- and assert
    the probe notices.
    """
    from research.labelfree import arms

    real = arms.ArmConfig

    def gamma_ignored(*a, **kw):                   # gamma passed, logged, and inert
        kw["gamma"] = 1.0                          # ...and silently dropped on the way in
        return real(*a, **kw)

    monkeypatch.setattr(arms, "ArmConfig", gamma_ignored)
    assert LV.check("gamma")["differs"] is False


def test_check_all_raises_on_a_dead_parameter(monkeypatch):
    monkeypatch.setattr(LV, "PROBES", {
        "fake": dict(controls="nothing at all",
                     a={"arm": "pnl_control"}, b={"arm": "pnl_control"},
                     measure=lambda spec: 1.0)})
    with pytest.raises(ValueError, match="is DEAD"):
        LV.check_all()


def test_probe_must_sit_in_its_active_regime():
    """participation_weight scales a FLOOR term that is off while exposure clears it.

    Probed at the default floor it reports DEAD on correct code -- which is why its probe
    pins min_exposure high. This pins the lesson so the probe is not 'simplified' back.
    """
    assert LV.PROBES["participation_weight"]["a"]["min_exposure"] == 0.9
    assert LV.PROBES["participation_weight"]["b"]["min_exposure"] == 0.9
